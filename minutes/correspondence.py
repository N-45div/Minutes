"""The inbound evidence stream.

Schools report service delivery in a shape that structurally cannot reveal a
shortfall. A progress report narrates the goal and disposes of the service in
one unquantified sentence ("services are being provided as outlined in the
IEP"). A provider's cancellation email names no minutes and keeps no running
total. An SIS export usually contains only the sessions that were *held*, so
every miss is invisible unless someone notices a gap in the dates. What the
parent actually receives is a heap of untyped prose in which the misses are
the negative space.

This module turns that heap into typed ``ServiceEvent`` facts. Five decisions
are deliberately kept away from the model:

* **Provenance is derived from the item, never generated.** A school email, a
  district service log and a progress report are ``SCHOOL_CONFIRMED``; a
  parent's quick log is ``PARENT_OBSERVED``. ``DOCUMENTED_SILENCE`` cannot be
  produced here at all — a classifier reading a document can only report what
  a document says, and silence is by definition not in one. That grade belongs
  to the discovery module, which earns it from an unanswered request.
* **Service names are snapped to the ledger.** A fact naming a service the IEP
  does not promise is dropped rather than carried into reconciliation, because
  a shortfall is only cognizable against the IEP's own stated frequency and
  duration.
* **Calendar arithmetic is precomputed.** Correspondence dates itself in
  fragments — "today", "Tuesday", "10/14". Each item is handed a resolved date
  table so the model looks a date up instead of counting weekdays, which is
  where a small classifier otherwise invents dates.
* **Every date must be grounded in the item's own text** — and grounded to a
  sentence that is actually reporting a session. This is the cite-or-stay-
  silent rule turned on the classifier itself. ``date_is_grounded`` does not
  merely ask whether the date token appears somewhere in the document: the
  line that carries the token must also name a service or an encounter, and
  the sentence that carries it must not disclaim that any session is being
  reported ("no speech *yet*", "she *hasn't* met the speech teacher", "she
  comes to the resource room *daily*"). A weekday name resolves only when it
  resolves to exactly one admissible day, so "Tuesday" written on a Friday —
  which could be either of two Tuesdays — grounds nothing unless the writer
  said which week. An item also cannot report a session that had not happened
  when it was written, and no session lands on a weekend.
* **Every duration must be grounded too.** The same rule applied to numbers.
  A batch is several items in one prompt, and a model reading item A's
  "about fifteen minutes" will occasionally attach it to item B. So a stated
  duration is accepted only when that number is actually written in *that*
  item; otherwise the fact falls back to the IEP's minutes per session. Being
  wrong here is not neutral — an ungrounded short duration silently inflates
  the shortfall asserted against a school.

Facts that disagree across passes are voted, never minimised. Both halves of a
fact — whether the session happened and how long it ran — are decided by the
same majority, because a rule that lets a lone bad reading lose on delivery
but win on duration is a rule biased toward finding a shortfall.

HANDOFF — EXCUSABLE NON-DELIVERY. A ``ServiceEvent`` records that a session
did not happen; it carries no field for *why*, and this module does not
invent one. But some non-deliveries are the child's absence, not the school's
failure, and ``models.Provenance`` says a ``SCHOOL_CONFIRMED`` fact may be
cited in an escalation letter without qualification — so a compensatory
demand could otherwise be built on a session the school could not have
delivered. ``student_absence_events`` names that subset explicitly.
Reconciliation and the letter compiler must call it and qualify or exclude
those events; in the shipped fixture it returns three (2026-10-14 OT, from
both the provider's email and the parent's log, and 2027-01-12 Specialized
Academic Instruction).

The fixture semester under ``fixtures/correspondence/`` is entirely synthetic
and every file carries a marker saying so; ``load_correspondence`` refuses a
file that does not.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

from pydantic import BaseModel, Field
from strands import Agent
from strands.models import BedrockModel

from .config import BEDROCK_REGION, CLASSIFIER_MODEL
from .models import (
    Correspondence,
    CorrespondenceKind,
    IEPLedger,
    Provenance,
    ServiceEvent,
    ServiceObligation,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
CORRESPONDENCE_DIR = FIXTURES / "correspondence"
CACHE_DIR = FIXTURES / "cache"

SYNTHETIC_MARKER = "_synthetic"

# Evidence grade follows from who wrote the item, so it is never asked of the
# model. DOCUMENTED_SILENCE is absent by construction, not by omission.
PROVENANCE_BY_KIND: dict[CorrespondenceKind, Provenance] = {
    CorrespondenceKind.SCHOOL_EMAIL: Provenance.SCHOOL_CONFIRMED,
    CorrespondenceKind.SERVICE_LOG: Provenance.SCHOOL_CONFIRMED,
    CorrespondenceKind.PROGRESS_REPORT: Provenance.SCHOOL_CONFIRMED,
    CorrespondenceKind.PARENT_LOG: Provenance.PARENT_OBSERVED,
}

# Compiled after the fact rather than written on the day, so these carry their
# own issue date in a header and must not be read as reporting a session on it.
RETROSPECTIVE_KINDS = frozenset({CorrespondenceKind.PROGRESS_REPORT, CorrespondenceKind.SERVICE_LOG})

# A service log is one long item that yields dozens of facts; a parent log is
# eight words. Packing by character budget keeps both kinds of batch inside one
# response without ever falling back to one call per item.
BATCH_MAX_ITEMS = 8
BATCH_CHAR_BUDGET = 3500

# Nova Lite's recall varies run to run: it skips items, and not the same ones
# each time. Reading the stack three times and pooling the drafts recovers
# most of that. VOTE_QUORUM is what keeps the extra passes honest — a fact
# only one pass ever saw is one reading, not a vote, and a single reading is
# not enough to assert a missed session against a school.
CLASSIFIER_PASSES = 3
VOTE_QUORUM = 2

# A session can legitimately run long — a make-up doubles up, an evaluation
# session overruns — so duration is not clamped to the IEP's per-session
# figure, only to something a school day could contain.
MAX_SESSION_FACTOR = 4

_WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")

SYSTEM_PROMPT = """You read a parent's inbound correspondence about a child's \
IEP services and report every statement it makes about whether a specific \
service session did or did not happen on a specific date.

Work through every item in the batch in order. Most items produce no events; \
that is normal and expected, but read all of them.

Rules:
- Emit one event per service per date. Never merge dates and never invent one.
- Every event needs a date the item actually points at. Each item carries a \
`dates:` line that resolves "today" and every weekday name for you — read the \
date off that line, never count days yourself. A numeric date such as 10/14 \
takes the year from the school year given above the items. Never emit a date \
the item does not itself name.
- If the item does not point at a day, emit nothing. "She hasn't had speech \
yet", "school started a week ago", "she comes to the resource room daily" and \
"she is receiving all of her services" name no session.
- A progress report narrates goals, not sessions. Unless it states an actual \
session date, it produces no events at all.
- A message announcing that school is CLOSED — snow, holiday, break, staff \
development day — is not about any service. Emit nothing for it, even though \
it names a date. A closed day is not a missed session.
- Copy `service` EXACTLY from the list of IEP services provided. If an item \
describes something that is not one of those services, emit nothing for it.
- delivered=true only where the item says the session took place. A \
cancellation, a "we weren't able to see her", a suspension notice, or a \
parent recording that the child reported no session is delivered=false with \
minutes=0.
- A statement about the future is not a delivery: "services will begin the \
week of September 21", "we will make these up", "sessions may be adjusted \
during testing" — emit nothing.
- If the item records that the child could not remember whether a session \
happened, that is not a statement either way. Emit nothing.
- minutes: use the duration THIS item states, in this item's own words. Items \
in the same batch are unrelated; never carry a duration from one item to \
another. If this item states no duration, use the IEP's minutes per session \
for that service.
- Many items are routine school announcements — picture day, fundraisers, bus \
routes, menus, book fairs. Emit nothing for those.

Report what the document says, not what you conclude from it. You are not \
judging whether anything was owed, whether a miss was excusable, or whether \
anyone was at fault."""


# ---------------------------------------------------------------------------
# Internal transport types.
#
# Neither of these crosses the module boundary: the public surface takes and
# returns models.py types only. They exist because structured output needs a
# schema to fill in, and because that schema must NOT be ServiceEvent — see
# EventDraft's docstring.
# ---------------------------------------------------------------------------


class EventDraft(BaseModel):
    """One fact the classifier believes an item asserts. Internal.

    Deliberately not a ``ServiceEvent``: evidence grade is not the model's to
    decide, so there is no provenance field for it to fill in.
    """

    item_id: str = Field(description="item_id of the correspondence item this fact came from")
    event_date: date = Field(description="Calendar date the session was or was not delivered")
    service: str = Field(description="Service name, copied exactly from the IEP service list given")
    delivered: bool = Field(description="True only if the item says the session took place")
    minutes: int = Field(ge=0, description="Minutes delivered; 0 when the session did not happen")


class ClassificationBatch(BaseModel):
    """Structured-output envelope for one batch. Internal.

    The class name reaches Bedrock as the tool name, and some models silently
    normalize a leading underscore away — which then fails the name match on
    the way back. Hence a public name for what is otherwise an internal type.
    """

    events: list[EventDraft] = Field(
        default_factory=list,
        description="Empty when nothing in the batch states that a session did or did not happen",
    )


def load_correspondence(name: str = "maya_fall_2026") -> list[Correspondence]:
    """Load one synthetic semester, oldest item first.

    A semester is split across several files by source (school email, parent
    log, service log, progress report) because that is how a parent actually
    accumulates it. Item ids must be unique across the whole set, since every
    downstream ``EvidenceRef`` points back through one.
    """
    paths = sorted(CORRESPONDENCE_DIR.glob(f"{name}_*.json"))
    if not paths:
        raise FileNotFoundError(f"no correspondence fixture files match {name}_*.json in {CORRESPONDENCE_DIR}")

    items: list[Correspondence] = []
    seen: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if SYNTHETIC_MARKER not in payload:
            raise ValueError(f"{path.name} is not marked synthetic; Minutes fixtures never contain real records")
        for raw in payload["items"]:
            item = Correspondence.model_validate(raw)
            if item.item_id in seen:
                raise ValueError(f"duplicate item_id {item.item_id!r} in {path.name}")
            seen.add(item.item_id)
            items.append(item)

    items.sort(key=lambda i: (i.received, i.item_id))
    return items


def load_cached_events(name: str = "maya_fall_2026") -> list[ServiceEvent]:
    """Read the events cached by ``scripts/classify_once.py``. No LLM call."""
    path = CACHE_DIR / f"{name}_events.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/classify_once.py {name}")
    return [ServiceEvent.model_validate(raw) for raw in json.loads(path.read_text(encoding="utf-8"))]


def classify_to_events(items: list[Correspondence], ledger: IEPLedger) -> list[ServiceEvent]:
    """Read the correspondence and emit the delivery facts it states.

    A small classifier reading a long batch skips items, and not the same ones
    each time — which is what makes reading the stack more than once worth
    paying for. Each pass is independent and the results are pooled, then voted
    on per fact in ``_drafts_to_events``: a pass that omits a session is
    silence, but a pass that reverses one is a contradiction, and the two are
    not treated alike.

    Silence is not free, though. A key that only one of the three passes ever
    emitted has been read once, not voted on, so ``VOTE_QUORUM`` requires two
    readings before a fact is established at all. That is the difference
    between three passes actually protecting against a lone bad reading and
    merely appearing to.
    """
    model = BedrockModel(model_id=CLASSIFIER_MODEL, region_name=BEDROCK_REGION, max_tokens=4096)
    batches = list(_batches(items))

    drafts: list[EventDraft] = []
    for _ in range(CLASSIFIER_PASSES):
        for batch in batches:
            # A fresh Agent per batch: batches are independent, and a shared one
            # would carry every previous batch forward as conversation history.
            agent = Agent(model=model, system_prompt=SYSTEM_PROMPT, callback_handler=None)
            drafts.extend(agent.structured_output(ClassificationBatch, _batch_prompt(batch, ledger)).events)

    return _drafts_to_events(drafts, items, ledger, min_readings=VOTE_QUORUM)


def _batches(items: list[Correspondence]) -> Iterator[list[Correspondence]]:
    batch: list[Correspondence] = []
    size = 0
    for item in items:
        weight = len(item.subject) + len(item.body)
        if batch and (len(batch) >= BATCH_MAX_ITEMS or size + weight > BATCH_CHAR_BUDGET):
            yield batch
            batch, size = [], 0
        batch.append(item)
        size += weight
    if batch:
        yield batch


def _batch_prompt(batch: list[Correspondence], ledger: IEPLedger) -> str:
    services = "\n".join(
        f"- {o.service} — {o.minutes_per_session} minutes per session, "
        f"{o.sessions_per_period}x per {o.period.value}, {o.setting}"
        for o in ledger.obligations
    )
    covers_start = min(o.start_date for o in ledger.obligations)
    covers_end = max(o.end_date for o in ledger.obligations)
    rendered = "\n".join(_render(item) for item in batch)
    return (
        f"IEP services for {ledger.student_alias}, school year {ledger.school_year}:\n"
        f"{services}\n\n"
        f"These services run {covers_start.isoformat()} through {covers_end.isoformat()}. "
        f"A date written without a year falls in whichever of those years has that month.\n\n"
        f"Correspondence items:\n\n{rendered}"
    )


def _render(item: Correspondence) -> str:
    return (
        f"--- item_id: {item.item_id}\n"
        f"kind: {item.kind.value}\n"
        f"received: {item.received.isoformat()} ({item.received.strftime('%A')})\n"
        f"dates: {_date_hints(item.received)}\n"
        f"from: {item.sender}\n"
        f"subject: {item.subject}\n"
        f"body:\n{item.body}\n"
    )


def _date_hints(received: date) -> str:
    """Resolve every relative date form an item can use, so the model need not.

    Weekday arithmetic is the one thing a small classifier reliably gets wrong,
    and it is trivially deterministic, so it is done here.
    """
    monday = received - timedelta(days=received.weekday())
    this_week = [monday + timedelta(days=n) for n in range(5)]
    last_week = [day - timedelta(days=7) for day in this_week]

    def week(days: list[date]) -> str:
        return ", ".join(f"{name} {day.isoformat()}" for name, day in zip(_WEEKDAY_NAMES, days))

    return (
        f"today {received.isoformat()}; "
        f"this week: {week(this_week)}; "
        f"last week: {week(last_week)}"
    )


# ---------------------------------------------------------------------------
# Grounding: a claim is admissible only where the document actually makes it.
# ---------------------------------------------------------------------------

# Words by which a line announces that it is talking about a service or an
# encounter at all. A date sitting in a line with none of these is a date in a
# sentence about picture day, report cards or the bus route.
_SESSION_MENTION = re.compile(
    r"\b(?:speech|slp|ot|occupational|therap\w*|counsel\w*|psycholog\w*"
    r"|resource\s+room|specialized\s+academic|session\w*|group\w*|service\w*"
    r"|min|mins|minute\w*|held|saw|seen|met|meet|miss\w*|cancel\w*|pull\w*"
    r"|make-?up|makeup|encounter\w*|appointment\w*|caseload)\b"
    r"|\bsee\s+(?:her|him|them|maya)\b",
    re.IGNORECASE,
)

# Sentences that name a date but disclaim reporting a session on it. These are
# the exact patterns SYSTEM_PROMPT tells the model to emit nothing for; the
# guard exists because telling it is not the same as preventing it.
_NO_SESSION_ASSERTED = re.compile(
    r"\byet\b"
    r"|\b(?:hasn'?t|haven'?t|has\s+not|have\s+not)\b"
    r"|\bnever\b"
    r"|\bstill\s+nothing\b"
    r"|\bdaily\b|\bevery\s+(?:day|week)\b"
    r"|\bas\s+outlined\b"
    r"|\b(?:does\s*n'?t|do\s*n'?t|did\s*n'?t|can'?t|cannot|could\s*n'?t)\s+"
    r"(?:remember|recall|tell)\b"
    r"|\bnext\s+week\b|\bweek\s+of\b"
    # A closed day is not a missed session, however many dates the notice
    # names. Snow days, holidays and staff development days announce
    # themselves in this shape.
    r"|\bno\s+school\b|\bsnow\s+day\b"
    r"|\b(?:school|schools|offices|office)\b[^.]{0,60}\bclosed\b"
    r"|\bclosed\s+(?:today|for|from)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|;")


def date_is_grounded(item: Correspondence, day: date) -> bool:
    """Does ``item`` actually report a session on ``day``?

    The cite-or-stay-silent rule, applied to the classifier. A model asked to
    read a stack of school correspondence will occasionally summarize rather
    than report — turning "services are being provided as outlined in the IEP"
    into a run of sessions that were never written down anywhere.

    Three conditions, all necessary:

    1. Some line of the item names ``day`` — as "today"/"yesterday" relative to
       when the item was received, as a written or numeric date, or as a
       weekday name that resolves unambiguously.
    2. That line also names a service or an encounter, so a date in a sentence
       about picture day or report cards grounds nothing.
    3. The sentence naming it does not disclaim that a session is being
       reported — "no speech *yet*", "she *hasn't* met the speech teacher",
       "she comes to the resource room *daily*".

    This is a necessary condition, not a sufficient one: it cannot make a
    reading true, only refuse one the document does not carry.
    """
    for line in _lines(item):
        if not _SESSION_MENTION.search(line):
            continue
        for sentence in _SENTENCE_SPLIT.split(line):
            if _NO_SESSION_ASSERTED.search(sentence):
                continue
            if _sentence_names_day(sentence, item, day):
                return True
    return False


def _lines(item: Correspondence) -> list[str]:
    return [line for line in f"{item.subject}\n{item.body}".splitlines() if line.strip()]


def _sentence_names_day(sentence: str, item: Correspondence, day: date) -> bool:
    text = sentence.casefold()

    if day == item.received and _mentions(text, "today"):
        return True
    if day == item.received - timedelta(days=1) and _mentions(text, "yesterday"):
        return True

    written_forms = (
        day.isoformat(),
        f"{day.month}/{day.day}",
        f"{day.month:02d}/{day.day:02d}",
        f"{day.month}/{day.day}/{day.year}",
        f"{day.month}/{day.day}/{day.year % 100}",
        f"{day.strftime('%B')} {day.day}".casefold(),
        f"{day.strftime('%b')} {day.day}".casefold(),
    )
    if any(_mentions(text, form) for form in written_forms):
        return True

    return _names_weekday(text, item, day)


def _names_weekday(text: str, item: Correspondence, day: date) -> bool:
    """A weekday name resolves only when it resolves to exactly one day.

    "Tuesday" written on a Friday could be either of two Tuesdays inside the
    window the item's own date hints cover, and the guard cannot read the
    writer's mind — so unless the sentence says which week, it names no date
    at all. Written on a Tuesday or Wednesday, only last week's Tuesday has
    happened yet, and that one is unambiguous.
    """
    if day.weekday() >= 5:
        return False
    if not _mentions(text, day.strftime("%A").casefold()):
        return False

    monday = item.received - timedelta(days=item.received.weekday())
    this_week = monday + timedelta(days=day.weekday())
    last_week = this_week - timedelta(days=7)
    if day not in (this_week, last_week):
        return False

    if _mentions(text, "this week"):
        return day == this_week
    if _mentions(text, "last week"):
        return day == last_week

    already_happened = [candidate for candidate in (this_week, last_week) if candidate <= item.received]
    return already_happened == [day]


def _mentions(text: str, token: str) -> bool:
    """Whole-token search, so '1/5' does not match inside '11/5'."""
    return re.search(rf"(?<![\w/]){re.escape(token)}(?![\w/])", text) is not None


# ---------------------------------------------------------------------------
# Durations: a number is a claim, and claims need grounding too.
# ---------------------------------------------------------------------------

_MINUTE_WORDS = {
    "five": 5,
    "ten": 10,
    "fifteen": 15,
    "twenty": 20,
    "twenty-five": 25,
    "twenty five": 25,
    "thirty": 30,
    "thirty-five": 35,
    "thirty five": 35,
    "forty": 40,
    "forty-five": 45,
    "forty five": 45,
    "fifty": 50,
    "fifty-five": 55,
    "fifty five": 55,
    "sixty": 60,
    "ninety": 90,
}

_DIGIT_DURATION = re.compile(r"\b(\d{1,3})[\s-]{0,3}(?:mins?|minutes?)\b", re.IGNORECASE)
_WORD_DURATION = re.compile(
    r"\b(" + "|".join(sorted((re.escape(w) for w in _MINUTE_WORDS), key=len, reverse=True)) + r")"
    r"\s+(?:mins?|minutes?)\b",
    re.IGNORECASE,
)
# A service-log row states its duration as a column, not as prose:
#   2026-09-22  10:15-10:45   30   G    3    Therapy room   Held
_LOG_ROW_DURATION = re.compile(r"\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\s+(\d{1,3})\b")


def stated_minutes(item: Correspondence) -> frozenset[int]:
    """Every session duration this item actually writes down.

    A batch is several unrelated items in one prompt, and a model that has just
    read "about fifteen minutes" in one email will occasionally hang it on the
    parent log two items later. That error is not symmetric: an ungrounded
    short duration deletes delivered minutes from the record and inflates the
    shortfall asserted against a school. So a duration is believed only where
    the number is written in the item the fact claims to come from.
    """
    text = f"{item.subject}\n{item.body}"
    found = {int(match) for match in _DIGIT_DURATION.findall(text)}
    found |= {_MINUTE_WORDS[word.casefold()] for word in _WORD_DURATION.findall(text)}
    found |= {int(match) for match in _LOG_ROW_DURATION.findall(text)}
    return frozenset(found)


def _voted_minutes(readings: list[int]) -> int | None:
    """Plurality, or None when the readings cannot be reconciled.

    Duration is voted on exactly as delivery is. Taking the minimum instead —
    the reflex that looks conservative — is the opposite of conservative here:
    it lets a lone bad reading lose the question of whether the session
    happened while winning the question of how long it ran, and it moves in the
    one direction that manufactures a shortfall.
    """
    if not readings:
        return None
    ranked = Counter(readings).most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def _delivered_minutes(group: list[EventDraft], item: Correspondence, obligation: ServiceObligation) -> int:
    grounded = stated_minutes(item)
    readings = [draft.minutes for draft in group if draft.delivered and draft.minutes in grounded]
    voted = _voted_minutes(readings)
    if voted is None:
        # The item states no duration, or states one the readings cannot agree
        # on. The IEP's own figure is the only number anyone has written down.
        return obligation.minutes_per_session
    return min(voted, MAX_SESSION_FACTOR * obligation.minutes_per_session)


# ---------------------------------------------------------------------------
# Excusable non-delivery: carried forward, never silently folded in.
# ---------------------------------------------------------------------------

_STUDENT_ABSENCE = re.compile(
    r"\babsent\b|\bhome\s+sick\b|\bout\s+sick\b|\bstayed\s+home\b"
    r"|\bnurse'?s?\s+office\b|\bkept\s+(?:her|him)\s+home\b"
    r"|\bnot\s+(?:at|in)\s+school\b",
    re.IGNORECASE,
)
_THIRD_PERSON = re.compile(r"\b(?:she|her|he|him|maya|the\s+(?:student|child))\b", re.IGNORECASE)
_FIRST_PERSON_SUBJECT = re.compile(r"\bi\s*(?:'m|am|was|'ve|have)\b", re.IGNORECASE)


def reports_student_absence(item: Correspondence) -> bool:
    """Does this item attribute a non-delivery to the child being away?

    "Maya was absent today so we missed OT" and "I am out sick today so there
    is no OT" are the same sentence shape and opposite facts, so the test is
    per sentence and turns on who is missing: a third-person subject with an
    absence marker, and no first-person one.
    """
    for line in _lines(item):
        for sentence in _SENTENCE_SPLIT.split(line):
            if not _STUDENT_ABSENCE.search(sentence):
                continue
            if _FIRST_PERSON_SUBJECT.search(sentence):
                continue
            if _THIRD_PERSON.search(sentence):
                return True
    return False


def student_absence_events(
    events: list[ServiceEvent],
    items: list[Correspondence],
) -> list[ServiceEvent]:
    """The non-deliveries the child's own absence caused.

    ``ServiceEvent`` is frozen and has no field for fault, and this module does
    not judge fault — but it does know which documents blame the child's
    absence, and refusing to pass that on is not neutrality. Provenance says a
    SCHOOL_CONFIRMED fact may be cited without qualification, so without this
    list a compensatory demand can be built on a session the school could not
    have delivered. Reconciliation and the letter compiler must qualify or
    exclude everything this returns.
    """
    absence_sources = {item.item_id for item in items if reports_student_absence(item)}
    return [event for event in events if not event.delivered and event.source in absence_sources]


# ---------------------------------------------------------------------------
# Drafts to facts.
# ---------------------------------------------------------------------------


def _is_admissible(draft: EventDraft, item: Correspondence, obligation: ServiceObligation) -> bool:
    if draft.event_date > item.received:
        return False  # nothing can document a session that had not happened yet
    if draft.event_date.weekday() >= 5:
        return False  # school services are delivered on school days
    if not obligation.start_date <= draft.event_date <= obligation.end_date:
        return False  # outside the term the IEP promises this service for
    if item.kind in RETROSPECTIVE_KINDS and draft.event_date == item.received:
        # A compiled document reports on a period that has already closed, so
        # its own issue date is masthead metadata rather than an encounter.
        # Without this, "Date issued: 2027-01-29" reads as a session.
        return False
    return date_is_grounded(item, draft.event_date)


def _drafts_to_events(
    drafts: list[EventDraft],
    items: list[Correspondence],
    ledger: IEPLedger,
    min_readings: int = 1,
) -> list[ServiceEvent]:
    """Turn model drafts into ledger facts, deciding everything load-bearing here.

    Provenance comes from the source item, the service name is snapped to the
    IEP, minutes must be written in the source document, and every date must be
    admissible — so a hallucinated service, an unknown source, a duration
    borrowed from a neighbouring item, or an ungrounded date cannot reach
    reconciliation.

    Drafts naming the same service on the same date in the same document are
    repeat readings of one sentence, so they are voted rather than accumulated —
    on delivery and on duration alike. A fact the readings contradict each other
    about is not established, and is dropped: better a gap the discovery module
    can chase than a shortfall asserted against a school on the strength of one
    bad reading.

    ``min_readings`` is the quorum. ``classify_to_events`` passes
    ``VOTE_QUORUM`` because a key only one of its three passes emitted has been
    read once, not voted on. It defaults to 1 so that a caller replaying facts
    that have already been established — ``scripts/classify_once.py --regroom``,
    and the unit tests — re-applies the deterministic guards without re-running
    an election it has no ballots for.
    """
    by_id = {item.item_id: item for item in items}
    by_service = {o.service.casefold(): o for o in ledger.obligations}

    votes: dict[tuple[str, date, str], list[EventDraft]] = {}
    for draft in drafts:
        item = by_id.get(draft.item_id)
        obligation = by_service.get(draft.service.strip().casefold())
        if item is None or obligation is None or not _is_admissible(draft, item, obligation):
            continue
        votes.setdefault((item.item_id, draft.event_date, obligation.service), []).append(draft)

    events: list[ServiceEvent] = []
    for (item_id, event_date, service), group in votes.items():
        if len(group) < min_readings:
            continue

        delivered = _majority_delivered(group)
        if delivered is None:
            continue

        item = by_id[item_id]
        obligation = by_service[service.casefold()]
        minutes = _delivered_minutes(group, item, obligation) if delivered else 0

        events.append(
            ServiceEvent(
                event_date=event_date,
                service=service,
                minutes=minutes,
                delivered=delivered,
                provenance=PROVENANCE_BY_KIND[item.kind],
                source=item_id,
            )
        )

    events.sort(key=lambda e: (e.event_date, e.service, e.source))
    return events


def _majority_delivered(group: list[EventDraft]) -> bool | None:
    """None when the readings split evenly, i.e. the fact is not established."""
    held = sum(1 for draft in group if draft.delivered)
    missed = len(group) - held
    return None if held == missed else held > missed
