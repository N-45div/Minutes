"""The agent's hands — the deterministic engine, exposed as Strands tools.

The model decides WHAT to look at and WHEN. It never decides a number, a date,
or whether a school failed. Every figure a tool returns was computed by
``reconcile.py``, ``deadlines.py`` or ``discovery.py``; every letter was
assembled by ``letters.py`` from evidence it can cite. The model's job is
orchestration and connective prose, and this module is the only surface it
gets, so the arithmetic is structurally out of its reach.

Two properties hold for every tool here, and both are load-bearing:

**Nothing in this module reaches the outside world.** Every tool is read-only
or drafting-only. ``draft_*`` compiles a document and hands it back; it does
not send it, and it does not record that anything was sent. Sending is a
decision a parent makes, so it lives behind an approval interrupt in the agent
layer where a human can answer. A tool the model can call unattended must never
be able to mail a district.

**Nothing here raises into the event loop.** A tool that raises becomes an
opaque ``Error: ValueError - ...`` toolResult and the model's usual recovery is
to call it again the same way. So failures come back as an ``error`` key
carrying a sentence the model can act on, and the loop stays legible.

Returns are JSON-friendly primitives — ISO date strings in, dicts and strings
out — because the model reads a serialized rendering of whatever a tool
returns, not a Python object. They are also deliberately compact: full letter
bodies and the statement are returned because they *are* the artifact, but
evidence tables are summarized to counts. A wall of JSON is paid for twice, in
context and in tokens, and re-reading forty citation records has never changed
a model's next move.

**EVERY TOOL THAT READS REQUESTS READS THE AGENT'S VIEW OF THEM.** The case
record is a historical baseline and, in this build, an empty one — so a tool
reading ``case.requests`` alone would report that no records request had ever
been made, on a case where the agent released one last month and is waiting on
it. That is not a stale number, it is a false statement to a parent who is
holding their own copy of the letter, and it would re-raise decisions they had
already acted on. So the request-aware tools take a ``ToolContext`` and read
:func:`case_requests`, which merges the baseline with what this agent has since
sent. Called as a plain Python function with no context, they fall back to the
baseline — which is what a test without an agent should see.
"""

from __future__ import annotations

import copy
import functools
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

from strands import ToolContext, tool
from strands.interrupt import InterruptException

from .cases import case_store, current_case_id, is_sample, validate_case_id
from .correspondence import attributed, load_cached_events, load_correspondence
from .deadlines import evaluate_deadlines, next_action_date, urgency_for
from .decisions import REQUEST_CADENCE_DAYS, capped_urgency, decisions_for, is_quiet
from .discovery import RESPONSE_WINDOW_DAYS, due_requests, refresh_states
from .letters import (
    compile_compensatory_request,
    compile_records_request,
    compile_shortfall_notice,
    validate_letter,
)
from .models import (
    Provenance,
    DeadlineState,
    DeadlineStatus,
    DecisionCard,
    IEPLedger,
    Letter,
    RecordsRequest,
    RequestState,
    ServiceEvent,
    ServiceShortfall,
)
from .reconcile import reconcile
from .statement import build_statement, render_markdown

__all__ = [
    "STATE_REQUESTS",
    "CaseRecord",
    "build_monthly_statement",
    "case_requests",
    "check_deadlines",
    "draft_records_request",
    "draft_shortfall_letter",
    "load_case",
    "load_case_record",
    "pending_decisions",
    "reconcile_services",
    "records_requests_status",
    "store_requests",
]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

# The demo case ships as two fixture names because it was produced by two
# one-shot scripts: extract_once.py wrote the ledger, classify_once.py wrote
# the events read out of the correspondence.
LEDGER_FIXTURE = "iep_maya"
CORRESPONDENCE_FIXTURE = "maya_fall_2026"

# The fixture case's own name. Every tool in this module reports one case:
# the one named by minutes.cases.current_case_id for this invocation, or this
# one when nothing named a case -- which is a test, a script, or the demo.
DEFAULT_CASE = "maya"

_OUTSTANDING_STATES = frozenset({RequestState.SENT, RequestState.UNANSWERED_OVERDUE})

UNDOCUMENTED_NOTE = (
    "undocumented_minutes are promised minutes with no record either way. They are a gap "
    "in the evidence, not a record of non-delivery, and must never be described as minutes "
    "the school failed to deliver."
)

EXCUSED_NOTE = (
    "excused_minutes fall on dates a record notes the student was absent. They stay inside "
    "owed_minutes and are excluded from the shortfall, because a session the school could "
    "not have delivered is not a session it owes back."
)

ABSENCE_NOTE = (
    "dates_noting_student_absence counts dates a record notes the student was away. It "
    "exists for one purpose: so those minutes can be taken out of what is asked of the "
    "district. It is never a reason the services fell short, never a claim about the "
    "child, and must never appear in prose about her attendance, her needs or her "
    "progress."
)

DRAFT_NOTE = (
    "Nothing was sent. This is a compiled draft: it goes to the parent for approval, and "
    "only a human answer can release it."
)


# ---------------------------------------------------------------------------
# The data seam.
#
# Everything below reads its case through load_case_record and nothing else,
# so replacing committed fixtures with a real per-family data source is a
# change to one function.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseRecord:
    """One family's case as the tools see it: the promise, the evidence, and
    the history of what has been asked of the district."""

    case_id: str
    ledger: IEPLedger
    events: list[ServiceEvent]
    requests: list[RecordsRequest]
    correspondence_items: int


@functools.lru_cache(maxsize=4)
def _read_case(case_id: str) -> CaseRecord:
    """Parse the fixture case once. Callers get a copy, never this object.

    Only the sample lives here. A stored case goes through
    :func:`_read_stored_case`, which is deliberately NOT cached: a note a
    parent added a second ago must be in the record the next call reads.
    """
    if case_id != DEFAULT_CASE:
        raise ValueError(
            f"unknown case {case_id!r}; this build ships the single demo case {DEFAULT_CASE!r}"
        )

    ledger_path = FIXTURES / "cache" / f"{LEDGER_FIXTURE}_ledger.json"
    if not ledger_path.exists():
        raise FileNotFoundError(
            f"{ledger_path} missing; run scripts/extract_once.py {LEDGER_FIXTURE}"
        )

    return CaseRecord(
        case_id=case_id,
        ledger=IEPLedger.model_validate_json(ledger_path.read_text(encoding="utf-8")),
        events=load_cached_events(CORRESPONDENCE_FIXTURE),
        # The fixture case carries no request history: the first records request
        # this agent drafts is the first one this family has ever sent. A real
        # store returns the sent/answered history here and the discovery cadence
        # picks up where it left off with no change to any tool.
        requests=[],
        correspondence_items=len(load_correspondence(CORRESPONDENCE_FIXTURE)),
    )


def _read_stored_case(case_id: str) -> CaseRecord:
    """A parent's own case, read fresh from the case store on every call.

    Built exactly the way the fixture record is built: the events are stamped
    with :func:`~minutes.correspondence.attributed` against the correspondence
    they came from, so a non-delivery whose own note says the child was away is
    excused here and nowhere else. The request history is empty for the same
    reason it is empty for the fixture -- what this agent has sent lives in the
    caseworker's session and :func:`case_requests` merges it in.
    """
    store = case_store()
    ledger = store.read_ledger(case_id)
    if ledger is None:
        raise ValueError(
            f"unknown case {case_id!r}; no IEP has been ingested for it (ingest_iep creates a case)"
        )
    items = store.read_correspondence(case_id)
    return CaseRecord(
        case_id=case_id,
        ledger=ledger,
        events=attributed(store.read_events(case_id), items),
        requests=[],
        correspondence_items=len(items),
    )


def load_case_record(case_id: str | None = None) -> CaseRecord:
    """Read one case. THE SWAP POINT for a real data source.

    ``case_id`` is the case to read; ``None`` means the case this invocation is
    about (:data:`minutes.cases.current_case_id`, set by the runtime entrypoint
    for the whole request), and when nothing named one, the committed sample.
    Every tool in this module calls this with no argument, which is what makes
    a parent's case reach every tool without any tool knowing it.

    The sample is parsed once from the committed synthetic fixtures under
    ``fixtures/`` and handed out as a deep copy, so a caller that mutates what
    it gets back cannot corrupt the case for the next tool call. A stored case
    is read fresh each time and needs no copy.
    """
    if case_id is None:
        case_id = current_case_id.get() or DEFAULT_CASE
    if is_sample(case_id):
        return copy.deepcopy(_read_case(DEFAULT_CASE))
    return _read_stored_case(validate_case_id(case_id))


# ---------------------------------------------------------------------------
# Failure handling and coercion.
# ---------------------------------------------------------------------------


def _guard(fn: Callable[..., dict]) -> Callable[..., dict]:
    """Turn any failure into an ``error`` key the model can read and act on.

    ``InterruptException`` is re-raised deliberately. It subclasses
    ``Exception``, so a bare ``except Exception`` here would swallow a pending
    human-approval pause and turn it into a caught error with no diagnostic.
    Nothing in this module interrupts today, but a guard that quietly breaks
    human-in-the-loop the day one does is not a guard worth having.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            return fn(*args, **kwargs)
        except InterruptException:
            raise
        except (ValueError, FileNotFoundError, KeyError, TypeError) as failure:
            return {"error": str(failure)}
        except Exception as failure:  # pragma: no cover -- last resort
            return {"error": f"{type(failure).__name__}: {failure}"}

    return wrapper


_ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def _parse_date(value: str, field: str) -> date:
    """An ISO date, or a message naming the parameter that was wrong.

    The shape is checked before parsing, because ``date.fromisoformat`` accepts
    more than this signature promises and the extras are the dangerous kind.
    ``'2026-W01-1'`` parses cleanly to 2025-12-29 — a different *year* than the
    caller wrote — and ``'20261219'`` parses to 2026-12-19. Both would then flow
    into every clock in the run: deadline states, response windows, the
    discovery cadence, the date a silence is asserted on. In a product where the
    date is the product, a silent reinterpretation is worse than a rejection, so
    the accepted set is exactly the ``YYYY-MM-DD`` the docstring and the error
    message both name.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date string like '2026-12-19', got {value!r}")
    if not _ISO_DAY.fullmatch(value):
        raise ValueError(
            f"{field}={value!r} is not an ISO date; use YYYY-MM-DD, for example '2026-12-19'"
        )
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError(
            f"{field}={value!r} is not an ISO date; use YYYY-MM-DD, for example '2026-12-19'"
        ) from None


def _window(start: str, end: str) -> tuple[date, date]:
    first, last = _parse_date(start, "start"), _parse_date(end, "end")
    if last < first:
        raise ValueError(f"end={end} precedes start={start}; a service period runs forwards")
    return first, last


# ---------------------------------------------------------------------------
# The case as the agent knows it.
#
# load_case_record reads the file. This reads the file PLUS whatever the running
# agent has done since, which for records requests is the difference between a
# true statement and a false one. It lives here, beside the data seam, so that
# both the read tools below and the send tools in minutes.agent see one case.
# ---------------------------------------------------------------------------

STATE_REQUESTS = "records_requests"
"""Key under ``agent.state`` holding the requests this agent has sent.

The session manager persists ``agent.state`` verbatim, so this is what lets a
request released in October still be tracked, and go overdue, in a December
process that shares nothing with the October one but a session id.
"""


def case_requests(agent: Any, case: CaseRecord) -> list[RecordsRequest]:
    """The records requests for this case: the file's history plus what we sent.

    The case record is the historical baseline — what the family already had on
    file before Minutes existed. ``agent.state`` holds what this agent has sent
    since, and it wins on a shared id, because our state machine has been
    advancing that request while the baseline stayed frozen.

    ``agent`` may be ``None``, which means "no agent is running": the baseline is
    the whole answer. That is the case in a unit test calling a tool directly,
    and it must not be the case anywhere a parent is being told something.
    """
    merged = {request.request_id: request for request in case.requests}
    for stored in (agent.state.get(STATE_REQUESTS) or []) if agent is not None else []:
        request = RecordsRequest.model_validate(stored)
        merged[request.request_id] = request
    return sorted(merged.values(), key=lambda request: request.request_id)


def store_requests(agent: Any, requests: list[RecordsRequest]) -> None:
    """Write the request state machine back to ``agent.state``."""
    agent.state.set(STATE_REQUESTS, [request.model_dump(mode="json") for request in requests])


def _known_requests(tool_context: Any, case: CaseRecord, today: date) -> list[RecordsRequest]:
    """Every request this case knows about, with its clocks wound to ``today``."""
    agent = getattr(tool_context, "agent", None)
    return refresh_states(case_requests(agent, case), today)


# ---------------------------------------------------------------------------
# Compact renderings. Each one answers "what would change the model's next
# move?" and drops the rest.
# ---------------------------------------------------------------------------


def _iso_or_none(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _shortfall_row(shortfall: ServiceShortfall) -> dict:
    return {
        "service": shortfall.service,
        "owed_minutes": shortfall.owed_minutes,
        "delivered_minutes": shortfall.delivered_minutes,
        "excused_minutes": shortfall.excused_minutes,
        "shortfall_minutes": shortfall.shortfall_minutes,
        "school_confirmed_minutes": shortfall.school_confirmed_minutes,
        "parent_observed_minutes": shortfall.parent_observed_minutes,
        "undocumented_minutes": shortfall.undocumented_minutes,
        # Descriptive, and labelled as such so the model does not read a reason
        # as a finding. Whether missed minutes are owed back is a FAPE question
        # decided case by case; the letter compilers state these reasons and
        # ask the district about them, and never conclude from them.
        "reasons_the_records_give": [
            {
                "reason": reason.cause.phrase,
                "sessions": reason.sessions,
                "dates": [day.isoformat() for day in reason.dates],
                "written_by": (
                    "the district"
                    if reason.evidence_grade is Provenance.SCHOOL_CONFIRMED
                    else "the family"
                ),
            }
            for reason in shortfall.stated_reasons
        ],
        "evidence_count": len(shortfall.evidence),
    }


def _deadline_row(status: DeadlineStatus) -> dict:
    """One deadline, on the same urgency scale a decision card would use.

    ``urgency_for`` is the calendar's answer and it is not the one that governs
    an interruption: :data:`minutes.decisions.URGENCY_CAP_BY_KIND` lowers a
    progress report, because a report that has not arrived is a document to
    chase and not an opportunity actively lapsing. Reporting the raw value here
    would have this tool call a missed progress report ``deadline_imminent``
    while ``pending_decisions`` calls the same fact on the same day
    ``time_sensitive``, and a model composing prose from both would be reading
    two scales it was never told apart.
    """
    return {
        "kind": status.deadline.kind.value,
        "due": status.deadline.due.isoformat(),
        "description": status.deadline.description,
        "state": status.state.value,
        "days_remaining": status.days_remaining,
        "urgency": capped_urgency(urgency_for(status), status.deadline.kind).value,
        "met_on": _iso_or_none(status.met_on),
    }


def _request_row(request: RecordsRequest) -> dict:
    return {
        "request_id": request.request_id,
        "covers_start": request.covers_start.isoformat(),
        "covers_end": request.covers_end.isoformat(),
        "services": list(request.services),
        "state": request.state.value,
        "sent_on": _iso_or_none(request.sent_on),
        "response_due": _iso_or_none(request.response_due),
        "answered_on": _iso_or_none(request.answered_on),
    }


def _letter_payload(letter: Letter) -> dict:
    """The letter as the model needs it: the whole body, and proof it is sound.

    The body is returned in full because it is the artifact a parent approves.
    The citation table is not: forty evidence records change nothing about what
    the model does next, and their split by evidence grade — which is the part
    that decides how strongly the letter may speak — is three numbers.
    """
    blocking = validate_letter(letter, for_sending=True)
    return {
        "kind": letter.kind.value,
        "subject": letter.subject,
        "body": letter.body,
        "legal_basis": list(letter.legal_basis),
        "disclaimer": letter.disclaimer,
        "citations": {
            "count": len(letter.citations),
            "by_evidence_grade": dict(
                Counter(c.evidence.provenance.value for c in letter.citations)
            ),
        },
        "integrity_violations": validate_letter(letter),
        "blocking_before_send": blocking,
        "ready_to_send": not blocking,
    }


# ---------------------------------------------------------------------------
# Decision cards.
# ---------------------------------------------------------------------------


def _decision_row(card: DecisionCard) -> dict:
    return {
        "card_id": card.card_id,
        "title": card.title,
        "why_now": card.why_now,
        "facts": list(card.facts),
        "recommended_action": card.recommended_action,
        "urgency": card.urgency.value,
        "deadline": _iso_or_none(card.deadline),
        "has_draft_letter": card.draft is not None,
    }


# ---------------------------------------------------------------------------
# The tools. Docstrings below are written to the model: they become the tool
# descriptions it reads when deciding what to call.
# ---------------------------------------------------------------------------


@tool
@_guard
def load_case() -> dict:
    """Load the child's IEP obligations and a summary of the evidence on file.

    Call this first. It returns what the IEP legally promises — each service,
    how many minutes per session, how often, over what dates — plus the
    statutory deadlines it sets and a count of the delivery evidence collected
    so far. Every other tool reports on this same case.

    Figures here are the promise, not the delivery. Use reconcile_services to
    find out what was actually delivered.

    Two things here are about the child and not about the district, and neither
    is ever a reason a service fell short: the count of dates a record notes she
    was absent exists only so those minutes can be excluded from what is asked
    for, and the accommodations describe what the district owes her. Read the
    absence_note before writing a sentence containing either.
    """
    case = load_case_record()
    ledger = case.ledger
    events = case.events

    return {
        "student": ledger.student_alias,
        "school_year": ledger.school_year,
        "iep_date": ledger.iep_date.isoformat(),
        "promised_minutes_per_week": round(ledger.promised_minutes_per_week, 1),
        "obligations": [
            {
                "service": o.service,
                "minutes_per_session": o.minutes_per_session,
                "sessions_per_period": o.sessions_per_period,
                "period": o.period.value,
                "minutes_per_week": round(o.minutes_per_week, 1),
                "provider_role": o.provider_role,
                "setting": o.setting,
                "start_date": o.start_date.isoformat(),
                "end_date": o.end_date.isoformat(),
            }
            for o in ledger.obligations
        ],
        "deadlines": [
            {
                "kind": d.kind.value,
                "due": d.due.isoformat(),
                "description": d.description,
            }
            for d in ledger.deadlines
        ],
        "accommodations": [a.description for a in ledger.accommodations],
        "evidence": {
            "events": len(events),
            "correspondence_items": case.correspondence_items,
            "earliest_event": _iso_or_none(min((e.event_date for e in events), default=None)),
            "latest_event": _iso_or_none(max((e.event_date for e in events), default=None)),
            "by_evidence_grade": dict(Counter(e.provenance.value for e in events)),
            "dates_noting_student_absence": sum(
                1 for e in events if e.attribution.value == "student_absence"
            ),
            "absence_note": ABSENCE_NOTE,
        },
        "records_requests_on_file": len(case.requests),
        "note": (
            "Every obligation above carries the verbatim IEP sentence it came from. Those "
            "quotes are not returned here; the letter compilers attach them automatically."
        ),
    }


@tool
@_guard
def reconcile_services(start: str, end: str) -> dict:
    """Compare minutes the IEP owed against minutes documented as delivered.

    This is the arithmetic at the centre of the product. For each service over
    the period you name it returns owed, delivered, excused, and short, plus
    how much of the gap nobody has a record for either way.

    Read the four buckets carefully before writing anything about them.
    delivered_minutes splits into school_confirmed (the district's own record)
    and parent_observed (the family's note, which is an observation, not proof).
    undocumented_minutes is missing evidence, NOT proven non-delivery — never
    describe it as minutes the school failed to deliver. excused_minutes fell
    on dates a record notes the child was absent, and is already excluded from
    shortfall_minutes.

    Args:
        start: First day of the period, ISO format, e.g. '2026-09-01'.
        end: Last day of the period, ISO format, e.g. '2026-12-19'.
    """
    first, last = _window(start, end)
    case = load_case_record()
    result = reconcile(case.ledger, case.events, first, last)

    return {
        "period": {"start": first.isoformat(), "end": last.isoformat()},
        "events_considered": result.events_considered,
        "totals": {
            "owed_minutes": sum(s.owed_minutes for s in result.shortfalls),
            "delivered_minutes": sum(s.delivered_minutes for s in result.shortfalls),
            "excused_minutes": sum(s.excused_minutes for s in result.shortfalls),
            "shortfall_minutes": result.total_shortfall_minutes,
            "undocumented_minutes": sum(s.undocumented_minutes for s in result.shortfalls),
        },
        "services": [_shortfall_row(s) for s in result.shortfalls],
        "services_short": sum(1 for s in result.shortfalls if s.shortfall_minutes > 0),
        "notes": [UNDOCUMENTED_NOTE, EXCUSED_NOTE],
    }


@tool
@_guard
def check_deadlines(today: str) -> dict:
    """Check the statutory clocks the IEP sets: reviews, reevaluations, reports.

    Returns every deadline with its state — upcoming, due_soon, overdue, met —
    the days remaining, and how urgent it is.

    A deadline being overdue is a fact about a calendar. It is not a finding
    that the school violated anything, and must not be written as one.

    Args:
        today: The date to evaluate against, ISO format, e.g. '2026-12-19'.
    """
    now = _parse_date(today, "today")
    case = load_case_record()
    statuses = evaluate_deadlines(case.ledger, now)
    next_action = next_action_date(statuses)

    return {
        "today": now.isoformat(),
        "deadlines": [_deadline_row(s) for s in statuses],
        "overdue": sum(1 for s in statuses if s.state is DeadlineState.OVERDUE),
        "due_soon": sum(1 for s in statuses if s.state is DeadlineState.DUE_SOON),
        "next_action_date": next_action.isoformat() if next_action else None,
    }


@tool(context=True)
@_guard
def records_requests_status(today: str, tool_context: ToolContext | None = None) -> dict:
    """Check the records requests: what is outstanding, overdue, or due to send.

    Minutes asks the district for its own service logs on a fixed cadence
    rather than waiting to be told. This reports where every one of those asks
    stands, and whether a new one has come due.

    An unanswered, overdue request is itself evidence — but only about the
    RECORDS. It establishes that the district produced no documentation when
    it was required to. It never establishes that a session was missed.
    Use draft_records_request to compile the letter for anything due to send.

    Args:
        today: The date to evaluate against, ISO format, e.g. '2026-12-19'.
    """
    now = _parse_date(today, "today")
    case = load_case_record()
    existing = _known_requests(tool_context, case, now)
    newly_due = due_requests(case.ledger, existing, now, cadence_days=REQUEST_CADENCE_DAYS)

    outstanding = [r for r in existing if r.state in _OUTSTANDING_STATES]
    overdue = [r for r in existing if r.state is RequestState.UNANSWERED_OVERDUE]

    return {
        "today": now.isoformat(),
        "requests": [_request_row(r) for r in existing],
        "outstanding": [r.request_id for r in outstanding],
        "overdue_unanswered": [r.request_id for r in overdue],
        "due_to_send": [_request_row(r) for r in newly_due],
        "response_window_days": RESPONSE_WINDOW_DAYS,
        "note": (
            "due_to_send lists requests the cadence says should go out now. Nothing has "
            "been sent; drafting and sending are separate steps."
        ),
    }


@tool(context=True)
@_guard
def draft_records_request(today: str, tool_context: ToolContext | None = None) -> dict:
    """Compile (do NOT send) a letter asking the district for its service records.

    Only drafts when the cadence says a request is due — it will refuse rather
    than crowd a district that is still inside its 45-day response window, or
    ask twice for the same period.

    Nothing is sent. The returned draft goes to the parent, and only the parent
    can release it. Note ready_to_send: a records request is sound but is not
    sendable until the parent fills the placeholder it leaves for them.

    Args:
        today: The date the letter is dated, ISO format, e.g. '2026-12-19'.
    """
    now = _parse_date(today, "today")
    case = load_case_record()
    existing = _known_requests(tool_context, case, now)
    newly_due = due_requests(case.ledger, existing, now, cadence_days=REQUEST_CADENCE_DAYS)

    if not newly_due:
        return {
            "drafted": False,
            "today": now.isoformat(),
            "reason": (
                "No records request is due. Either a request is already open and inside "
                "its response window, or the cadence period has not yet accrued."
            ),
            "outstanding": [r.request_id for r in existing if r.state in _OUTSTANDING_STATES],
        }

    request = newly_due[0]
    letter = compile_records_request(case.ledger, request, today=now)

    return {
        "drafted": True,
        "today": now.isoformat(),
        "request": _request_row(request),
        "letter": _letter_payload(letter),
        "note": DRAFT_NOTE,
    }


@tool(context=True)
@_guard
def draft_shortfall_letter(
    start: str,
    end: str,
    today: str,
    escalate: bool = False,
    tool_context: ToolContext | None = None,
) -> dict:
    """Compile (do NOT send) a letter about the service gap for a period.

    The letter is assembled from the ledger and the evidence, sentence by
    sentence, with a footnote on every factual claim. You cannot edit it and
    you should not restate its figures in your own words — quote it or point
    to it. It never names an amount the family is owed; that is for the IEP
    team or a hearing officer, not for this product.

    The letter is dated with today, not with the end of the period it reports.
    A parent asking in March about the autumn term gets a letter dated in March,
    because a correspondence document dated three months in the past is not
    correspondence a district should ever receive.

    Nothing is sent. The draft goes to the parent for approval.

    Args:
        start: First day of the period, ISO format, e.g. '2026-09-01'.
        end: Last day of the period, ISO format, e.g. '2026-12-19'.
        today: The date the letter is dated and the records requests are
            evaluated against, ISO format. May be later than end.
        escalate: False compiles a shortfall notice, which puts the
            reconciliation in front of the district and asks it to compare
            against its own records. True compiles a compensatory request,
            which asks the IEP team to convene and consider make-up services.
            Escalate only once the gap is evidenced and the district has had
            the chance to answer the notice.
    """
    first, last = _window(start, end)
    now = _parse_date(today, "today")
    # The tool spec declares a boolean, and nothing enforces it at the call
    # boundary — a model that emits the string "false" would otherwise get the
    # ESCALATION letter, because every non-empty string is truthy. Refusing is
    # the only safe reading: the two letters ask a district for different things.
    if not isinstance(escalate, bool):
        raise ValueError(
            f"escalate must be true or false, got {escalate!r}; a non-boolean cannot be "
            "read as a choice between a shortfall notice and a compensatory request"
        )

    case = load_case_record()
    result = reconcile(case.ledger, case.events, first, last)
    requests = _known_requests(tool_context, case, now)

    letter = (
        compile_compensatory_request(case.ledger, result, requests, today=now)
        if escalate
        else compile_shortfall_notice(case.ledger, result, today=now)
    )

    return {
        "drafted": True,
        "period": {"start": first.isoformat(), "end": last.isoformat()},
        "today": now.isoformat(),
        "escalated": escalate,
        "shortfall_minutes": result.total_shortfall_minutes,
        "undocumented_minutes": sum(s.undocumented_minutes for s in result.shortfalls),
        "letter": _letter_payload(letter),
        "note": DRAFT_NOTE,
    }


@tool(context=True)
@_guard
def build_monthly_statement(
    start: str, end: str, today: str, tool_context: ToolContext | None = None
) -> dict:
    """Build the monthly statement — the artifact the parent actually receives.

    One document: what was owed, what was delivered, what is short, which
    clocks are running, what is outstanding with the district, and anything
    waiting on the parent. The rendered markdown is returned ready to send.

    The markdown is the finished artifact. Do not summarize it back to the
    parent in your own words and do not restate its numbers — hand it over.

    Args:
        start: First day of the service period, ISO format, e.g. '2026-09-01'.
        end: Last day of the service period, ISO format, e.g. '2026-12-19'.
        today: The date the statement is issued, ISO format. Deadlines and
            request states are evaluated against this, which may be later than
            the end of the period being reported.
    """
    first, last = _window(start, end)
    now = _parse_date(today, "today")
    case = load_case_record()

    result = reconcile(case.ledger, case.events, first, last)
    deadlines = evaluate_deadlines(case.ledger, now)
    requests = _known_requests(tool_context, case, now)
    cards = decisions_for(case.ledger, result, deadlines, requests, now)

    statement = build_statement(case.ledger, result, deadlines, requests, cards)

    return {
        "period": {"start": first.isoformat(), "end": last.isoformat()},
        "today": now.isoformat(),
        "student": statement.student_alias,
        "totals": {
            "owed_minutes": statement.total_owed,
            "delivered_minutes": statement.total_delivered,
            "shortfall_minutes": statement.total_shortfall,
        },
        "services": len(statement.lines),
        "open_deadlines": len(statement.open_deadlines),
        "unanswered_requests": len(statement.unanswered_requests),
        "decisions": len(statement.decisions),
        "markdown": render_markdown(statement),
    }


@tool(context=True)
@_guard
def pending_decisions(today: str, tool_context: ToolContext | None = None) -> dict:
    """List the decisions actually waiting on the parent right now.

    This is the only thing that should ever interrupt a family. Minutes runs
    in the background and stays quiet; a decision card means there is a real
    choice to make, with the facts behind it and a recommended action.

    An empty list (quiet: true) is the normal, good answer. Say so plainly and
    stop — do not manufacture something for the parent to look at.

    Args:
        today: The date to evaluate against, ISO format, e.g. '2026-12-19'.
    """
    now = _parse_date(today, "today")
    case = load_case_record()

    # Everything owed so far: the promise runs from the first obligation start
    # date, and a decision waiting today is weighed against the whole record,
    # not against an arbitrary recent slice.
    starts = [o.start_date for o in case.ledger.obligations]
    if not starts or min(starts) > now:
        return {
            "today": now.isoformat(),
            "quiet": True,
            "count": 0,
            "decisions": [],
            "note": "No IEP service had started by this date, so nothing can be waiting.",
        }

    result = reconcile(case.ledger, case.events, min(starts), now)
    deadlines = evaluate_deadlines(case.ledger, now)
    requests = _known_requests(tool_context, case, now)
    cards = decisions_for(case.ledger, result, deadlines, requests, now)

    return {
        "today": now.isoformat(),
        "considered_period": {"start": min(starts).isoformat(), "end": now.isoformat()},
        "quiet": is_quiet(cards),
        "count": len(cards),
        "decisions": [_decision_row(c) for c in cards],
    }
