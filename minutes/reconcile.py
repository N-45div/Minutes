"""Owed versus delivered, graded by evidence.

This is the arithmetic core of the product. The IEP is a quantified promise —
34 CFR 300.320(a)(7) requires it to state the frequency, location and duration
of every service — and this module is the only place that promise is turned
into numbers. Nothing here calls a model. A letter that footnotes a number is
only as trustworthy as the number, so every figure below is reproducible from
the ledger and the event list alone.

Four honesty rules are built into the semantics:

1. ``shortfall_minutes`` is arithmetic (owed minus delivered), not an
   accusation. Absent evidence looks identical to non-delivery in the
   subtraction, so the subtraction alone must never be read as a finding.
2. ``undocumented_minutes`` is the correction for that. It is the portion of
   the shortfall for which the district has produced no record either way — no
   delivery, no missed-session note. It is what an unanswered records request
   later converts into DOCUMENTED_SILENCE, and what keeps the product from
   claiming a school failed to deliver when the truth is only that nobody can
   tell.
3. **Only a record the district itself produced retires a promised session from
   the undocumented bucket.** A parent's own note — that a session happened, or
   that it did not — never shrinks that bucket, and neither does a documented
   silence. Anything else would let the family's own unverified observation, or
   the exercise of the family's own records right, quietly convert "nobody can
   tell" into "documented", stripping the hedge that protects the school. See
   :func:`reconcile` for the full rule and ``discovery.py`` for the silence half
   of the contract.
4. Nothing is silently dropped. Every event inside the window is counted in
   ``events_considered``, every matched event becomes exactly one
   ``EvidenceRef`` on exactly one line, and the ones that bind to no active
   obligation are retrievable via :func:`unmatched_events`.

Two invariants every caller may rely on::

    0 <= undocumented_minutes <= shortfall_minutes <= owed_minutes

    result.events_considered - sum(len(line.evidence) for line in
        result.shortfalls) == len(unmatched_events(...))

The second one matters: ``ReconciliationResult`` has no unmatched-event field,
so that subtraction is how a caller holding only the result detects that
in-window records bound to nothing and the row it is about to send may
therefore be overstated. A non-zero difference means "ask :func:`unmatched_events`
before you post this letter".
"""

from __future__ import annotations

import calendar
import math
import re
from datetime import date, timedelta
from fractions import Fraction

from .models import (
    EvidenceRef,
    IEPLedger,
    Period,
    Provenance,
    ReconciliationResult,
    ServiceEvent,
    ServiceObligation,
    ServiceShortfall,
)

__all__ = [
    "CALENDAR_MONTHS_PER_PERIOD",
    "SCHOOL_DAYS_PER_PERIOD",
    "canonical_service",
    "expected_sessions",
    "reconcile",
    "unmatched_events",
]

# School days each period is nominally worth. Mirrors the calendar factors in
# models.py (weeks per period x 5 school days per week) so that a weekly summary
# and a reconciled total never disagree.
#
# DAY and WEEK are exact: one school day is one weekday, one school week is five
# of them. MONTH, QUARTER and YEAR are *instructional*-day counts, and a real
# calendar month holds ~22 weekdays, not 20 — see CALENDAR_MONTHS_PER_PERIOD.
SCHOOL_DAYS_PER_PERIOD: dict[Period, int] = {
    Period.DAY: 1,
    Period.WEEK: 5,
    Period.MONTH: 20,
    Period.QUARTER: 45,
    Period.YEAR: 180,
}

# The periods whose school-day factor above is an approximation, expressed in
# the calendar unit the IEP actually means. "20 sessions per month" means per
# calendar month, and a window is capped at what that many calendar months can
# hold, so a long window cannot manufacture extra promised sessions out of the
# gap between 20 nominal instructional days and 22 real weekdays.
CALENDAR_MONTHS_PER_PERIOD: dict[Period, int] = {
    Period.MONTH: 1,
    Period.QUARTER: 3,
    Period.YEAR: 12,
}

# Service names are written by hand into IEPs, typed into parent logs, and
# exported by district systems, so the same service arrives spelled three ways.
# Matching is deliberately table-driven rather than fuzzy: a wrong match moves
# minutes between services, and this is the extension point for that risk.
_SERVICE_ALIASES: dict[str, str] = {
    "speech": "speech",
    "speech therapy": "speech",
    "speech language": "speech",
    "speech language therapy": "speech",
    "speech and language therapy": "speech",
    "speech language pathology": "speech",
    "speech language pathology services": "speech",
    "speech language services": "speech",
    "speech pathology": "speech",
    "language therapy": "speech",
    "slp": "speech",
    "st": "speech",
    "speech therapy st": "speech",
    "occupational therapy": "occupational therapy",
    "occupational therapy ot": "occupational therapy",
    "ot": "occupational therapy",
    "physical therapy": "physical therapy",
    "physical therapy pt": "physical therapy",
    "pt": "physical therapy",
    "counseling": "counseling",
    "individual counseling": "counseling",
    "psychological counseling": "counseling",
    "school counseling": "counseling",
    "specialized academic instruction": "specialized academic instruction",
    "special academic instruction": "specialized academic instruction",
    "sai": "specialized academic instruction",
}

# Tokens that carry no distinguishing meaning at the end of a service name.
_GENERIC_TAIL = frozenset({"service", "services"})

# Delivery-model qualifiers. Districts and IEPs append or prepend these to the
# same service ("Individual Speech Therapy", "Speech Therapy - Group"), and the
# obligation they reconcile against is the same one either way.
_QUALIFIERS = frozenset(
    {"individual", "individualized", "group", "direct", "indirect", "push", "pull"}
)

# Filler that survives punctuation stripping and would otherwise defeat the
# alias table ("Speech and Language Pathology").
_FILLER = frozenset({"and"})

# The second half of a two-word delivery model, dropped only after its own
# first half so that 'Push-In Speech Therapy' folds while a service genuinely
# named with 'in' or 'out' is left alone.
_QUALIFIER_TAILS = {"push": "in", "pull": "out"}

_PARENTHETICAL = re.compile(r"\([^)]*\)")

# Only a record produced by the district can retire a promised session from the
# "no record either way" bucket. Everything else — a parent's log, a documented
# silence — leaves the slot undocumented on purpose. See rule 3 at module top.
_DISTRICT_ANSWERABLE = frozenset({Provenance.SCHOOL_CONFIRMED})


def _normalize(service: str) -> str:
    """Fold a service name to lowercase words.

    Parenthetical abbreviations are dropped first — 'Occupational Therapy (OT)'
    is how a district export normally writes the service, and it must reconcile
    against an 'Occupational Therapy' obligation — then punctuation becomes
    whitespace, and filler words that carry no distinction are removed: 'and',
    and the tail of a two-word delivery model ('push in', 'pull out'), which is
    dropped only when its own first half precedes it so that a service genuinely
    named with 'in' or 'out' keeps the word. A name that is *only* a
    parenthetical keeps its contents rather than vanishing.
    """
    lowered = service.lower()
    stripped = _PARENTHETICAL.sub(" ", lowered)
    if not re.sub(r"[^a-z0-9]+", "", stripped):
        stripped = lowered
    words = re.sub(r"[^a-z0-9]+", " ", stripped).split()
    kept: list[str] = []
    for index, word in enumerate(words):
        if word in _FILLER:
            continue
        if index and _QUALIFIER_TAILS.get(words[index - 1]) == word:
            continue
        kept.append(word)
    return " ".join(kept or words)


def _strip_edges(tokens: list[str]) -> list[str]:
    """Drop generic tails and leading/trailing delivery-model qualifiers."""
    while len(tokens) > 1 and tokens[-1] in _GENERIC_TAIL:
        tokens.pop()
    changed = True
    while changed and len(tokens) > 1:
        changed = False
        if tokens[-1] in _QUALIFIERS:
            tokens.pop()
            changed = True
        if len(tokens) > 1 and tokens[0] in _QUALIFIERS:
            tokens.pop(0)
            changed = True
        while len(tokens) > 1 and tokens[-1] in _GENERIC_TAIL:
            tokens.pop()
            changed = True
    return tokens


def canonical_service(service: str) -> str:
    """Reduce a service name to the key events and obligations are matched on.

    Case, punctuation and whitespace never matter; a parenthetical abbreviation
    is dropped; a trailing 'service(s)' is dropped; delivery-model qualifiers
    ('individual', 'group', 'direct') are dropped from either end; and a known
    alias table folds the common spellings of one service together. So
    'Speech Therapy', 'Speech-Language Therapy (Direct)', 'Individual Speech
    Therapy' and 'speech/language services' all reconcile against the same
    obligation. An unrecognized name is its own key rather than a guess at a
    neighbour, because a wrong fold moves minutes between services.
    """
    normalized = _normalize(service)
    if not normalized:
        return ""
    if normalized in _SERVICE_ALIASES:
        return _SERVICE_ALIASES[normalized]

    stripped = " ".join(_strip_edges(normalized.split()))
    return _SERVICE_ALIASES.get(stripped, stripped or normalized)


def _school_days(start: date, end: date) -> int:
    """Weekdays in an inclusive date range."""
    if end < start:
        return 0
    total = (end - start).days + 1
    full_weeks, remainder = divmod(total, 7)
    days = full_weeks * 5
    first = start.weekday()
    days += sum(1 for offset in range(remainder) if (first + offset) % 7 < 5)
    return days


def _add_months(day: date, months: int) -> date:
    """``day`` shifted by whole months, clamped to the end of a short month."""
    index = day.month - 1 + months
    year = day.year + index // 12
    month = index % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def _calendar_months(start: date, end: date) -> Fraction:
    """Length of the inclusive span ``[start, end]`` in calendar months.

    Exact rational arithmetic on real month boundaries: Oct 5 .. Oct 30 is
    26/31 of a month, Mar 1 .. Mar 31 is exactly 1, and Aug 20 .. Jun 11 is
    9 + 23/31. Fractions rather than floats so that a whole number of months
    is exactly a whole number and never 0.999999.
    """
    stop = end + timedelta(days=1)  # measure the span half-open
    whole = (stop.year - start.year) * 12 + (stop.month - start.month)
    if _add_months(start, whole) > stop:
        whole -= 1
    anchor = _add_months(start, whole)
    following = _add_months(start, whole + 1)
    return Fraction(whole) + Fraction((stop - anchor).days, (following - anchor).days)


def _overlap(
    a_start: date, a_end: date, b_start: date, b_end: date
) -> tuple[date, date] | None:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return (start, end) if start <= end else None


def expected_sessions(obligation: ServiceObligation, start: date, end: date) -> int:
    """How many sessions the IEP promises between ``start`` and ``end``.

    The window is clipped to the obligation's own start and end dates first, so
    a service that began mid-window is only counted from the day it was owed.

    School-calendar assumptions, stated plainly because they bound how far this
    number may be trusted:

    * Sessions are counted in **school days**, meaning weekdays. Saturdays and
      Sundays are never counted.
    * **No holiday calendar is applied.** Breaks, snow days, staff development
      days and the summer recess inside the window are still counted as school
      days, because no district calendar is available to the ledger. For a DAY
      or WEEK obligation this therefore *over*-counts what was owed across a
      window containing a break, and that is the module's largest remaining
      source of overstatement. A caller holding a real calendar should
      reconcile over instruction-only sub-windows rather than expect this
      function to know.
    * A period is worth a fixed number of school days --- a week is 5, a month
      20, a quarter 45, a year 180 --- matching the factors in ``models.py``.
      For DAY and WEEK those factors are exact. For MONTH, QUARTER and YEAR
      they are *instructional*-day counts, and a real calendar month holds
      about 22 weekdays rather than 20, so weekday pro-rata alone would invent
      promised sessions on any window longer than one period (a 36-a-year
      service over a full school year would come out at 42). So those three
      periods are additionally **capped by the calendar**: at most
      ``sessions_per_period`` for every calendar month/quarter/year the window
      actually covers, rounded up to the next whole session so that a window
      covering most of a period is not zeroed out. The smaller of the two
      figures wins.
    * The result is **floored**, and every approximation above is resolved
      toward the smaller number. A short window never inflates the promise, and
      neither does a long one. The cost is deliberate under-statement in one
      direction: a QUARTER or YEAR obligation reconciled over the instructional
      part of the year is measured against calendar quarters and calendar
      years, so a 36-sessions-per-year service over Aug 20 .. Jun 11 reports 30
      owed, not 36. Understating what is owed understates the shortfall, which
      is the safe direction for a document that accuses a school.

    Returns 0 when the obligation is not in force anywhere in the window.

    Raises:
        ValueError: if ``end`` precedes ``start``.
    """
    if end < start:
        raise ValueError(f"window end {end} precedes start {start}")

    span = _overlap(obligation.start_date, obligation.end_date, start, end)
    if span is None:
        return 0

    per_period = obligation.sessions_per_period
    days = _school_days(*span)
    pro_rata = days * per_period // SCHOOL_DAYS_PER_PERIOD[obligation.period]

    months_per_period = CALENDAR_MONTHS_PER_PERIOD.get(obligation.period)
    if months_per_period is None:
        return pro_rata

    periods = _calendar_months(*span) / months_per_period
    return min(pro_rata, math.ceil(per_period * periods))


def _group_obligations(ledger: IEPLedger) -> dict[str, list[ServiceObligation]]:
    """Obligations keyed by canonical service, in ledger order.

    Two IEP lines for the same service (a mid-year amendment, say) reconcile as
    one grouped line so that delivered minutes are never double-counted against
    both, and so the Statement shows one row per service.
    """
    groups: dict[str, list[ServiceObligation]] = {}
    for obligation in ledger.obligations:
        groups.setdefault(canonical_service(obligation.service), []).append(obligation)
    return groups


def _in_force(obligations: list[ServiceObligation], on: date) -> bool:
    return any(o.start_date <= on <= o.end_date for o in obligations)


def _claims_delivery(event: ServiceEvent) -> bool:
    """Whether an event asserts that a session was actually delivered.

    A DOCUMENTED_SILENCE record never does, whatever its ``delivered`` flag
    says: silence is the absence of a district record, so it cannot confirm a
    delivery. Such an event is contradictory input, and the claimed delivery is
    disregarded rather than credited to the district.
    """
    return event.delivered and event.provenance is not Provenance.DOCUMENTED_SILENCE


def _evidence_ref(event: ServiceEvent) -> EvidenceRef:
    """A footnote-ready pointer to one observed fact.

    The wording states what a record shows, never what it proves, and it
    branches on provenance before ``delivered``: a documented silence speaks
    about the missing record, never about the session, which is the one claim
    silence can never establish.
    """
    if event.provenance is Provenance.DOCUMENTED_SILENCE:
        detail = (
            f"{event.event_date.isoformat()}: no {event.service} service record "
            f"was produced by the district for this date"
        )
        if event.delivered:
            detail += "; a delivery claimed on a silence record is not counted"
    elif event.delivered:
        detail = (
            f"{event.event_date.isoformat()}: {event.minutes} minutes of "
            f"{event.service} recorded as delivered"
        )
    else:
        detail = (
            f"{event.event_date.isoformat()}: {event.service} session recorded "
            f"as not delivered"
        )
    return EvidenceRef(provenance=event.provenance, source=event.source, detail=detail)


def unmatched_events(
    ledger: IEPLedger, events: list[ServiceEvent], start: date, end: date
) -> list[ServiceEvent]:
    """Events in the window that bind to no obligation in force on their date.

    Two things land here, and both are worth a parent's attention rather than a
    silent discard: a service the ledger never promised (a name the alias table
    does not know, or a service delivered outside the IEP), and a dated event
    falling outside the obligation's own start/end dates.

    An unmatched delivery is the dangerous case: its minutes are missing from
    the row, so the row overstates the shortfall. A caller that holds only a
    :class:`ReconciliationResult` can detect the same condition without calling
    this function --- ``events_considered`` exceeds the total number of
    evidence refs across the lines by exactly the number of unmatched events ---
    and should then call this to see what they were.
    """
    if end < start:
        raise ValueError(f"window end {end} precedes start {start}")

    groups = _group_obligations(ledger)
    return [
        event
        for event in events
        if start <= event.event_date <= end
        and not _in_force(groups.get(canonical_service(event.service), []), event.event_date)
    ]


def _delivered_by_date(matched: list[ServiceEvent]) -> dict[date, list[ServiceEvent]]:
    """The records that decide delivery on each date, duplicates resolved.

    One calendar date is one promised session slot, but it can carry several
    records: the district's service log and the parent's log both describing the
    same session, or the two flatly contradicting each other. Summing them all
    would credit one 30-minute session as 60 delivered minutes.

    So per date, the district's own record wins where there is one, and the
    parent's log is used only where the district has said nothing. Records that
    make no delivery claim at all (a documented silence) are excluded here; they
    remain evidence, they simply do not decide what was delivered.
    """
    by_date: dict[date, list[ServiceEvent]] = {}
    for event in matched:
        if event.provenance is Provenance.DOCUMENTED_SILENCE:
            continue
        by_date.setdefault(event.event_date, []).append(event)

    resolved: dict[date, list[ServiceEvent]] = {}
    for day, records in by_date.items():
        district = [e for e in records if e.provenance is Provenance.SCHOOL_CONFIRMED]
        resolved[day] = district or records
    return resolved


def reconcile(
    ledger: IEPLedger, events: list[ServiceEvent], start: date, end: date
) -> ReconciliationResult:
    """Reconcile promised minutes against evidence of delivery.

    One :class:`ServiceShortfall` is produced for every service in the ledger,
    including services that were owed nothing in this window --- a predictable
    row per service is worth more downstream than a tidy list.

    Per service:

    * ``owed_minutes`` --- :func:`expected_sessions` x minutes per session.
    * ``delivered_minutes`` --- minutes from in-window events that claim a
      delivery, whose service matches and whose date falls inside the
      obligation. Where one date carries more than one record, the district's
      own record decides and the parent's duplicate is not added on top; a
      delivery claimed on DOCUMENTED_SILENCE provenance is disregarded outright,
      because silence never confirms a delivery.
    * ``school_confirmed_minutes`` / ``parent_observed_minutes`` --- that same
      delivered total split by evidence grade. They always sum to
      ``delivered_minutes``.
    * ``undocumented_minutes`` --- promised sessions the district has produced
      no record for, in either direction, priced at the obligation's own
      per-session rate, rounded up, and capped at the shortfall. Delivered
      minutes in excess of the promise absorb it.
    * ``shortfall_minutes`` --- owed minus delivered, floored at zero. When a
      window holds no evidence at all this equals ``owed_minutes``; that is
      subtraction, not a finding of non-delivery, which is precisely why
      ``undocumented_minutes`` is reported beside it.

    WHICH RECORDS RETIRE A PROMISED SESSION, and why it is only one of the
    three grades:

    * SCHOOL_CONFIRMED --- yes. The district described the session, held or
      missed. That slot is documented and the shortfall it produces can be
      stated without a hedge.
    * PARENT_OBSERVED --- no. A parent's note that a session was missed is an
      allegation, not a district record; letting it retire the slot would move
      those minutes out of "undocumented" and into the plainly-stated shortfall,
      deleting the sentence that tells the district our count describes our
      records rather than what happened at school. The family's own unverified
      observation must never make the accusation sound better documented than
      it is. (A parent-observed *delivery* still counts in ``delivered_minutes``
      and so reduces the shortfall, which caps the undocumented figure anyway.)
    * DOCUMENTED_SILENCE --- no, by explicit contract with ``discovery.py``: a
      silence event documents the absence of a record, not the disposition of a
      session. Counting it as an accounted-for session would make exercising
      the parent's records right *shrink* the undocumented bucket, which is
      exactly backwards.

    A promised slot is a calendar date, not a record, so several records for one
    date retire one slot.

    ``period_start`` and ``period_end`` on each line are the window clipped to
    the obligation's own dates, falling back to the requested window for a
    service that was not in force at all.

    ``events_considered`` counts every event inside the window, matched or not;
    every matched event becomes one ``EvidenceRef`` on one line, so a caller can
    recover the unmatched count by subtraction. :func:`unmatched_events` returns
    the events themselves.

    Raises:
        ValueError: if ``end`` precedes ``start``.
    """
    if end < start:
        raise ValueError(f"window end {end} precedes start {start}")

    in_window = [event for event in events if start <= event.event_date <= end]
    shortfalls: list[ServiceShortfall] = []

    for obligations in _group_obligations(ledger).values():
        sessions_owed = 0
        owed_minutes = 0
        period_start: date | None = None
        period_end: date | None = None

        for obligation in obligations:
            count = expected_sessions(obligation, start, end)
            sessions_owed += count
            owed_minutes += count * obligation.minutes_per_session
            span = _overlap(obligation.start_date, obligation.end_date, start, end)
            if span is not None:
                period_start = span[0] if period_start is None else min(period_start, span[0])
                period_end = span[1] if period_end is None else max(period_end, span[1])

        if period_start is None or period_end is None:
            period_start, period_end = start, end

        key = canonical_service(obligations[0].service)
        matched = [
            event
            for event in in_window
            if canonical_service(event.service) == key
            and _in_force(obligations, event.event_date)
        ]

        deciding = _delivered_by_date(matched)
        delivered = [
            event
            for records in deciding.values()
            for event in records
            if _claims_delivery(event)
        ]
        delivered_minutes = sum(event.minutes for event in delivered)
        school_confirmed = sum(
            event.minutes for event in delivered
            if event.provenance is Provenance.SCHOOL_CONFIRMED
        )
        parent_observed = sum(
            event.minutes for event in delivered
            if event.provenance is Provenance.PARENT_OBSERVED
        )
        shortfall_minutes = max(0, owed_minutes - delivered_minutes)

        # One promised session slot is one calendar date. A slot is retired from
        # "no record either way" only by a record the district produced.
        documented_dates = {
            event.event_date
            for event in matched
            if event.provenance in _DISTRICT_ANSWERABLE
        }
        sessions_undocumented = max(0, sessions_owed - len(documented_dates))
        per_session = Fraction(owed_minutes, sessions_owed) if sessions_owed else Fraction(0)
        # Rounded up: a fraction of a minute belongs in "we cannot tell", never
        # in the documented shortfall.
        undocumented_minutes = min(
            shortfall_minutes, math.ceil(sessions_undocumented * per_session)
        )

        shortfalls.append(
            ServiceShortfall(
                service=obligations[0].service,
                period_start=period_start,
                period_end=period_end,
                owed_minutes=owed_minutes,
                delivered_minutes=delivered_minutes,
                shortfall_minutes=shortfall_minutes,
                school_confirmed_minutes=school_confirmed,
                parent_observed_minutes=parent_observed,
                undocumented_minutes=undocumented_minutes,
                evidence=[
                    _evidence_ref(event)
                    for event in sorted(matched, key=lambda e: (e.event_date, e.source))
                ],
            )
        )

    return ReconciliationResult(
        period_start=start,
        period_end=end,
        shortfalls=shortfalls,
        events_considered=len(in_window),
    )
