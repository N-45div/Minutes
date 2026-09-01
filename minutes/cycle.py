"""The heartbeat — one scheduled wake, and the semester replayed a week at a time.

A parent does not want an agent. They want the thing an agent makes possible:
somebody checked, on the day it mattered, and told them only when it mattered.
This module is that somebody. :func:`run_cycle` is one wake-up. It refreshes the
records-request clocks, turns any that have gone silent into dated facts,
reconciles the promise against the evidence, reads the statutory dates, and asks
:mod:`minutes.decisions` whether any of it is worth a parent's attention.

Most weeks the answer is no, and :attr:`CycleOutcome.quiet` says so out loud.
That is not the boring case to be tolerated — it is the product. A tool that
raises something every week trains a parent to stop reading it, and then the
annual review passes anyway. So a quiet week still reports what was checked:
trust in the silence comes from being able to see the work behind it.

**NO MODEL RUNS HERE.** Not one call, by design and not by omission. Whether to
escalate a child's legal claim has to be reproducible — the same ledger on the
same day must produce the same answer forever, a reviewer must be able to point
at the line that fired, and a test must be able to replay a whole semester in
milliseconds for nothing. The agent in :mod:`minutes.agent` reads unstructured
mail and writes the connective prose around this; the deciding is here, in code
you can read.

**SUPPRESSION IS WHY WEEKLY RUNNING IS TOLERABLE.** The same true fact is true
again next week. Without memory of what has already been raised, a district that
missed one deadline would generate an identical interruption every seven days
until the parent muted it. :func:`run_cycle` therefore takes what has already
been raised and shows a card again only once it has genuinely got worse — which
is what :mod:`minutes.decisions` designed its ``card_id`` scheme for: ids name
the underlying fact and deliberately exclude the observation date, so a
month-to-date window that moves every run does not mint a new id every run.

"Worse" is two things, not one, and getting that wrong is how a suppression
layer swallows the very week it exists for. A card is re-raised when its
``urgency`` has risen **or** its ``stage`` has — the ladder
:func:`minutes.decisions.staged_decisions_for` returns beside each card. Urgency
alone is not enough: a progress report is capped at ``time_sensitive`` for its
whole life, so on urgency alone the card saying *the date has passed* carries
the same value as the one saying *it is ten days away*, and the parent hears
about the report only while it is still in the future. See
:class:`RaisedCard`.

**THIS MODULE NEVER SENDS ANYTHING, AND NEVER ASKS A PARENT ANYTHING.** It
surfaces decisions; the parent answers them through the interrupt-gated tools in
:mod:`minutes.agent`. :func:`run_semester` takes a :data:`SendPolicy` for
exactly this reason — the replay needs a stand-in for the answer a parent would
have given, and making that a named, injected policy keeps it honest about being
a stand-in rather than quietly deciding on a family's behalf. The default is
:func:`send_nothing`, so a caller who has not thought about it gets the honest
floor rather than a simulated approval.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta

from pydantic import BaseModel, Field

from .deadlines import evaluate_deadlines
from .decisions import REQUEST_CADENCE_DAYS, StagedDecision, staged_decisions_for
from .discovery import due_requests, mark_sent, refresh_states, silence_events
from .models import (
    AuditEntry,
    DeadlineState,
    DeadlineStatus,
    DecisionCard,
    EvidenceRef,
    Provenance,
    ReconciliationResult,
    RecordsRequest,
    RequestState,
    ServiceEvent,
    Urgency,
)
from .reconcile import reconcile
from .tools import CaseRecord, load_case_record

__all__ = [
    "ACTOR",
    "QUIET_HEADLINE",
    "REPLAY_ACTOR",
    "CycleOutcome",
    "RaisedCard",
    "SendPolicy",
    "run_cycle",
    "run_semester",
    "send_nothing",
    "send_when_due",
]

ACTOR = "minutes-cycle"
"""The ``actor`` on every entry this module writes.

Distinct from :data:`minutes.agent.ACTOR` so a trail that mixes the scheduled
deterministic pass with the agent's own actions still says which one acted.
"""

REPLAY_ACTOR = "minutes-cycle-replay"
"""The ``actor`` on entries a :data:`SendPolicy` stand-in produced.

A third actor, and it earns its place. Everything :data:`ACTOR` writes is a fact
the cycle observed; everything under this one rests on a simulated answer from a
parent who was never asked. They must not be indistinguishable in a trail
somebody may one day read as a record of what a family consented to.
"""

QUIET_HEADLINE = "Nothing needs you. Minutes checked, and there is no decision to make."
"""The sentence most runs end on, and the one the product is judged by."""

# Lower is more urgent. Matches the ordering in minutes.decisions and
# minutes.statement, so a card ranks the same everywhere a parent might see it.
_URGENCY_RANK: dict[Urgency, int] = {
    Urgency.DEADLINE_IMMINENT: 0,
    Urgency.TIME_SENSITIVE: 1,
    Urgency.ROUTINE: 2,
}


class RaisedCard(BaseModel):
    """The worst a fact had got the last time the parent was shown it.

    Both halves are needed, and each covers a blind spot in the other. Urgency
    says how fast the thing must be dealt with, and it is capped per deadline
    kind, so it cannot express a progress report going from due-soon to missed.
    ``stage`` is :mod:`minutes.decisions`' own escalation ladder for the rule
    that raised the card, and it counts up only on a step that rule considers a
    genuinely new state of affairs.

    Stored as the *worst* of every rendering so far rather than the latest one,
    so a fact that escalates and then relaxes — an imminent deadline a later run
    rates merely time-sensitive — does not read as new again on the way back
    down.
    """

    urgency: Urgency
    stage: int = Field(
        default=0, ge=0, description="Rung of the rule's own escalation ladder"
    )


class CycleOutcome(BaseModel):
    """What one scheduled wake found, and what it decided to do about it.

    Defined here rather than in :mod:`minutes.models` because models.py is the
    frozen contract between engine modules and this is a report about running
    them, not a new kind of fact about a child's case.

    ``requests`` and ``raised`` are the state to carry into the next wake, which
    is what makes a cycle a pure function of what it was given. Nothing here
    reads a clock or writes a file: the same inputs produce the same outcome
    forever, which is what lets a whole semester be replayed in a test.
    """

    today: date
    period_start: date
    period_end: date
    checked: list[str] = Field(
        description="What was examined, in plain sentences. A quiet week still shows its work."
    )
    new_cards: list[DecisionCard] = Field(
        description="Decisions that need the parent now, most urgent first"
    )
    suppressed_cards: list[DecisionCard] = Field(
        description="True facts already raised at this urgency, deliberately not raised again"
    )
    due_requests: list[RecordsRequest] = Field(
        default_factory=list,
        description="Records requests the cadence says are due; drafts, never sent by this module",
    )
    requests: list[RecordsRequest] = Field(
        default_factory=list,
        description=(
            "Request state to carry into the next cycle, including any send a "
            "SendPolicy stand-in made on this wake"
        ),
    )
    silence: list[ServiceEvent] = Field(
        default_factory=list,
        description="Documented-silence facts derived from overdue requests this run",
    )
    events_considered: int = Field(
        default=0,
        description="Records that fell inside the reconciled window, matched or not",
    )
    raised: dict[str, RaisedCard] = Field(
        default_factory=dict,
        description=(
            "Every suppression key ever raised, at the worst urgency and stage it reached"
        ),
    )
    audit: list[AuditEntry] = Field(default_factory=list)

    @property
    def quiet(self) -> bool:
        """True when nothing needs the parent. The normal, good result.

        A suppressed card does not break quiet: the parent has already seen that
        fact and nothing about it has got worse, so there is nothing new to say.
        """
        return not self.new_cards

    @property
    def headline(self) -> str:
        """One sentence: either nothing needs you, or how much does."""
        if self.quiet:
            return QUIET_HEADLINE
        count = len(self.new_cards)
        verb = "decision needs" if count == 1 else "decisions need"
        return f"{count} {verb} you."

    def summary(self) -> str:
        """The headline, then the work behind it, then the decisions themselves."""
        lines = [self.headline, ""]
        lines.extend(f"  - {item}" for item in self.checked)
        for card in self.new_cards:
            lines.extend(["", f"  [{card.urgency.value}] {card.title}", f"      {card.why_now}"])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# One wake.
# ---------------------------------------------------------------------------


def run_cycle(
    today: date,
    *,
    case: CaseRecord | None = None,
    requests: Sequence[RecordsRequest] | None = None,
    raised: Mapping[str, RaisedCard] | None = None,
    period_start: date | None = None,
) -> CycleOutcome:
    """One scheduled wake-up, start to finish, with no model in the loop.

    The order is not arbitrary. Request clocks are refreshed first, because a
    request that went overdue overnight becomes evidence in this run rather than
    the next one. Silence facts are derived next, so reconciliation and the
    letter compilers can see them. Only then is the ledger reconciled, the
    statutory dates read, and the decision rules asked.

    Args:
        today: The date being woken on. Every clock in the run is read against
            it, and nothing here consults the real one.
        case: The family's case. Defaults to the committed demo case through
            :func:`minutes.tools.load_case_record`, the project's single data
            seam.
        requests: The records-request state carried from the previous wake.
            Defaults to the case's own history.
        raised: ``suppression key -> RaisedCard`` for every card already put in
            front of this parent, at the worst it reached. Cards in here are
            suppressed unless they have since got more urgent or moved up their
            rule's escalation ladder.
        period_start: First day of the service period to reconcile. Defaults to
            the earliest obligation start date — everything owed so far —
            because a decision waiting today is weighed against the whole
            record, not an arbitrary recent slice.
    """
    case = case or load_case_record()
    ledger = case.ledger
    carried = list(requests if requests is not None else case.requests)
    already = dict(raised or {})

    starts = [obligation.start_date for obligation in ledger.obligations]
    first_day = period_start or (min(starts) if starts else None)
    if first_day is None or first_day > today:
        return _nothing_started(today, first_day, carried, already)

    audit: list[AuditEntry] = []
    checked: list[str] = []

    # 1. The request clocks. A request whose response window closed overnight
    #    becomes evidence today, not next week.
    current = refresh_states(carried, today)
    newly_overdue = [
        request.request_id
        for before, request in zip(carried, current)
        if before.state is not request.state
    ]
    outstanding = [request for request in current if request.state in _OUTSTANDING]
    checked.append(_requests_checked(current, outstanding))
    audit.append(
        AuditEntry(
            entry_date=today,
            actor=ACTOR,
            action="checked the records requests",
            detail=(
                f"{len(current)} request(s) on file; {len(outstanding)} outstanding. "
                + (
                    f"Newly past their response deadline: {', '.join(newly_overdue)}."
                    if newly_overdue
                    else "None passed a response deadline since the last check."
                )
            ),
        )
    )

    # 2. Silence facts. This is a FULL RECOMPUTATION every run, so it REPLACES
    #    the previous set and is never appended to a growing list -- appending
    #    would multiply one silence fact into many and, downstream, shrink the
    #    undocumented bucket that keeps a shortfall honest.
    silence = silence_events(current, ledger, today)
    for request_id in sorted({event.source for event in silence}):
        due = next(event.event_date for event in silence if event.source == request_id)
        audit.append(
            AuditEntry(
                entry_date=today,
                actor=ACTOR,
                action="recorded documented silence",
                detail=(
                    f"Records request {request_id} was due {due.isoformat()} and no "
                    "response is recorded. That is a dated fact about the records: the "
                    "district produced none within the time 34 CFR 300.613(a) allows. It "
                    "is not a fact about whether any session happened."
                ),
                evidence=EvidenceRef(
                    provenance=Provenance.DOCUMENTED_SILENCE,
                    source=request_id,
                    detail=(
                        f"{due.isoformat()}: no records produced in response to "
                        f"{request_id} by the date they were due."
                    ),
                ),
            )
        )
    if silence:
        checked.append(
            f"Derived {len(silence)} documented-silence fact(s) from "
            f"{len({event.source for event in silence})} unanswered request(s)."
        )

    # 3. The arithmetic, over the base evidence plus this run's silence facts.
    result = reconcile(ledger, [*case.events, *silence], first_day, today)
    checked.append(_reconciliation_checked(result))
    audit.append(
        AuditEntry(
            entry_date=today,
            actor=ACTOR,
            action="reconciled the ledger",
            detail=(
                f"{first_day.isoformat()} to {today.isoformat()}: "
                f"{sum(line.owed_minutes for line in result.shortfalls)} minutes promised, "
                f"{sum(line.delivered_minutes for line in result.shortfalls)} documented as "
                f"delivered, {result.total_shortfall_minutes} short, of which "
                f"{sum(line.undocumented_minutes for line in result.shortfalls)} have no "
                "record either way."
            ),
        )
    )

    # 4. The statutory clocks.
    deadlines = evaluate_deadlines(ledger, today)
    checked.append(_deadlines_checked(deadlines))

    # 5. The decision rules, and only then the suppression.
    staged = staged_decisions_for(ledger, result, deadlines, current, today)
    new_cards, suppressed, updated = _partition(staged, already)
    audit.extend(_decision_entries(today, staged, new_cards, already))

    pending = due_requests(ledger, current, today, cadence_days=REQUEST_CADENCE_DAYS)

    return CycleOutcome(
        today=today,
        period_start=first_day,
        period_end=today,
        checked=checked,
        new_cards=new_cards,
        suppressed_cards=suppressed,
        due_requests=pending,
        requests=current,
        silence=silence,
        events_considered=result.events_considered,
        raised=updated,
        audit=audit,
    )


_OUTSTANDING = (RequestState.SENT, RequestState.UNANSWERED_OVERDUE)


def _nothing_started(
    today: date,
    first_day: date | None,
    carried: list[RecordsRequest],
    already: dict[str, RaisedCard],
) -> CycleOutcome:
    """The wake before any service was owed. Quiet, and honest about why.

    Reconciling a window that ends before the IEP begins would report the whole
    promise as a shortfall — arithmetically true and completely misleading — so
    the run stops here and says what it found instead of computing it.
    """
    note = (
        "No IEP service had started by this date, so nothing could be owed yet."
        if first_day is not None
        else "This IEP promises no services, so there is nothing to reconcile."
    )
    return CycleOutcome(
        today=today,
        period_start=first_day or today,
        period_end=today,
        checked=[note],
        new_cards=[],
        suppressed_cards=[],
        requests=list(carried),
        raised=dict(already),
        audit=[
            AuditEntry(
                entry_date=today,
                actor=ACTOR,
                action="woke and found nothing owed yet",
                detail=note,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Suppression.
# ---------------------------------------------------------------------------


def _partition(
    staged: Sequence[StagedDecision], raised: Mapping[str, RaisedCard]
) -> tuple[list[DecisionCard], list[DecisionCard], dict[str, RaisedCard]]:
    """Split cards into what the parent has not seen and what they have.

    A card is new if its suppression key has never been raised, or if it has
    since got worse than the worst it ever reached — MORE urgent, or further up
    its rule's escalation ladder. Both halves matter. Re-raising an unchanged
    fact every week is how a product teaches a parent to ignore it; never
    re-raising means a deadline that slid from "due soon" to "passed" goes
    unmentioned precisely when it finally matters.

    THE LADDER IS NOT DECORATION. Comparing urgency alone looks sufficient and
    is not: :data:`minutes.decisions.URGENCY_CAP_BY_KIND` pins a progress report
    at ``time_sensitive`` from the day it enters its lead-time window until the
    end of the school year, so on urgency alone the card that says the report
    never arrived is worth exactly what the card that said it was coming was
    worth, and is held back forever. An annual review has the opposite problem:
    it is already ``deadline_imminent`` a week out, so passing the date raises
    nothing either. ``stage`` is what carries those transitions.

    What is stored is the worst of every rendering, never the latest, so a fact
    that escalates and then relaxes does not read as new on the way back down.
    """
    updated = dict(raised)
    new_cards: list[DecisionCard] = []
    suppressed: list[DecisionCard] = []

    for card, key, stage in staged:
        seen = updated.get(key)
        worse = seen is None or (
            _URGENCY_RANK[card.urgency] < _URGENCY_RANK[seen.urgency] or stage > seen.stage
        )
        if worse:
            new_cards.append(card)
            updated[key] = RaisedCard(
                urgency=_more_urgent(card.urgency, seen.urgency if seen else None),
                stage=max(stage, seen.stage if seen else 0),
            )
        else:
            suppressed.append(card)

    return new_cards, suppressed, updated


def _more_urgent(urgency: Urgency, previous: Urgency | None) -> Urgency:
    if previous is None:
        return urgency
    return urgency if _URGENCY_RANK[urgency] < _URGENCY_RANK[previous] else previous


def _decision_entries(
    today: date,
    staged: Sequence[StagedDecision],
    new_cards: Sequence[DecisionCard],
    before: Mapping[str, RaisedCard],
) -> list[AuditEntry]:
    """The trail's record of what was raised, what was held back, and why.

    A re-raise names which dimension moved. "It was already raised at
    time_sensitive and is still time_sensitive" would read as a contradiction of
    the suppression rule to anybody reading the trail, when in fact the fact
    moved up its own ladder — from a date approaching to a date missed.
    """
    entries: list[AuditEntry] = []
    raised_ids = {card.card_id for card in new_cards}

    for card, key, stage in staged:
        previous = before.get(key)
        if card.card_id in raised_ids:
            entries.append(
                AuditEntry(
                    entry_date=today,
                    actor=ACTOR,
                    action="raised a decision" if previous is None else "raised a decision again",
                    detail=(
                        f"{card.card_id} — {card.title}. "
                        + (
                            f"It is new: {card.why_now}"
                            if previous is None
                            else _why_again(card, previous, stage)
                        )
                    ),
                )
            )
        else:
            entries.append(
                AuditEntry(
                    entry_date=today,
                    actor=ACTOR,
                    action="held back a decision already raised",
                    detail=(
                        f"{card.card_id} — {card.title}. Still true, still "
                        f"{card.urgency.value}, and the parent has already seen it, so it was "
                        "not put in front of them again."
                    ),
                )
            )

    if not staged:
        entries.append(
            AuditEntry(
                entry_date=today,
                actor=ACTOR,
                action="found nothing that needs the parent",
                detail=QUIET_HEADLINE,
            )
        )

    return entries


def _why_again(card: DecisionCard, previous: RaisedCard, stage: int) -> str:
    """The sentence explaining what changed about a fact already raised."""
    if _URGENCY_RANK[card.urgency] < _URGENCY_RANK[previous.urgency]:
        return (
            f"It was already raised at {previous.urgency.value}, and has become "
            f"{card.urgency.value}."
        )
    return (
        f"It was already raised at {previous.urgency.value}, and while that has not "
        f"changed the fact itself has moved on (escalation step {previous.stage} to "
        f"{stage}): {card.why_now}"
    )


# ---------------------------------------------------------------------------
# What the run looked at, in sentences a parent can read.
# ---------------------------------------------------------------------------


def _requests_checked(
    current: Sequence[RecordsRequest], outstanding: Sequence[RecordsRequest]
) -> str:
    if not current:
        return "No records request has been made on this case yet."
    overdue = sum(1 for request in current if request.state is RequestState.UNANSWERED_OVERDUE)
    return (
        f"Checked {len(current)} records request(s): {len(outstanding)} awaiting a response, "
        f"{overdue} past the date it was due."
    )


def _reconciliation_checked(result: ReconciliationResult) -> str:
    """The record count comes from the reconciliation, not from a parallel sum.

    Counting the inputs here instead would make this sentence agree with itself
    while disagreeing with the arithmetic it describes — and a silence set that
    was appended rather than replaced would be invisible in exactly the line
    written to make it visible.
    """
    owed = sum(line.owed_minutes for line in result.shortfalls)
    delivered = sum(line.delivered_minutes for line in result.shortfalls)
    return (
        f"Reconciled {len(result.shortfalls)} service(s) against "
        f"{result.events_considered} record(s) for {result.period_start.isoformat()} to "
        f"{result.period_end.isoformat()}: {owed} minutes promised, {delivered} "
        "documented as delivered."
    )


def _deadlines_checked(deadlines: Sequence[DeadlineStatus]) -> str:
    overdue = sum(1 for status in deadlines if status.state is DeadlineState.OVERDUE)
    due_soon = sum(1 for status in deadlines if status.state is DeadlineState.DUE_SOON)
    return (
        f"Read {len(deadlines)} date(s) the IEP sets: {due_soon} coming up, {overdue} past."
    )


# ---------------------------------------------------------------------------
# The semester replay.
# ---------------------------------------------------------------------------

SendPolicy = Callable[[RecordsRequest, date], bool]
"""Stands in for the parent's answer to "shall I send this records request?".

In the product that answer comes from a human, through the interrupt in
:func:`minutes.agent.send_records_request`. A replay has no human, so it needs a
policy — and making it an explicit, injected parameter is the difference between
a simulation that says what it is and a system that quietly decides for a
family.

WHAT A POLICY THAT RETURNS ``True`` IS ASSUMING, stated once so no reader has to
infer it: that a parent approved, that they posted the request, and that the
district received it the same day. The real product cannot assume any of the
three — :func:`minutes.agent.send_records_request` puts the letter in an outbox
and waits for the family to confirm delivery, because the statutory clock runs
from receipt and Minutes operates no mail channel. A replay collapses all of
that into one day so a term fits in a test, which is precisely why every entry
it produces is written under :data:`REPLAY_ACTOR`.
"""


def send_when_due(request: RecordsRequest, today: date) -> bool:
    """A parent who approves every records request, and a district that receives it at once.

    Not the default, deliberately — see :func:`run_semester`. Pass it explicitly
    to see the mechanism that makes Minutes different from a filing cabinet:
    asking on a cadence produces evidence either way, and a request that goes
    unanswered becomes a dated fact the family could never have created by
    watching.
    """
    return True


def send_nothing(request: RecordsRequest, today: date) -> bool:
    """A parent who never answers. The honest floor of what the cycle does alone."""
    return False


def run_semester(
    start: date,
    end: date,
    step_days: int = 7,
    *,
    case: CaseRecord | None = None,
    sends: SendPolicy = send_nothing,
) -> list[CycleOutcome]:
    """Replay a term one wake at a time, carrying state forward between them.

    This is the demo, and it is free: no model, no network, no credentials, so
    a whole semester runs in milliseconds and produces the same arc every time.
    What that arc shows is the argument for the product — long runs of quiet
    weeks, and then a week where something genuinely changed.

    State threads through in exactly the two places a real deployment threads
    it: the records-request state machine, and the ids already raised. Each wake
    is otherwise a pure function of what it is handed.

    Args:
        start: The first wake.
        end: The last day a wake may fall on.
        step_days: Days between wakes; 7 is a weekly schedule.
        case: The family's case, defaulting to the committed demo case.
        sends: The stand-in for the parent's approval, defaulting to
            :func:`send_nothing` — nobody answered, so nothing went out. A
            caller who wants the full arc passes :func:`send_when_due` and, by
            typing it, says out loud that the approvals in the replay are
            simulated. Defaulting the other way would mean a caller who never
            thought about it got a term of fabricated consent.

    Raises:
        ValueError: if ``step_days`` is less than 1 (a non-positive step is not
            "check continuously", it is an infinite loop) or ``end`` precedes
            ``start``.
    """
    if step_days < 1:
        raise ValueError(f"step_days must be at least 1, got {step_days}")
    if end < start:
        raise ValueError(f"end {end.isoformat()} precedes start {start.isoformat()}")

    case = case or load_case_record()
    requests = list(case.requests)
    raised: dict[str, RaisedCard] = {}
    outcomes: list[CycleOutcome] = []

    today = start
    while today <= end:
        outcome = run_cycle(today, case=case, requests=requests, raised=raised)
        requests = list(outcome.requests)
        raised = dict(outcome.raised)

        for request in outcome.due_requests:
            if not sends(request, today):
                continue
            sent = mark_sent(request, today)
            requests = [r for r in requests if r.request_id != sent.request_id] + [sent]
            # Recorded on the cycle that raised it, because that is the week a
            # parent would have been asked and would have answered.
            due = sent.response_due.isoformat() if sent.response_due else "unrecorded"
            outcome.audit.append(
                AuditEntry(
                    entry_date=today,
                    # Never ACTOR. The sentence below describes a send that did
                    # not happen, and a reader must be able to tell it from the
                    # observations around it by the actor alone.
                    actor=REPLAY_ACTOR,
                    action="simulated a records request going out",
                    detail=(
                        f"{sent.request_id} covering {sent.covers_start.isoformat()} to "
                        f"{sent.covers_end.isoformat()} was released by a stand-in send "
                        "policy in a replay: no parent was asked, nothing was posted, and "
                        "the district received nothing. The response clock is run from "
                        f"{today.isoformat()} so the replay has a date, making a response "
                        f"notionally due {due}."
                    ),
                )
            )

        # The field says "the state to carry into the next cycle", so it has to
        # include a send made on this wake. Without this a caller persisting the
        # last outcome would lose that send and re-issue the same request.
        outcome.requests = list(requests)
        outcomes.append(outcome)
        today += timedelta(days=step_days)

    return outcomes
