"""When the parent is worth interrupting — and what to put in front of them.

Every other module in this package answers "what is true?". This one answers
the only question the parent actually experiences: *does this need me today?*
Most weeks the honest answer is no, and saying so is the feature. A tool that
raises a card every week trains the parent to close it unread, and then the
annual review passes anyway. So the bar to interrupt is set deliberately high
and every threshold that sets it is a named constant with its reasoning
attached.

**No LLM call happens here, on purpose.** Whether to escalate a child's legal
claim must be reproducible, inspectable and testable — the same ledger on the
same day must produce the same cards forever, and a reviewer must be able to
point at the line of code that fired. A model's judgement about "does this feel
serious" is none of those things. This module is rules over the engine's own
output; the only prose it writes is the plain-language explanation of a fact
the engine already established.

FOUR RULES EARN AN INTERRUPT, and nothing else does:

1. :func:`_silence_cards` — a records request went unanswered past its
   statutory response window. The silence has become evidence the parent did
   not have before.
2. :func:`_deadline_cards` — a date the IEP itself sets is inside its lead-time
   window, or has passed with no evidence it was met.
3. :func:`_shortfall_cards` — the *documented* part of a service gap has
   crossed :data:`MATERIAL_SHORTFALL_PERCENT` and
   :data:`MATERIAL_SHORTFALL_MINUTES`.
4. :func:`_records_due_cards` — the discovery cadence has come round and a
   request is due to go out.

THE LINE RULE 3 MUST NOT CROSS. Undocumented minutes never, on their own,
raise a shortfall card. A gap made of minutes nobody wrote anything down about
is a hole in the *parent's* records, and the answer to a hole in your records
is to ask for records (rule 4), not to accuse a district of something you
cannot show. Rule 3 therefore measures ``shortfall_minutes -
undocumented_minutes`` — the part of the gap a district record already speaks
to — and never the raw shortfall. This is the same quantity
:func:`minutes.letters._is_evidenced_gap` gates escalation letters on, with one
deliberate difference: that function also escalates on a documented silence
standing where a service log should be, and here that case belongs to rule 1,
which states it as what it is (a records failure) instead of dressing it up as
a service gap. Rule 3 is the stricter of the two tests, and the strictness is
the point.

CARD IDENTITY, AND WHY SUPPRESSION NEEDS IT. ``card_id`` names the *fact*
that drove the card, so a caller can suppress a card it has already shown the
parent by comparing ids across runs. The scheme is
``"<rule-slug>:<subject>[:<fixed-date>]"``, and it obeys one rule that makes
weekly suppression actually work:

    **A card_id never embeds an observation date.** Not ``today``, and not a
    window end that tracks ``today``. Only dates the underlying fact itself
    fixes.

That is why the shortfall id carries ``period_start`` but not ``period_end``,
and the cadence id carries ``covers_start`` but not ``covers_end``: a caller
reconciling month-to-date moves both ends' *end* every single run, and an id
built on it would change weekly and suppress nothing. Concretely:

* ``records-silence:{request_id}:{response_due}``
* ``deadline:{kind}:{due}:{description}`` — via
  :func:`minutes.deadlines.deadline_key`, which carries the description
  because kind and date alone are not unique within one IEP.
* ``shortfall:{period_start}:{service}+{service}...`` — canonical service
  names (:func:`minutes.reconcile.canonical_service`) sorted, so a district
  export that respells a service does not mint a second card for it.
* ``records-due:{covers_start}``

``card_id`` is deliberately **independent of state and of urgency**. A deadline
that was ``DUE_SOON`` last month and is ``OVERDUE`` today is the same fact and
keeps its id; what changed is how bad it is. Encoding the state into the id
instead would mean every escalation looked like a brand-new fact, and the parent
would be re-interrupted by things they had already dealt with.

WHY URGENCY ALONE CANNOT DRIVE SUPPRESSION, AND WHAT ``stage`` IS FOR. A caller
comparing only ``urgency`` across runs silently loses the escalations that
matter most. A progress report is capped at ``TIME_SENSITIVE``
(:data:`URGENCY_CAP_BY_KIND`) for its whole life, so "due in ten days" and "that
date has passed three weeks ago" carry the identical urgency and the second one
never reaches the parent. An annual review is already ``DEADLINE_IMMINENT`` a
week before the date, so going overdue does not move it either. And a documented
gap that grows by a factor of a hundred stays ``TIME_SENSITIVE`` throughout.
Urgency answers "how fast must this be dealt with"; it was never a measure of
how far a fact has travelled.

So :func:`staged_decisions_for` returns, beside each card, a ``stage``: a small
integer that starts at zero and only ever counts UP as the underlying fact gets
worse. A caller re-raises a card when either its urgency has risen or its stage
has. The ladders are per-rule and each one is a named, testable step:

* silence — one step per further :data:`~minutes.discovery.RESPONSE_WINDOW_DAYS`
  the district stays silent past the date it was due.
* deadline — ``0`` while the date is merely approaching, ``1`` once it has
  passed. This is the step the capped progress report was losing.
* shortfall — one step per doubling of the documented gap above the material
  floor (:data:`SHORTFALL_ESCALATION_FACTOR`), plus one per additional service
  that has crossed the bar.
* records due — no ladder. A request is either due or it is not, and it stops
  being due the moment it goes out.

Beside ``stage`` each card also carries a ``suppression_key``, which is the
``card_id`` for every rule but one. The exception is rule 3: its id names the
exact set of short services, so that a service *joining* the gap reads as the
new fact it is — but that also means a service *improving out* of the set mints
a fresh id, and interrupting a parent because the news got better is the same
alert fatigue by another route. So shortfall cards suppress on
``shortfall:{period_start}`` — one running conversation about one period — while
the breadth of the gap is carried in the stage, where growth escalates and
shrinkage does not.

THE DRAFT GATE RUNS HERE TOO. A compiled letter is attached to a card only
once :func:`minutes.letters.validate_letter` returns clean (:func:`_gated`). A
letter that cannot footnote its own claims must never reach the parent, and
dropping the draft while keeping the card is the right failure: the fact behind
the card is still true and still worth surfacing, it just travels without a
letter. No card's ``recommended_action`` promises a draft for this reason.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

from .deadlines import deadline_key, urgency_for
from .discovery import RESPONSE_WINDOW_DAYS, due_requests, refresh_states
from .letters import (
    compile_deadline_reminder,
    compile_records_request,
    compile_shortfall_notice,
    validate_letter,
)
from .models import (
    DeadlineKind,
    DeadlineState,
    DeadlineStatus,
    DecisionCard,
    IEPLedger,
    Letter,
    ReconciliationResult,
    RecordsRequest,
    RequestState,
    ServiceShortfall,
    Urgency,
)
from .reconcile import canonical_service

__all__ = [
    "MATERIAL_SHORTFALL_MINUTES",
    "MATERIAL_SHORTFALL_PERCENT",
    "MEETING_DEADLINE_KINDS",
    "RECORDS_BEFORE_MEETING_DAYS",
    "REQUEST_CADENCE_DAYS",
    "SHORTFALL_ESCALATION_FACTOR",
    "URGENCY_CAP_BY_KIND",
    "StagedDecision",
    "capped_urgency",
    "crosses_material_bar",
    "decisions_for",
    "documented_shortfall_minutes",
    "is_quiet",
    "meeting_pressure",
    "staged_decisions_for",
]


# ---------------------------------------------------------------------------
# The thresholds. Every number that decides whether a parent is interrupted
# lives here, with the reasoning that chose it.
# ---------------------------------------------------------------------------

MATERIAL_SHORTFALL_PERCENT = 10
"""Share of a period's promised minutes the documented gap must reach.

There is no legal percentage to copy, and pretending otherwise would be the
dishonest move. The controlling standard is *material failure to implement*:
Van Duyn ex rel. Van Duyn v. Baker Sch. Dist. 5J, 502 F.3d 811, 822 (9th Cir.
2007) holds that a shortfall is actionable when there is more than a minor
discrepancy between the services an IEP requires and the services delivered,
and the court expressly declined to fix a number to it. Houston Indep. Sch.
Dist. v. Bobby R., 200 F.3d 341, 349 (5th Cir. 2000) asks the same question as
whether the failure was significant. Both are individualized inquiries for an
adjudicator.

So 10% is **a product threshold for when to interrupt a parent, not a legal
test**, and nothing this module emits ever characterizes a gap as material —
that word appears in this docstring and in the constant's name, never in a
card. What 10% buys is a floor beneath which the difference is indistinguishable
from ordinary scheduling noise: a session moved across a period boundary, a
provider out sick, a shortened session. Interrupting a parent about that trains
them to ignore the interruption that matters.

Measured against ``owed_minutes`` for the service, using the documented gap
only (:func:`documented_shortfall_minutes`).
"""

MATERIAL_SHORTFALL_MINUTES = 60
"""Absolute floor the documented gap must also clear.

A percentage alone misfires on small services. A service promising 120 minutes
in a month crosses 10% at 12 minutes — less than half of one session, and quite
possibly a session that ran short rather than one that did not happen. A card
should always correspond to at least one identifiable session's worth of
service, and 60 minutes is one typical related-service session (related
services are commonly written at 30 to 60 minutes each).

Both thresholds must be met, never either. Requiring both is the conservative
composition, and conservative is correct here: the cost of a missed card is one
delayed month, while the cost of a card that turns out to be noise is a parent
who stops reading them.
"""

REQUEST_CADENCE_DAYS = 30
"""How much uncovered service time accrues before records are asked for again.

Matches :func:`minutes.discovery.due_requests`' own default and is named here
so that every threshold governing an interruption is readable in one file. A
month is the shortest cadence that is not harassment and the longest that keeps
each request's window small enough for a district to answer without a project.
"""

RECORDS_BEFORE_MEETING_DAYS = RESPONSE_WINDOW_DAYS
"""How close an IEP meeting must be before outstanding records turn urgent.

34 CFR 300.613(a) stacks three deadlines and the earliest governs: records are
owed without unnecessary delay, *before any meeting regarding an IEP*, and in
no case more than 45 days. That middle clause is the one that bites here.

The value is derived rather than picked. Once a meeting is nearer than the
45-day ceiling a district may take on a fresh request, the parent can no longer
fix the problem by simply asking again — a new request would come due after the
meeting has happened. The outstanding request is the only one that can still
land in time, which is exactly when an unanswered one stops being a routine
follow-up and becomes the thing standing between the parent and walking into a
meeting without the district's own service logs.
"""

MEETING_DEADLINE_KINDS: frozenset[DeadlineKind] = frozenset(
    {DeadlineKind.ANNUAL_REVIEW, DeadlineKind.REEVALUATION}
)
"""Deadline kinds that are "a meeting regarding an IEP" for the rule above.

An annual review is one on its face. A reevaluation culminates in a team
meeting to review the results, so records owed before that meeting are owed on
the same clause. A progress report is a document that arrives in the post; it
is not a meeting and creates no records pressure, so it is not here.
"""

URGENCY_CAP_BY_KIND: dict[DeadlineKind, Urgency] = {
    DeadlineKind.PROGRESS_REPORT: Urgency.TIME_SENSITIVE,
}
"""Ceilings applied on top of :func:`minutes.deadlines.urgency_for`.

That function maps any overdue deadline to ``DEADLINE_IMMINENT``, which is
right for the dates it was written for: an annual review or a reevaluation that
has slipped is an opportunity actively lapsing, and there is something to
schedule today. A progress report is different in kind. It requires nothing of
the parent in advance, there is no meeting to arrange and no consent to give,
and a report that has not arrived is not an opportunity closing — it is a
document to chase. Ranking it beside a lapsed annual review would flatten the
distinction that makes the urgency field worth reading at all.

Caps only ever make a card *less* urgent. This module never raises an urgency
above what the engine computed.

Because a cap flattens a whole kind onto one value, it also flattens the only
signal a suppression layer could read from urgency alone. That is why a
deadline card carries a ``stage`` as well: see the module docstring, and
:func:`staged_decisions_for`.
"""

SHORTFALL_ESCALATION_FACTOR = 2
"""How far a documented gap must grow before the parent hears about it again.

A doubling, and the choice is the same conservatism as the bars above. A gap
that inches from 61 minutes to 74 is the same conversation the parent already
had; a gap that has doubled since they last saw it is a different one, and the
letter they were shown no longer describes the case. Anything smaller than a
doubling would put a growing shortfall back in front of them most weeks, which
is the interruption this module exists to prevent, and anything larger would
let a gap widen for most of a school year in silence.

Counted in whole steps above :data:`MATERIAL_SHORTFALL_MINUTES` rather than
against the first figure raised, so the ladder is a property of the case and not
of the day a caller happened to start watching.
"""


# Rank for ordering and for capping. Lower is more urgent, matching
# minutes.statement's own ordering so a card sorts the same way here and in the
# artifact the parent finally reads.
_URGENCY_RANK: dict[Urgency, int] = {
    Urgency.DEADLINE_IMMINENT: 0,
    Urgency.TIME_SENSITIVE: 1,
    Urgency.ROUTINE: 2,
}

# Why a deadline of this kind needs its lead time, in the parent's words. The
# numeric windows themselves belong to minutes.deadlines; these are the
# sentences that explain to a tired reader why they are hearing about a date
# that is still weeks away.
_LEAD_TIME_REASONS: dict[DeadlineKind, str] = {
    DeadlineKind.ANNUAL_REVIEW: (
        "Raising it now leaves time to get the district's service records before the "
        "meeting and to agree a date and time you can actually attend."
    ),
    DeadlineKind.REEVALUATION: (
        "A reevaluation needs your written consent before any testing begins, then the "
        "assessments, then the team meeting — so the useful time is now, not in the week "
        "before the date."
    ),
    DeadlineKind.PROGRESS_REPORT: (
        "A report that does not arrive is easy to forget by the time the next one is due, "
        "and the gap in the record is hard to reconstruct later."
    ),
    DeadlineKind.OTHER: (
        "The IEP records this date, so it is tracked and raised in time to act on it."
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class StagedDecision(NamedTuple):
    """A card, plus everything a suppression layer needs to decide about it.

    ``card`` is what the parent reads. ``suppression_key`` names the running
    conversation the card belongs to, and ``stage`` says how far along that
    conversation this rendering sits — see the module docstring for the ladders
    and for why urgency alone was never enough.

    A caller shows a card again when its urgency has risen *or* its stage has,
    and otherwise holds it back.
    """

    card: DecisionCard
    suppression_key: str
    stage: int


def documented_shortfall_minutes(line: ServiceShortfall) -> int:
    """The part of a service's gap the district's own records already speak to.

    ``shortfall_minutes - undocumented_minutes``, floored at zero. This is the
    only quantity rule 3 is allowed to weigh, and separating it out gives the
    distinction a name that can be tested directly.

    The subtraction is what keeps absence of evidence from becoming evidence.
    Undocumented minutes are promised minutes with no record in either
    direction — reconciliation is explicit that reporting them beside the
    shortfall is what stops the shortfall reading as a finding of
    non-delivery — so a gap made entirely of them supports asking for records
    and supports nothing else.

    It is also why an absence record can never drive an escalation:
    :mod:`minutes.reconcile` has already removed excused minutes from
    ``shortfall_minutes``, so a record the family used to give minutes up
    cannot double as the reason to demand the rest.
    """
    return max(line.shortfall_minutes - line.undocumented_minutes, 0)


def is_quiet(cards: list[DecisionCard]) -> bool:
    """True when nothing needs the parent.

    Trivially ``not cards``, and it exists anyway because this is the sentence
    the product is judged on and it deserves a name that tests and callers can
    assert on out loud.

    Note what it does *not* do: it does not treat a ``ROUTINE`` card as quiet.
    A routine card still asks the parent to read something and send it, and a
    week containing one is not a week in which Minutes stayed out of the way.
    """
    return not cards


def decisions_for(
    ledger: IEPLedger,
    result: ReconciliationResult,
    deadlines: list[DeadlineStatus],
    requests: list[RecordsRequest],
    today: date,
) -> list[DecisionCard]:
    """Every decision worth waking the parent for, most urgent first.

    An empty list is the expected result and the desirable one. See
    :func:`is_quiet`.

    Cards come back ordered by urgency, then by driving deadline, then by
    ``card_id`` — the same key :mod:`minutes.statement` sorts by, so the order
    the caller sees is the order the parent will read. Ids are unique within a
    result: should two rules ever land on one id, the more urgent card wins and
    the duplicate is dropped rather than shown twice.

    A caller that shows the same parent cards week after week wants
    :func:`staged_decisions_for` instead, which carries what it needs to hold
    back a card it has already shown.
    """
    return [staged.card for staged in staged_decisions_for(
        ledger, result, deadlines, requests, today
    )]


def staged_decisions_for(
    ledger: IEPLedger,
    result: ReconciliationResult,
    deadlines: list[DeadlineStatus],
    requests: list[RecordsRequest],
    today: date,
) -> list[StagedDecision]:
    """:func:`decisions_for`, with each card's suppression key and stage beside it.

    Same rules, same order, same cards. The extra two values are what lets a
    weekly caller tell "this is the fact you saw a fortnight ago, unchanged"
    from "this is that fact, and it has since got worse" — a distinction
    urgency cannot express once a cap has flattened it. See the module
    docstring.
    """
    meeting_due = meeting_pressure(deadlines, today)
    # Refreshed once, here, so both request-driven rules read the same state
    # machine. Refreshing inside only one of them would let a single run treat
    # one request as overdue for rule 1 while rule 4's cadence still saw it as
    # open and in flight — two rules contradicting each other about one fact,
    # decided by whether the caller happened to run the state machine first.
    current = refresh_states(requests, today)

    staged = [
        *_silence_cards(ledger, current, today, meeting_due),
        *_deadline_cards(ledger, deadlines, today),
        *_shortfall_cards(ledger, result, today),
        *_records_due_cards(ledger, current, today, meeting_due),
    ]
    return _deduplicate(sorted(staged, key=_order))


# ---------------------------------------------------------------------------
# Rule 1 — the district did not answer
# ---------------------------------------------------------------------------


def _silence_cards(
    ledger: IEPLedger,
    requests: list[RecordsRequest],
    today: date,
    meeting_due: date | None,
) -> list[StagedDecision]:
    """A records request whose statutory response window has closed unanswered.

    ``requests`` arrives already refreshed by :func:`staged_decisions_for`, so a
    caller that has not run the state machine this week still gets the right
    answer. The dates are re-checked anyway: a request labelled overdue whose
    ``response_due`` has not arrived, or which is dated before its own send, is
    a contradictory record, and this module will not assert a silence that has
    not happened — the same refusal :func:`minutes.discovery.silence_events`
    makes.

    THE LADDER. One step per further response window the district stays silent.
    A district four days past its deadline and a district four months past it
    are not the same fact, and the second one should reach a parent who was
    told about the first — but "still nothing, a week later" is not news, and
    saying it weekly is how a parent learns to stop reading. Another whole
    45-day window elapsing is the smallest increment that is unambiguously a
    new state of affairs rather than the clock ticking.
    """
    cards: list[StagedDecision] = []
    for request in requests:
        if request.state is not RequestState.UNANSWERED_OVERDUE:
            continue
        due = request.response_due
        if due is None or due >= today:
            continue
        if request.sent_on is not None and due < request.sent_on:
            continue

        overdue_days = (today - due).days
        services = ", ".join(request.services) or "the services in the IEP"
        sent = request.sent_on.isoformat() if request.sent_on else "a date not recorded"

        facts = [
            f"Records request {request.request_id} covered {services} for "
            f"{request.covers_start.isoformat()} through {request.covers_end.isoformat()}.",
            f"It was sent on {sent}, with the response due {due.isoformat()}.",
            f"As of {today.isoformat()} no response is recorded as received — "
            f"{_plural(overdue_days, 'day')} past the date it was due.",
        ]
        if meeting_due is not None:
            facts.append(
                f"An IEP meeting recorded in the IEP falls on {meeting_due.isoformat()}, "
                "and 34 CFR 300.613(a) provides that records are to be produced before "
                "a meeting regarding an IEP."
            )

        card = DecisionCard(
            card_id=f"records-silence:{request.request_id}:{due.isoformat()}",
            title="The district has not sent the service records you asked for",
            why_now=(
                f"The response deadline for request {request.request_id} was "
                f"{due.isoformat()}, and that date passed "
                f"{_plural(overdue_days, 'day')} ago. That turns the request into "
                "something it was not before: a dated record that the district did "
                "not produce its service logs for this period inside the time "
                "34 CFR 300.613(a) allows. Nothing about your child's services is "
                "established by this — only that the records did not come."
            ),
            facts=facts,
            recommended_action=(
                "Ask again in writing, by a method that proves delivery, and keep "
                "the dated copy of both requests together. The unanswered first "
                "request is evidence in its own right, so it is worth naming in the "
                "second."
            ),
            urgency=(
                Urgency.DEADLINE_IMMINENT if meeting_due is not None else Urgency.TIME_SENSITIVE
            ),
            deadline=meeting_due,
            draft=_gated(compile_records_request(ledger, request, today=today)),
        )
        cards.append(
            StagedDecision(
                card=card,
                suppression_key=card.card_id,
                stage=overdue_days // RESPONSE_WINDOW_DAYS,
            )
        )
    return cards


# ---------------------------------------------------------------------------
# Rule 2 — a date the IEP set
# ---------------------------------------------------------------------------


def _deadline_cards(
    ledger: IEPLedger, deadlines: list[DeadlineStatus], today: date
) -> list[StagedDecision]:
    """Deadlines inside their lead-time window, or passed with no evidence met.

    The windows are not decided here. :func:`minutes.deadlines.evaluate_deadlines`
    has already applied the per-kind lead time that makes a status ``DUE_SOON``
    — 60 days for an annual review, 90 for a reevaluation, 14 for a progress
    report — so "approaching within its lead time" is read off the state rather
    than recomputed, and there is exactly one place those numbers live.

    ``MET`` and ``UPCOMING`` never produce a card. A met deadline is discharged,
    however late it arrived, and interrupting a parent about a satisfied
    obligation is the definition of crying wolf.

    THE LADDER, AND THE FAILURE IT EXISTS TO PREVENT. Stage ``0`` while the date
    is still ahead, ``1`` once it has passed. Without it, a caller suppressing on
    urgency alone never re-raises a date that has actually been missed: a
    progress report is capped at ``TIME_SENSITIVE`` for its whole life, so "due
    in ten days" and "that date has passed" are indistinguishable, and an annual
    review is already ``DEADLINE_IMMINENT`` a week out, so passing the date moves
    nothing there either. In both cases the card saying the date is *gone* — the
    one moment the parent most needs — would be raised once, weeks early, and
    then held back forever.
    """
    cards: list[StagedDecision] = []
    for status in deadlines:
        if status.state not in (DeadlineState.DUE_SOON, DeadlineState.OVERDUE):
            continue

        deadline = status.deadline
        days = status.days_remaining
        overdue = status.state is DeadlineState.OVERDUE

        if overdue:
            title = f"{deadline.description} — that date has passed"
            why_now = (
                f"The IEP sets this for {deadline.due.isoformat()}. That date was "
                f"{_plural(abs(days), 'day')} ago and your file holds no record that it "
                "happened. Asking in writing now both finds out and creates the dated "
                "record that you asked."
            )
            # Only once the date is gone is the absence of a record notable. Before
            # then it is simply what an unarrived date looks like, and saying it
            # like a finding would be the manufactured concern this module refuses.
            evidence_fact = "Your file records no evidence that this has been satisfied."
        else:
            title = f"{deadline.description} — due in {_plural(days, 'day')}"
            why_now = (
                f"The IEP sets this for {deadline.due.isoformat()}, which is "
                f"{_plural(days, 'day')} away. "
                f"{_LEAD_TIME_REASONS[deadline.kind]}"
            )
            evidence_fact = "That date has not arrived yet, so nothing is late."

        card = DecisionCard(
            card_id=f"deadline:{deadline_key(deadline)}",
            title=title,
            why_now=why_now,
            facts=[
                f'The IEP dated {ledger.iep_date.isoformat()} records: '
                f'"{deadline.description}", no later than {deadline.due.isoformat()}.',
                (
                    f"As of {today.isoformat()} that date is "
                    f"{_plural(abs(days), 'day')} "
                    f"{'in the past' if days < 0 else 'away'}."
                ),
                evidence_fact,
            ],
            recommended_action=(
                "Write to the case manager naming the date and asking what is "
                "scheduled. Keep it short and keep the reply."
            ),
            urgency=capped_urgency(urgency_for(status), deadline.kind),
            deadline=deadline.due,
            draft=_gated(compile_deadline_reminder(ledger, status, today=today)),
        )
        cards.append(
            StagedDecision(card=card, suppression_key=card.card_id, stage=1 if overdue else 0)
        )
    return cards


# ---------------------------------------------------------------------------
# Rule 3 — a documented service gap
# ---------------------------------------------------------------------------


def _shortfall_cards(
    ledger: IEPLedger, result: ReconciliationResult, today: date
) -> list[StagedDecision]:
    """One card for the period when a service's documented gap crosses both bars.

    ONE card, not one per service, and the reason is what a
    :class:`~minutes.models.DecisionCard` is for. A card carries a single
    ``recommended_action`` and a single draft, and the action here is "send the
    district this one letter asking it to reconcile these figures" — the same
    letter regardless of whether one service crossed or three, because
    :func:`minutes.letters.compile_shortfall_notice` reconciles the whole
    period. Three cards pointing at one letter would be three interruptions for
    one decision, which is the failure this module exists to prevent. Every
    service that crossed is named in ``facts``, and the set of them is what the
    ``card_id`` is built from, so the id always names exactly the fact the card
    states.

    THE LADDER, AND WHY THE SUPPRESSION KEY IS NOT THE ID. The id is the right
    name for the fact and the wrong key for a running conversation: built from
    the currently-crossing set, it changes when a service drops out, so a parent
    would be interrupted afresh because the news got *better*. So these cards
    suppress on the period alone, and both the ways this fact can worsen are
    carried in the stage instead — the documented gap doubling
    (:data:`SHORTFALL_ESCALATION_FACTOR`), and another service crossing the bar.
    Growth escalates; shrinkage is quietly absorbed.

    Only :func:`documented_shortfall_minutes` is weighed. See the module
    docstring for why undocumented minutes may never reach this rule.
    """
    crossing = [line for line in result.shortfalls if crosses_material_bar(line)]
    if not crossing:
        return []

    crossing.sort(key=lambda line: (-documented_shortfall_minutes(line), line.service))
    documented_total = sum(documented_shortfall_minutes(line) for line in crossing)
    names = [line.service for line in crossing]

    facts: list[str] = []
    for line in crossing:
        documented = documented_shortfall_minutes(line)
        facts.append(
            f"{line.service}: the IEP provides {line.owed_minutes} minutes for "
            f"{line.period_start.isoformat()} to {line.period_end.isoformat()}, and "
            f"{line.delivered_minutes} minutes are documented as delivered"
            + (
                f", with {line.excused_minutes} minutes excluded because a record notes "
                f"{ledger.student_alias} was absent."
                if line.excused_minutes
                else "."
            )
        )
        facts.append(
            f"{line.service}: {documented} of the {line.shortfall_minutes}-minute "
            f"difference is covered by records. The other {line.undocumented_minutes} "
            "minutes have no record either way, and this card does not rest on them."
        )

    card = DecisionCard(
        card_id=f"shortfall:{result.period_start.isoformat()}:{_service_key(names)}",
        title=f"{_service_phrase(names)} running short of what the IEP provides",
        why_now=(
            f"The documented part of the difference has reached {documented_total} "
            f"minutes across {_plural(len(crossing), 'service')}, which is past the "
            "point where Minutes treats a gap as ordinary scheduling noise. Below "
            "that bar a difference is the kind of thing one rescheduled or shortened "
            "session explains, and it is not worth your week. This one is past it, "
            "and the figures behind it come from records rather than from an absence "
            "of them."
        ),
        facts=facts,
        recommended_action=(
            "Put the arithmetic in front of the district and ask it to compare the "
            "figures against its own service logs, in writing. Ask a question rather "
            "than making a claim: the answer either corrects your records or "
            "documents the difference for you."
        ),
        urgency=Urgency.TIME_SENSITIVE,
        deadline=None,
        draft=_gated(compile_shortfall_notice(ledger, result, today=today)),
    )
    return [
        StagedDecision(
            card=card,
            suppression_key=f"shortfall:{result.period_start.isoformat()}",
            stage=_shortfall_stage(documented_total) + len(crossing) - 1,
        )
    ]


def _shortfall_stage(documented_total: int) -> int:
    """How many doublings the documented gap sits above the material floor.

    ``60`` and ``119`` are the same step; ``120`` is the next one. Counted by
    repeated multiplication rather than a logarithm so the boundary is exact
    integer arithmetic and a test can sit on it, the same discipline
    :func:`crosses_material_bar` uses for the bars themselves.
    """
    stage = 0
    bar = MATERIAL_SHORTFALL_MINUTES * SHORTFALL_ESCALATION_FACTOR
    while documented_total >= bar:
        bar *= SHORTFALL_ESCALATION_FACTOR
        stage += 1
    return stage


def crosses_material_bar(line: ServiceShortfall) -> bool:
    """Both thresholds, on the documented gap only.

    Integer arithmetic throughout, so the percentage boundary is exact and a
    test can sit on it: at ``owed = 600`` the bar is ``documented >= 60``, and
    59 does not cross it by a floating-point hair.

    Public because it is the stricter of this codebase's two gap tests (see the
    module docstring) and so the right gate for anything that would escalate to
    a district — :func:`minutes.agent.send_letter` refuses to compile a
    compensatory request unless some service crosses it.
    """
    if line.owed_minutes <= 0:
        return False
    documented = documented_shortfall_minutes(line)
    if documented < MATERIAL_SHORTFALL_MINUTES:
        return False
    return documented * 100 >= line.owed_minutes * MATERIAL_SHORTFALL_PERCENT


# ---------------------------------------------------------------------------
# Rule 4 — the cadence came round
# ---------------------------------------------------------------------------


def _records_due_cards(
    ledger: IEPLedger,
    requests: list[RecordsRequest],
    today: date,
    meeting_due: date | None,
) -> list[StagedDecision]:
    """A records request is due to go out.

    :func:`minutes.discovery.due_requests` decides this entirely — including
    the suppression that matters most here, that nothing is issued while a
    draft or sent request is still open. This rule adds only the framing and
    the urgency.

    An overdue, unanswered request does *not* stop the cadence, which is
    discovery's own deliberate choice: a district that simply never answers
    would otherwise stall every future request behind the one it ignored. So
    this rule and rule 1 can both fire in one week, and they should — they name
    different windows and ask for different things.

    NO LADDER, DELIBERATELY. Every other rule reports a fact that can get worse
    while the parent does nothing. This one reports that it is time to ask, and
    a request that is due is simply due; the card disappears the moment it goes
    out, and a parent who read it and chose not to send has answered. Inventing
    an escalation here would be pestering them for a decision they already made.
    """
    cards: list[StagedDecision] = []
    first_ever = not requests
    for request in due_requests(ledger, requests, today, cadence_days=REQUEST_CADENCE_DAYS):
        window_days = (request.covers_end - request.covers_start).days + 1
        services = ", ".join(request.services) or "the services in the IEP"
        accrued = (
            f"{_plural(window_days, 'day')} of service have accrued since "
            f"{request.covers_start.isoformat()} and none of it has been asked about yet"
            if first_ever
            else (
                f"{_plural(window_days, 'day')} of service have accrued since the last "
                "request's window ended"
            )
        )

        facts = [
            f"The request would cover {request.covers_start.isoformat()} through "
            f"{request.covers_end.isoformat()} — {_plural(window_days, 'day')} not yet "
            "covered by any request.",
            f"Services asked about: {services}.",
            "The district's own service logs are the strongest record of what was "
            "delivered, and they are the one record it controls.",
        ]
        if meeting_due is not None:
            facts.append(
                f"An IEP meeting recorded in the IEP falls on {meeting_due.isoformat()}, "
                "and 34 CFR 300.613(a) provides that records are to be produced before "
                "a meeting regarding an IEP."
            )

        card = DecisionCard(
            card_id=f"records-due:{request.covers_start.isoformat()}",
            title="Time to ask the district for its service records",
            why_now=(
                f"{accrued}, which is the cadence Minutes asks on. "
                "Asking on a schedule is what produces evidence rather than waiting "
                "for it: either the logs come back and your ledger gains the "
                "district's own confirmation, or they do not, and the district's "
                "silence past the response date becomes a dated fact you could not "
                "have created by watching."
            ),
            facts=facts,
            recommended_action=(
                "Review the request and send it by a method that proves delivery — "
                "the response clock runs from receipt, so the delivery proof is what "
                "makes the deadline enforceable."
            ),
            urgency=(Urgency.TIME_SENSITIVE if meeting_due is not None else Urgency.ROUTINE),
            deadline=meeting_due,
            draft=_gated(compile_records_request(ledger, request, today=today)),
        )
        cards.append(StagedDecision(card=card, suppression_key=card.card_id, stage=0))
    return cards


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------


def meeting_pressure(deadlines: list[DeadlineStatus], today: date) -> date | None:
    """The soonest IEP meeting close enough to make outstanding records urgent.

    ``None`` unless an unmet annual review or reevaluation falls within
    :data:`RECORDS_BEFORE_MEETING_DAYS`. A date already past is excluded: a
    meeting that has happened cannot create pressure to get records before it,
    and treating it as if it could would keep a card permanently imminent on a
    deadline nobody can act on any more.

    Public because the cards are not the only place this date belongs: it is
    also the ``before_meeting_on`` that shortens a real response clock in
    :func:`minutes.discovery.mark_sent`, and a card citing the meeting while the
    clock ignored it would state a deadline the file does not hold.
    """
    candidates = [
        status.deadline.due
        for status in deadlines
        if status.deadline.kind in MEETING_DEADLINE_KINDS
        and status.state is not DeadlineState.MET
        and 0 <= (status.deadline.due - today).days <= RECORDS_BEFORE_MEETING_DAYS
    ]
    return min(candidates) if candidates else None


def _gated(letter: Letter) -> Letter | None:
    """The compiled letter, or ``None`` if it fails the integrity gate.

    :func:`minutes.letters.validate_letter` is the gate a letter must pass
    before a parent is shown it, and it is applied here rather than assumed:
    a card is the one surface that puts a draft in front of somebody. A failing
    draft is dropped and its card still ships, because the fact that raised the
    card is true whether or not a letter could be assembled for it.

    ``for_sending`` is deliberately not set. A compiled records request is sound
    but carries a blank only the parent can fill; refusing it here would deny
    them the draft precisely because it correctly declined to invent a fact
    about their circumstances.
    """
    return letter if not validate_letter(letter) else None


def capped_urgency(urgency: Urgency, kind: DeadlineKind) -> Urgency:
    """Apply :data:`URGENCY_CAP_BY_KIND`. Only ever lowers urgency.

    Public because a deadline's urgency is reported in two places a parent may
    read in the same breath — a decision card and
    :func:`minutes.tools.check_deadlines` — and two numbers on two different
    scales for one date is worse than either scale alone.
    """
    cap = URGENCY_CAP_BY_KIND.get(kind)
    if cap is None:
        return urgency
    return cap if _URGENCY_RANK[urgency] < _URGENCY_RANK[cap] else urgency


def _order(staged: StagedDecision) -> tuple[int, date, str]:
    """Most urgent first, then soonest deadline, then id for a stable tie-break."""
    card = staged.card
    return (_URGENCY_RANK[card.urgency], card.deadline or date.max, card.card_id)


def _deduplicate(staged: list[StagedDecision]) -> list[StagedDecision]:
    """Keep the first card for each ``card_id``.

    Input is already ordered, so "first" means the most urgent rendering of the
    fact. Two rules landing on one id would be a bug in the id scheme rather
    than a thing to show the parent twice; this makes the invariant hold at the
    boundary instead of trusting it.
    """
    seen: set[str] = set()
    unique: list[StagedDecision] = []
    for item in staged:
        if item.card.card_id in seen:
            continue
        seen.add(item.card.card_id)
        unique.append(item)
    return unique


def _service_key(names: list[str]) -> str:
    """The canonical, sorted service component of a shortfall ``card_id``.

    Folded through :func:`minutes.reconcile.canonical_service` so that a
    district export respelling "Occupational Therapy (OT)" does not mint a
    second card for a service the parent already dealt with, and sorted so the
    id does not depend on reconciliation's ordering.
    """
    return "+".join(sorted({canonical_service(name) or name for name in names}))


def _service_phrase(names: list[str]) -> str:
    """A title fragment that reads naturally for one service or several."""
    if len(names) == 1:
        return f"{names[0]} is"
    if len(names) == 2:
        return f"{names[0]} and {names[1]} are"
    return f"{len(names)} services are"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"
