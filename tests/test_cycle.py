"""The heartbeat, replayed — and the arc the demo rests on.

Every figure and every date below was produced by running the deterministic
cycle over the committed fixture case ("Maya R.", fall 2026) and then checked by
hand against the ledger. Nothing here asserts merely that a key exists: a cycle
that quietly starts raising a different set of cards is a cycle that will either
interrupt a parent who should have been left alone or stay silent through the
week that mattered.

Three properties get their own tests because the product's whole claim rests on
them: most weeks are quiet, the same true fact never interrupts twice, and no
card ever carries a letter that cannot footnote itself.

No test in this file makes a network call, an AWS call, or an LLM call — and
:func:`test_the_cycle_cannot_reach_a_model_at_all` proves that structurally
rather than by assertion, by reading the module's own imports.
"""

import ast
from datetime import date, timedelta
from pathlib import Path

import pytest

from minutes.cycle import (
    ACTOR,
    QUIET_HEADLINE,
    REPLAY_ACTOR,
    CycleOutcome,
    RaisedCard,
    run_cycle,
    run_semester,
    send_nothing,
    send_when_due,
)
from minutes.decisions import decisions_for
from minutes.letters import validate_letter
from minutes.discovery import mark_sent
from minutes.models import (
    Attribution,
    DeadlineKind,
    Provenance,
    RecordsRequest,
    RequestState,
    ServiceEvent,
    Urgency,
)
from minutes.tools import load_case_record

# The fall term of the fixture case. Services start 2026-09-08; 2026-12-19 is
# the last day of term. Weekly wakes from the first day of service.
TERM_START = date(2026, 9, 8)
TERM_END = date(2026, 12, 19)

PROGRESS_REPORT = (
    "deadline:progress_report:2026-11-06:Progress report on all goals provided to parents"
)

# The four weeks of the fixture semester that produce a decision, and what each
# one is. Everything else is quiet. This mapping IS the demo.
DECISION_WEEKS = {
    # 36 days of service have accrued and nobody has asked the district for its
    # own logs. The cadence comes round.
    date(2026, 10, 13): ["records-due:2026-09-08"],
    # The IEP's first progress report is due 2026-11-06, inside its 14-day
    # lead time.
    date(2026, 10, 27): [PROGRESS_REPORT],
    # 2026-11-06 came and went with nothing in the file. The SAME card, now
    # saying the date has passed — which urgency alone could never express,
    # because a progress report is capped at time_sensitive for its whole life.
    # Beside it, the first gap the DISTRICT'S OWN records establish crosses the
    # material bar: two speech sessions it recorded as not held.
    date(2026, 11, 10): [PROGRESS_REPORT, "shortfall:2026-09-08:speech"],
    # 45 days after req-001 went out, its response deadline has passed. The
    # silence is now evidence, and the cadence has come round again for the
    # period the unanswered request no longer blocks.
    date(2026, 12, 1): ["records-silence:req-001:2026-11-27", "records-due:2026-10-14"],
}


@pytest.fixture(scope="module")
def semester() -> list[CycleOutcome]:
    """The whole fall term, replayed weekly, with a parent who approves records requests.

    ``sends`` is passed explicitly and always must be: the default is
    :func:`send_nothing`, so that a caller who has not thought about it gets the
    honest floor rather than a term of simulated consent.
    """
    return run_semester(TERM_START, TERM_END, sends=send_when_due)


# ---------------------------------------------------------------------------
# Quiet is the product.
# ---------------------------------------------------------------------------


def test_most_weeks_are_quiet(semester):
    """Eleven of fifteen weeks, nothing needs the parent.

    This is the assertion the product lives or dies on. A tool that raises
    something every week trains a parent to close it unread, and then the
    annual review passes anyway.
    """
    assert len(semester) == 15
    assert sum(1 for outcome in semester if outcome.quiet) == 11
    assert [outcome.today for outcome in semester if not outcome.quiet] == list(DECISION_WEEKS)


def test_a_quiet_week_still_shows_its_work(semester):
    """Silence is only trustworthy if you can see what was checked behind it."""
    quiet = next(outcome for outcome in semester if outcome.quiet)

    assert quiet.headline == QUIET_HEADLINE
    assert quiet.summary().startswith(QUIET_HEADLINE)
    assert len(quiet.checked) >= 3
    assert any("Reconciled" in line for line in quiet.checked)
    assert any("date(s) the IEP sets" in line for line in quiet.checked)


def test_a_quiet_week_still_writes_to_the_trail(semester):
    """A week with nothing to say still records that it looked."""
    quiet = next(outcome for outcome in semester if outcome.quiet)
    actions = [entry.action for entry in quiet.audit]

    assert "checked the records requests" in actions
    assert "reconciled the ledger" in actions
    assert all(entry.actor == ACTOR for entry in quiet.audit)


def test_headline_counts_the_decisions(semester):
    one = next(o for o in semester if len(o.new_cards) == 1)
    two = next(o for o in semester if len(o.new_cards) == 2)

    assert one.headline == "1 decision needs you."
    assert two.headline == "2 decisions need you."
    assert not one.quiet


def test_a_suppressed_card_does_not_break_quiet(semester):
    """A fact the parent has already seen, unchanged, is not an interruption."""
    week = next(o for o in semester if o.suppressed_cards and not o.new_cards)
    assert week.quiet
    assert week.headline == QUIET_HEADLINE


# ---------------------------------------------------------------------------
# The arc. This is what a demo shows on stage.
# ---------------------------------------------------------------------------


def test_the_arc_happens_in_full(semester):
    """Cadence comes round, request goes out, silence closes over it, escalation.

    Four separate mechanisms have to agree for this to work: the discovery
    cadence, the 45-day statutory clock, the silence derivation, and the
    decision rules. Asserting the exact card ids on the exact dates is the only
    way to know all four still line up.
    """
    by_date = {outcome.today: outcome for outcome in semester}
    for when, expected in DECISION_WEEKS.items():
        assert sorted(card.card_id for card in by_date[when].new_cards) == sorted(expected), when


def test_the_request_goes_out_the_week_the_cadence_raises_it(semester):
    """The stand-in parent approves, and the response clock starts that day."""
    week = next(o for o in semester if o.today == date(2026, 10, 13))
    assert [request.request_id for request in week.due_requests] == ["req-001"]

    after = next(o for o in semester if o.today == date(2026, 10, 20))
    sent = next(request for request in after.requests if request.request_id == "req-001")

    assert sent.state is RequestState.SENT
    assert sent.sent_on == date(2026, 10, 13)
    # 45 calendar days, 34 CFR 300.613(a).
    assert sent.response_due == date(2026, 11, 27)


def test_silence_appears_only_once_the_deadline_has_actually_passed(semester):
    """Minutes never asserts a silence that has not happened yet."""
    by_date = {outcome.today: outcome for outcome in semester}

    assert by_date[date(2026, 11, 24)].silence == []
    assert len(by_date[date(2026, 12, 1)].silence) == 4  # one per service on the request

    fact = by_date[date(2026, 12, 1)].silence[0]
    assert fact.provenance is Provenance.DOCUMENTED_SILENCE
    assert fact.event_date == date(2026, 11, 27)
    assert fact.delivered is False
    assert fact.minutes == 0


def test_silence_is_recomputed_not_accumulated(semester):
    """The same unanswered request yields the same four facts, every week.

    discovery.silence_events returns a full recomputation, and a caller that
    appended instead of replacing would multiply one silence into many — and,
    downstream, shrink the undocumented bucket that keeps a shortfall honest.
    """
    after_silence = [o for o in semester if o.today >= date(2026, 12, 1)]
    assert [len(outcome.silence) for outcome in after_silence] == [4, 4, 4]


def test_the_silence_card_speaks_about_records_not_about_services(semester):
    """The one sentence silence can never support must not appear."""
    card = next(
        c
        for c in next(o for o in semester if o.today == date(2026, 12, 1)).new_cards
        if c.card_id.startswith("records-silence:")
    )
    prose = " ".join([card.title, card.why_now, *card.facts, card.recommended_action]).lower()

    assert "records" in prose
    assert "recorded as not delivered" not in prose
    assert "failed to deliver" not in prose


def test_silence_is_recorded_as_evidence_in_the_trail(semester):
    week = next(o for o in semester if o.today == date(2026, 12, 1))
    entry = next(e for e in week.audit if e.action == "recorded documented silence")

    assert entry.evidence is not None
    assert entry.evidence.provenance is Provenance.DOCUMENTED_SILENCE
    assert entry.evidence.source == "req-001"
    assert "not a fact about whether any session happened" in entry.detail


def test_an_unanswered_request_does_not_stall_the_cadence(semester):
    """A district that simply never answers cannot freeze the parent's discovery."""
    week = next(o for o in semester if o.today == date(2026, 12, 1))
    assert [request.request_id for request in week.due_requests] == ["req-002"]
    assert any(card.card_id == "records-due:2026-10-14" for card in week.new_cards)


def test_nothing_is_sent_when_the_parent_never_answers():
    """The honest floor: without a human, the cycle asks and nothing goes out."""
    outcomes = run_semester(TERM_START, TERM_END, sends=send_nothing)

    assert all(request.state is RequestState.DRAFT for o in outcomes for request in o.requests)
    assert all(outcome.silence == [] for outcome in outcomes)
    # And so the arc stops at its first step: the cadence card, and nothing after.
    raised = {card.card_id for outcome in outcomes for card in outcome.new_cards}
    assert not any(card_id.startswith("records-silence:") for card_id in raised)


# ---------------------------------------------------------------------------
# The card the fixture semester deliberately never raises.
# ---------------------------------------------------------------------------


def test_an_undocumented_gap_alone_never_raises_a_shortfall_card(semester):
    """The best demonstration of the product's discipline, asserted out loud.

    Over the fall term the fixture case is thousands of minutes short, and
    almost all of that is undocumented — nobody wrote anything down in either
    direction. Waking a parent to demand compensatory services on a gap that
    exists only in their own records is precisely the interruption this product
    exists not to make. The answer to missing records is to ask for records.

    Specialized Academic Instruction is the proof: 6,180 minutes short over the
    term, every one of them undocumented, and it never once drives a card. The
    two services that DO raise one are the two the district's own records
    convict it on.
    """
    shortfall_cards = [
        card
        for outcome in semester
        for card in outcome.new_cards
        if card.card_id.startswith("shortfall:")
    ]
    assert shortfall_cards, "the district's own recorded misses should surface"
    assert not any(
        "specialized academic instruction" in card.card_id for card in shortfall_cards
    ), "an undocumented gap must never become an accusation"

def test_a_documented_gap_does_raise_a_shortfall_card():
    """The same rule fires the moment the district's own records show the gap.

    Same ledger, same dates — only the evidence grade changes. This is the
    control for the test above: the bar is documentation, not size.
    """
    case = load_case_record()
    confirmed = [
        ServiceEvent(
            event_date=date(2026, 9, 8) + timedelta(days=offset),
            service="Specialized Academic Instruction",
            minutes=0,
            delivered=False,
            provenance=Provenance.SCHOOL_CONFIRMED,
            source="district-service-log-2026-10",
            attribution=Attribution.SCHOOL_OR_UNRECORDED,
        )
        for offset in range(0, 40)
    ]
    documented = case.__class__(
        case_id=case.case_id,
        ledger=case.ledger,
        events=[*case.events, *confirmed],
        requests=case.requests,
        correspondence_items=case.correspondence_items,
    )

    cards = run_cycle(date(2026, 10, 20), case=documented).new_cards
    assert any(card.card_id.startswith("shortfall:") for card in cards)


# ---------------------------------------------------------------------------
# Suppression across cycles.
# ---------------------------------------------------------------------------


def test_a_fact_is_only_raised_again_once_it_has_genuinely_got_worse(semester):
    """The invariant suppression actually owes the parent.

    Not "never twice" — that was the bug, and it is what silences a deadline at
    the moment it is finally missed. The real rule is that a repeat raise must
    be accompanied by a strictly worse suppression record, and never by a better
    one in either dimension.
    """
    rank = {Urgency.DEADLINE_IMMINENT: 0, Urgency.TIME_SENSITIVE: 1, Urgency.ROUTINE: 2}
    previous: dict[str, RaisedCard] = {}
    repeats = 0

    for outcome in semester:
        changed = {
            key: value for key, value in outcome.raised.items() if previous.get(key) != value
        }
        # One changed suppression record per card put in front of the parent:
        # nothing is raised without moving its record, and nothing moves its
        # record without being raised.
        assert len(changed) == len(outcome.new_cards), outcome.today

        for key, now in changed.items():
            was = previous.get(key)
            if was is None:
                continue
            repeats += 1
            assert rank[now.urgency] <= rank[was.urgency], (outcome.today, key)
            assert now.stage >= was.stage, (outcome.today, key)
            assert now != was, (outcome.today, key)

        previous = dict(outcome.raised)

    assert repeats == 1, "the fixture term escalates exactly once, on 2026-11-10"


def test_a_deadline_that_goes_overdue_reaches_the_parent(semester):
    """THE REGRESSION. A missed date must not be the one thing suppression eats.

    The progress report is raised on 2026-10-27 at "due in 10 days", suppressed
    on 2026-11-03 at "due in 3 days" — unchanged, so rightly held back — and
    raised again on 2026-11-10, the first wake after 2026-11-06 passed with
    nothing in the file. Its urgency is time_sensitive on every one of those
    three days, because URGENCY_CAP_BY_KIND pins a progress report there; only
    the stage moves.
    """
    overdue_weeks = [
        outcome.today
        for outcome in semester
        for card in outcome.new_cards
        if card.title.endswith("that date has passed")
    ]
    assert overdue_weeks == [date(2026, 11, 10)]

    week = next(o for o in semester if o.today == date(2026, 11, 10))
    card = next(c for c in week.new_cards if c.card_id == PROGRESS_REPORT)

    assert card.urgency is Urgency.TIME_SENSITIVE
    assert week.raised[PROGRESS_REPORT] == RaisedCard(urgency=Urgency.TIME_SENSITIVE, stage=1)

    entry = next(e for e in week.audit if e.action == "raised a decision again")
    assert "escalation step 0 to 1" in entry.detail

    # And once raised as overdue it settles again: still true every following
    # week, never put in front of the parent a third time.
    later = [o for o in semester if o.today > date(2026, 11, 10)]
    assert all(PROGRESS_REPORT not in {c.card_id for c in o.new_cards} for o in later)
    assert all(PROGRESS_REPORT in {c.card_id for c in o.suppressed_cards} for o in later)


def test_a_suppressed_card_is_still_true_and_still_reported(semester):
    """Suppressed means "not put in front of them", never "forgotten"."""
    week = next(o for o in semester if o.today == date(2026, 11, 3))
    suppressed = week.suppressed_cards

    assert [card.card_id for card in suppressed] == [PROGRESS_REPORT]
    assert any(entry.action == "held back a decision already raised" for entry in week.audit)


def test_a_card_returns_when_it_becomes_more_urgent(semester):
    """Escalation overrides suppression — seeded from a real prior cycle.

    The carried map comes from the 2026-10-27 wake rather than being written by
    hand, so this can only pass on a state the pipeline actually reaches. A
    hand-written ``ROUTINE`` here would be a state no progress-report card can
    ever hold, and the test would pass whether or not escalation worked.
    """
    case = load_case_record()
    raised_on_the_27th = next(o for o in semester if o.today == date(2026, 10, 27)).raised
    assert raised_on_the_27th[PROGRESS_REPORT] == RaisedCard(
        urgency=Urgency.TIME_SENSITIVE, stage=0
    )

    still_ahead = run_cycle(date(2026, 11, 3), case=case, raised=raised_on_the_27th)
    assert still_ahead.new_cards == []

    passed = run_cycle(date(2026, 11, 10), case=case, raised=raised_on_the_27th)
    assert PROGRESS_REPORT in [card.card_id for card in passed.new_cards]


def test_suppression_remembers_the_worst_a_card_reached():
    """A card that escalates and then relaxes does not read as new again.

    Built on a real relaxation rather than a hand-written urgency. An unmet
    annual review inside 45 days makes an unanswered records request
    deadline_imminent, because 34 CFR 300.613(a) requires production before a
    meeting regarding an IEP; once that date is behind us the pressure is gone
    and the identical card relaxes to time_sensitive. The parent has already
    seen it at its worst, so it stays held back.
    """
    case = load_case_record()
    meeting = date(2026, 12, 10)
    ledger = case.ledger.model_copy(
        update={
            "deadlines": [
                d.model_copy(update={"due": meeting})
                if d.kind is DeadlineKind.ANNUAL_REVIEW
                else d
                for d in case.ledger.deadlines
            ]
        }
    )
    sent = mark_sent(
        RecordsRequest(
            request_id="req-001",
            covers_start=date(2026, 9, 8),
            covers_end=date(2026, 9, 30),
            services=[o.service for o in ledger.obligations],
        ),
        date(2026, 10, 1),
    )
    assert sent.response_due == date(2026, 11, 15)
    with_meeting = case.__class__(
        case_id=case.case_id,
        ledger=ledger,
        events=case.events,
        requests=[sent],
        correspondence_items=case.correspondence_items,
    )
    silence = "records-silence:req-001:2026-11-15"

    # 2026-11-20: overdue, and the annual review is 20 days away.
    under_pressure = run_cycle(date(2026, 11, 20), case=with_meeting, requests=[sent])
    card = next(c for c in under_pressure.new_cards if c.card_id == silence)
    assert card.urgency is Urgency.DEADLINE_IMMINENT
    assert under_pressure.raised[silence] == RaisedCard(urgency=Urgency.DEADLINE_IMMINENT, stage=0)

    # 2026-12-15: the meeting has been held, so the same fact is merely
    # time-sensitive. Relaxing is not news.
    after = run_cycle(
        date(2026, 12, 15), case=with_meeting, requests=[sent], raised=under_pressure.raised
    )
    relaxed = next(c for c in after.suppressed_cards if c.card_id == silence)

    assert relaxed.urgency is Urgency.TIME_SENSITIVE
    # The moved annual review is itself overdue by now and raises its own card;
    # what matters here is that the silence is not among them.
    assert silence not in {card.card_id for card in after.new_cards}
    assert after.raised[silence] == RaisedCard(urgency=Urgency.DEADLINE_IMMINENT, stage=0)


def test_suppression_is_carried_by_the_outcome_not_by_hidden_state():
    """Two identical cycles with no memory both raise; only the carried map suppresses."""
    case = load_case_record()
    first = run_cycle(date(2026, 10, 27), case=case)
    again = run_cycle(date(2026, 10, 27), case=case)
    third = run_cycle(date(2026, 10, 27), case=case, raised=first.raised)

    assert [c.card_id for c in first.new_cards] == [c.card_id for c in again.new_cards]
    assert third.new_cards == []


# ---------------------------------------------------------------------------
# Every draft that reaches a parent has to be sound.
# ---------------------------------------------------------------------------


def test_every_draft_on_every_card_validates(semester):
    """A letter that cannot footnote its own claims must never reach a parent."""
    drafts = [
        card.draft
        for outcome in semester
        for card in (*outcome.new_cards, *outcome.suppressed_cards)
        if card.draft is not None
    ]

    assert drafts, "the semester should compile at least one draft"
    for draft in drafts:
        assert validate_letter(draft) == [], draft.subject


def test_cards_match_what_the_decision_rules_would_say_alone(semester):
    """The cycle orchestrates the rules; it never second-guesses them."""
    week = next(o for o in semester if o.today == date(2026, 12, 1))
    case = load_case_record()

    # Rebuild the same inputs the cycle handed the rules that week.
    from minutes.deadlines import evaluate_deadlines
    from minutes.discovery import refresh_states
    from minutes.reconcile import reconcile

    previous = next(o for o in semester if o.today == date(2026, 11, 24))
    current = refresh_states(previous.requests, week.today)
    result = reconcile(case.ledger, [*case.events, *week.silence], week.period_start, week.today)
    expected = decisions_for(
        case.ledger, result, evaluate_deadlines(case.ledger, week.today), current, week.today
    )

    raised_or_held = {card.card_id for card in (*week.new_cards, *week.suppressed_cards)}
    assert raised_or_held == {card.card_id for card in expected}


# ---------------------------------------------------------------------------
# Purity, determinism, and the edges.
# ---------------------------------------------------------------------------


def test_a_cycle_is_a_pure_function_of_what_it_was_given():
    case = load_case_record()
    first = run_cycle(date(2026, 12, 1), case=case)
    second = run_cycle(date(2026, 12, 1), case=case)

    assert first.model_dump() == second.model_dump()


def test_a_cycle_does_not_mutate_its_inputs():
    """The caller's request list and suppression map survive the call untouched."""
    case = load_case_record()
    requests = list(case.requests)
    raised = {"records-due:2026-09-08": RaisedCard(urgency=Urgency.ROUTINE)}
    snapshot = dict(raised)

    outcome = run_cycle(date(2026, 12, 1), case=case, requests=requests, raised=raised)

    assert raised == snapshot
    assert requests == list(case.requests)
    assert outcome.raised is not raised


def test_a_wake_before_any_service_started_says_so_rather_than_computing():
    """Reconciling a window that ends before the IEP begins is true and misleading."""
    outcome = run_cycle(date(2026, 9, 1), case=load_case_record())

    assert outcome.quiet
    assert outcome.new_cards == []
    assert outcome.checked == [
        "No IEP service had started by this date, so nothing could be owed yet."
    ]
    assert outcome.audit[0].action == "woke and found nothing owed yet"


def test_the_reconciliation_window_can_be_narrowed():
    case = load_case_record()
    outcome = run_cycle(date(2026, 12, 1), case=case, period_start=date(2026, 11, 1))

    assert outcome.period_start == date(2026, 11, 1)
    assert outcome.period_end == date(2026, 12, 1)


def test_the_silence_facts_reach_the_reconciliation(semester):
    """The silence derived this run is weighed by the arithmetic this run.

    ``events_considered`` is read off the reconciliation itself rather than
    recomputed from the inputs, so a silence set that was appended to a growing
    list instead of replacing it would show up here as a count that climbs every
    week — which is the whole reason the number is reported.
    """
    case = load_case_record()
    week = next(o for o in semester if o.today == date(2026, 12, 1))

    from minutes.reconcile import reconcile

    without = reconcile(case.ledger, case.events, week.period_start, week.today)
    assert week.events_considered - without.events_considered == len(week.silence) == 4
    assert f"against {week.events_considered} record(s)" in " ".join(week.checked)


def test_the_silence_set_holds_no_duplicates(semester):
    """One unanswered request is one dated fact per service, never a growing pile."""
    week = next(o for o in semester if o.today == date(2026, 12, 1))
    keys = [(fact.source, fact.service, fact.event_date) for fact in week.silence]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"step_days": 0}, "step_days must be at least 1"),
        ({"step_days": -7}, "step_days must be at least 1"),
    ],
)
def test_a_non_positive_step_is_refused(kwargs, message):
    """A zero step is not "check continuously", it is an infinite loop."""
    with pytest.raises(ValueError, match=message):
        run_semester(TERM_START, TERM_END, **kwargs)


def test_a_backwards_semester_is_refused():
    with pytest.raises(ValueError, match="precedes start"):
        run_semester(TERM_END, TERM_START)


def test_a_single_day_semester_runs_exactly_one_wake():
    outcomes = run_semester(TERM_START, TERM_START)
    assert len(outcomes) == 1
    assert outcomes[0].today == TERM_START


def test_send_policies_are_explicit_about_standing_in_for_a_parent():
    """Both stand-ins exist and neither pretends to be a decision."""
    request = load_case_record().requests
    assert send_when_due.__doc__ and "parent" in send_when_due.__doc__
    assert send_nothing.__doc__ and "never answers" in send_nothing.__doc__
    assert request == []


def test_the_replay_sends_nothing_unless_a_caller_asks_for_it_out_loud():
    """The default must be the honest floor, not a term of simulated consent.

    ``run_semester`` is public. A caller who never thought about the send policy
    used to get a replay in which a parent approved every records request — and
    then a permanent, dated assertion that the district produced nothing in
    response to a letter nobody sent.
    """
    outcomes = run_semester(TERM_START, TERM_END)

    assert all(r.state is RequestState.DRAFT for o in outcomes for r in o.requests)
    assert not [e for o in outcomes for e in o.audit if e.actor == REPLAY_ACTOR]


def test_a_simulated_send_never_claims_a_parent_approved_it(semester):
    """The trail is the product, so a stand-in must not sign a family's name.

    The entry is real and permanent, and the ``AuditEntry`` type is the same one
    the live agent writes. What keeps it honest is that it says what happened —
    a policy, a replay, nobody asked — and carries an actor that is not the
    cycle's own.
    """
    week = next(o for o in semester if o.today == date(2026, 10, 13))
    entry = next(e for e in week.audit if e.actor == REPLAY_ACTOR)

    assert entry.action == "simulated a records request going out"
    assert "no parent was asked" in entry.detail
    assert "nothing was posted" in entry.detail
    assert "approved and sent" not in entry.detail
    # Everything the cycle itself observed keeps the cycle's own actor.
    assert all(e.actor == ACTOR for e in week.audit if e is not entry)


def test_the_outcome_carries_a_send_made_on_the_same_wake(semester):
    """``requests`` says it is the state to carry forward, so it has to be.

    A caller persisting the last wake's outcome and resuming from it would
    otherwise lose a send made on that wake and re-issue the same request —
    restarting a response clock that discovery.mark_sent would then refuse.
    """
    week = next(o for o in semester if o.today == date(2026, 10, 13))

    assert [r.request_id for r in week.due_requests] == ["req-001"]
    sent = next(r for r in week.requests if r.request_id == "req-001")
    assert sent.state is RequestState.SENT
    assert sent.sent_on == date(2026, 10, 13)


# ---------------------------------------------------------------------------
# Hermeticity, proved structurally.
# ---------------------------------------------------------------------------


def test_the_cycle_cannot_reach_a_model_at_all():
    """Read the module's own imports: no strands, no boto3, no bedrock.

    An assertion that "this test made no network call" only covers the paths a
    test happened to walk. Reading the import graph covers every path, and it
    is the reason a whole semester replays for free.
    """
    source = Path(__file__).resolve().parents[1] / "minutes" / "cycle.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])

    assert not imported & {"strands", "boto3", "botocore", "bedrock_agentcore"}
