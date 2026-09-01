"""Decision tests — when Minutes speaks, and (mostly) when it stays quiet.

The most important test in this file is the one asserting that a compliant
month produces NO cards at all. Everything else guards a boundary around it.

Every fixture here is synthetic. "Maya P." and "Ridgeview" are invented; no
real child, school, or district is described. No test in this file makes a
network call or an LLM call — ``decisions.py`` is deterministic rules over the
engine's output, which is exactly why it can be tested this way.
"""

from datetime import date, timedelta

import pytest

from minutes.deadlines import IMMINENT_DAYS, deadline_key, evaluate_deadlines, urgency_for
from minutes.decisions import (
    MATERIAL_SHORTFALL_MINUTES,
    MATERIAL_SHORTFALL_PERCENT,
    RECORDS_BEFORE_MEETING_DAYS,
    REQUEST_CADENCE_DAYS,
    SHORTFALL_ESCALATION_FACTOR,
    decisions_for,
    documented_shortfall_minutes,
    is_quiet,
    staged_decisions_for,
)
from minutes.discovery import RESPONSE_WINDOW_DAYS, mark_answered, mark_sent
from minutes.letters import validate_letter
from minutes.models import (
    Deadline,
    DeadlineKind,
    EvidenceRef,
    IEPLedger,
    Period,
    Provenance,
    ReconciliationResult,
    RecordsRequest,
    ServiceObligation,
    ServiceShortfall,
    Urgency,
)
from minutes.statement import WITHHELD_CARD_NOTE, build_statement, render_markdown

SPEECH = "Speech-Language Therapy"
OT = "Occupational Therapy"

SERVICES_BEGIN = date(2026, 9, 8)

# Rank used only to assert ordering; mirrors minutes.statement's own.
_RANK = {Urgency.DEADLINE_IMMINENT: 0, Urgency.TIME_SENSITIVE: 1, Urgency.ROUTINE: 2}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _obligation(service: str = SPEECH, **overrides) -> ServiceObligation:
    base = dict(
        service=service,
        minutes_per_session=30,
        sessions_per_period=2,
        period=Period.WEEK,
        provider_role="Licensed Speech-Language Pathologist",
        setting="therapy room",
        start_date=SERVICES_BEGIN,
        end_date=date(2027, 6, 11),
        source_quote="30 minutes per session, 2 sessions per week",
    )
    base.update(overrides)
    return ServiceObligation(**base)


def _deadline(kind: DeadlineKind, due: date, description: str | None = None) -> Deadline:
    label = description or {
        DeadlineKind.ANNUAL_REVIEW: "Annual review meeting",
        DeadlineKind.REEVALUATION: "Three-year reevaluation",
        DeadlineKind.PROGRESS_REPORT: "Quarterly progress report",
        DeadlineKind.OTHER: "Assistive technology consultation",
    }[kind]
    return Deadline(
        kind=kind,
        due=due,
        description=label,
        source_quote=f"{label} no later than {due.isoformat()}.",
    )


def _ledger(*, deadlines: list[Deadline] | None = None, obligations=None) -> IEPLedger:
    return IEPLedger(
        student_alias="Maya P.",
        school_year="2026-2027",
        iep_date=date(2026, 9, 1),
        obligations=list(obligations) if obligations is not None else [_obligation()],
        deadlines=list(deadlines or []),
        accommodations=[],
    )


def _shortfall(
    service: str = SPEECH,
    *,
    owed: int = 600,
    delivered: int = 600,
    excused: int = 0,
    undocumented: int = 0,
    school_confirmed: int | None = None,
    parent_observed: int = 0,
    evidence: list[EvidenceRef] | None = None,
) -> ServiceShortfall:
    """A reconciliation line. ``shortfall`` is derived the way reconcile derives it."""
    shortfall = max(owed - delivered - excused, 0)
    confirmed = delivered - parent_observed if school_confirmed is None else school_confirmed
    return ServiceShortfall(
        service=service,
        period_start=SERVICES_BEGIN,
        period_end=date(2026, 10, 7),
        owed_minutes=owed,
        delivered_minutes=delivered,
        excused_minutes=excused,
        shortfall_minutes=shortfall,
        school_confirmed_minutes=confirmed,
        parent_observed_minutes=parent_observed,
        undocumented_minutes=undocumented,
        evidence=list(evidence or []),
    )


def _result(*lines: ServiceShortfall) -> ReconciliationResult:
    return ReconciliationResult(
        period_start=SERVICES_BEGIN,
        period_end=date(2026, 10, 7),
        shortfalls=list(lines) or [_shortfall()],
        events_considered=20,
    )


def _in_flight(today: date) -> RecordsRequest:
    """A request sent today: still inside its response window.

    Used to isolate the deadline rule. A SENT request holds the discovery
    cadence closed (``due_requests`` never asks twice for one period) and is
    not yet overdue, so neither request rule can fire and whatever cards come
    back came from the clocks.
    """
    return mark_sent(
        RecordsRequest(
            request_id="req-001",
            covers_start=SERVICES_BEGIN,
            covers_end=today,
            services=[SPEECH],
        ),
        today,
    )


def _answered_request(covers_end: date, *, sent: date, answered: date) -> RecordsRequest:
    request = RecordsRequest(
        request_id="req-001",
        covers_start=SERVICES_BEGIN,
        covers_end=covers_end,
        services=[SPEECH],
    )
    return mark_answered(mark_sent(request, sent), answered)


def _cards(ledger, result, requests, today, met=None):
    return decisions_for(
        ledger, result, evaluate_deadlines(ledger, today, met), requests, today
    )


def _ids(cards) -> list[str]:
    return [card.card_id for card in cards]


def _staged(ledger, result, requests, today, met=None):
    return staged_decisions_for(
        ledger, result, evaluate_deadlines(ledger, today, met), requests, today
    )


def _stage(ledger, result, requests, today, prefix: str) -> int:
    matches = [s for s in _staged(ledger, result, requests, today) if s.card.card_id.startswith(prefix)]
    assert len(matches) == 1, f"expected exactly one {prefix!r} card"
    return matches[0].stage


def _key(ledger, result, requests, today, prefix: str) -> str:
    matches = [s for s in _staged(ledger, result, requests, today) if s.card.card_id.startswith(prefix)]
    assert len(matches) == 1, f"expected exactly one {prefix!r} card"
    return matches[0].suppression_key


def _card(cards, prefix: str):
    matches = [c for c in cards if c.card_id.startswith(prefix)]
    assert len(matches) == 1, f"expected exactly one {prefix!r} card, got {_ids(cards)}"
    return matches[0]


# ---------------------------------------------------------------------------
# THE MOST IMPORTANT TEST IN THE PRODUCT
#
# A month in which the school did what it promised must produce nothing. If
# this ever fails, Minutes has become the thing it was built to replace: a tool
# that generates noise until the parent stops reading it.
# ---------------------------------------------------------------------------


def test_a_compliant_month_produces_no_cards_at_all():
    ledger = _ledger(
        deadlines=[
            _deadline(DeadlineKind.ANNUAL_REVIEW, date(2027, 5, 3)),
            _deadline(DeadlineKind.PROGRESS_REPORT, date(2026, 12, 15)),
        ]
    )
    today = date(2026, 11, 1)
    answered = _answered_request(
        date(2026, 10, 7), sent=date(2026, 10, 8), answered=date(2026, 10, 12)
    )

    cards = _cards(ledger, _result(_shortfall()), [answered], today)

    assert cards == [], f"a quiet month raised {_ids(cards)}"
    assert is_quiet(cards) is True


def test_the_quiet_month_survives_every_near_miss_at_once():
    """Each rule sits one step below its bar. None of them may fire."""
    today = date(2026, 11, 1)
    ledger = _ledger(
        deadlines=[
            # One day outside the 60-day annual-review lead time.
            _deadline(DeadlineKind.ANNUAL_REVIEW, today + timedelta(days=61)),
            # One day outside the 14-day progress-report window.
            _deadline(DeadlineKind.PROGRESS_REPORT, today + timedelta(days=15)),
            # One day outside the 90-day reevaluation window.
            _deadline(DeadlineKind.REEVALUATION, today + timedelta(days=91)),
        ]
    )
    # A documented gap one minute under the absolute floor.
    line = _shortfall(owed=600, delivered=600 - (MATERIAL_SHORTFALL_MINUTES - 1))
    # A request answered inside its window, and the cadence one day short.
    answered = _answered_request(
        today - timedelta(days=REQUEST_CADENCE_DAYS - 1),
        sent=today - timedelta(days=20),
        answered=today - timedelta(days=5),
    )

    cards = _cards(ledger, _result(line), [answered], today)

    assert cards == [], f"a near-miss month raised {_ids(cards)}"
    assert is_quiet(cards) is True


def test_is_quiet_is_false_for_even_a_single_routine_card():
    """A routine card still asks the parent to do something."""
    ledger = _ledger()
    today = SERVICES_BEGIN + timedelta(days=REQUEST_CADENCE_DAYS)

    cards = _cards(ledger, _result(), [], today)

    assert [c.urgency for c in cards] == [Urgency.ROUTINE]
    assert is_quiet(cards) is False


# ---------------------------------------------------------------------------
# Rule 1 — an unanswered records request
# ---------------------------------------------------------------------------


def test_silence_card_fires_the_day_after_the_response_window_closes():
    ledger = _ledger()
    sent_on = date(2026, 10, 8)
    sent = mark_sent(
        RecordsRequest(
            request_id="req-001",
            covers_start=SERVICES_BEGIN,
            covers_end=date(2026, 10, 7),
            services=[SPEECH],
        ),
        sent_on,
    )
    response_due = sent_on + timedelta(days=RESPONSE_WINDOW_DAYS)
    assert sent.response_due == response_due

    # On the due date itself the district has still complied: 34 CFR 300.613(a)
    # allows production "in no case more than 45 days after the request".
    on_time = _cards(ledger, _result(), [sent], response_due)
    assert not [c for c in on_time if c.card_id.startswith("records-silence:")]

    # The first day that is no longer true.
    overdue = _cards(ledger, _result(), [sent], response_due + timedelta(days=1))
    card = _card(overdue, "records-silence:")
    assert card.card_id == f"records-silence:req-001:{response_due.isoformat()}"
    assert card.urgency is Urgency.TIME_SENSITIVE


def test_silence_card_states_the_records_failure_not_a_service_failure():
    """The integrity line: silence is evidence about records, never about sessions."""
    ledger = _ledger()
    sent = mark_sent(
        RecordsRequest(
            request_id="req-001",
            covers_start=SERVICES_BEGIN,
            covers_end=date(2026, 10, 7),
            services=[SPEECH],
        ),
        date(2026, 10, 8),
    )
    today = sent.response_due + timedelta(days=10)

    card = _card(_cards(ledger, _result(), [sent], today), "records-silence:")

    prose = " ".join([card.title, card.why_now, *card.facts]).lower()
    assert "records" in prose
    assert "no response is recorded as received" in prose
    # It must never convert the missing paperwork into a claim about delivery.
    for forbidden in ("was not delivered", "were not delivered", "sessions were missed"):
        assert forbidden not in prose


def test_a_request_marked_overdue_before_its_own_deadline_raises_nothing():
    """A contradictory record must not manufacture a silence that has not happened."""
    ledger = _ledger()
    contradictory = RecordsRequest(
        request_id="req-001",
        covers_start=SERVICES_BEGIN,
        covers_end=date(2026, 10, 7),
        services=[SPEECH],
        state="unanswered_overdue",
        sent_on=date(2026, 10, 8),
        response_due=date(2026, 12, 1),
    )

    cards = _cards(ledger, _result(), [contradictory], date(2026, 11, 1))

    assert not [c for c in cards if c.card_id.startswith("records-silence:")]


def test_an_answered_request_never_raises_a_silence_card():
    ledger = _ledger()
    answered = _answered_request(
        date(2026, 10, 7), sent=date(2026, 10, 8), answered=date(2026, 12, 20)
    )

    cards = _cards(ledger, _result(), [answered], date(2027, 1, 15))

    assert not [c for c in cards if c.card_id.startswith("records-silence:")]


def test_an_imminent_meeting_escalates_the_silence_card():
    """34 CFR 300.613(a) owes records before a meeting; that is when it turns urgent."""
    sent = mark_sent(
        RecordsRequest(
            request_id="req-001",
            covers_start=SERVICES_BEGIN,
            covers_end=date(2026, 10, 7),
            services=[SPEECH],
        ),
        date(2026, 10, 8),
    )
    today = sent.response_due + timedelta(days=1)

    far = _deadline(
        DeadlineKind.ANNUAL_REVIEW, today + timedelta(days=RECORDS_BEFORE_MEETING_DAYS + 1)
    )
    near = _deadline(
        DeadlineKind.ANNUAL_REVIEW, today + timedelta(days=RECORDS_BEFORE_MEETING_DAYS)
    )

    outside = _card(
        _cards(_ledger(deadlines=[far]), _result(), [sent], today), "records-silence:"
    )
    assert outside.urgency is Urgency.TIME_SENSITIVE
    assert outside.deadline is None

    inside = _card(
        _cards(_ledger(deadlines=[near]), _result(), [sent], today), "records-silence:"
    )
    assert inside.urgency is Urgency.DEADLINE_IMMINENT
    assert inside.deadline == near.due


def test_a_progress_report_is_not_a_meeting_and_creates_no_records_pressure():
    sent = mark_sent(
        RecordsRequest(
            request_id="req-001",
            covers_start=SERVICES_BEGIN,
            covers_end=date(2026, 10, 7),
            services=[SPEECH],
        ),
        date(2026, 10, 8),
    )
    today = sent.response_due + timedelta(days=1)
    ledger = _ledger(deadlines=[_deadline(DeadlineKind.PROGRESS_REPORT, today + timedelta(days=3))])

    card = _card(_cards(ledger, _result(), [sent], today), "records-silence:")

    assert card.urgency is Urgency.TIME_SENSITIVE
    assert card.deadline is None


# ---------------------------------------------------------------------------
# Rule 2 — the clocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "window"),
    [
        (DeadlineKind.ANNUAL_REVIEW, 60),
        (DeadlineKind.REEVALUATION, 90),
        (DeadlineKind.PROGRESS_REPORT, 14),
        (DeadlineKind.OTHER, 30),
    ],
)
def test_each_deadline_kind_fires_exactly_at_its_own_lead_time(kind, window):
    today = date(2026, 11, 1)
    quiet_requests = [_in_flight(today)]

    outside = _deadline(kind, today + timedelta(days=window + 1))
    assert _cards(_ledger(deadlines=[outside]), _result(), quiet_requests, today) == []

    at_the_boundary = _deadline(kind, today + timedelta(days=window))
    cards = _cards(_ledger(deadlines=[at_the_boundary]), _result(), quiet_requests, today)
    assert _ids(cards) == [f"deadline:{deadline_key(at_the_boundary)}"]


def test_an_overdue_deadline_raises_a_card_and_a_met_one_never_does():
    due = date(2026, 10, 1)
    deadline = _deadline(DeadlineKind.ANNUAL_REVIEW, due)
    ledger = _ledger(deadlines=[deadline])
    today = date(2026, 11, 1)

    overdue = _cards(ledger, _result(), [], today)
    card = _card(overdue, "deadline:")
    assert card.urgency is Urgency.DEADLINE_IMMINENT
    assert card.deadline == due
    assert "that date has passed" in card.title

    met = _cards(ledger, _result(), [], today, met={deadline_key(deadline): date(2026, 10, 5)})
    assert not [c for c in met if c.card_id.startswith("deadline:")]


def test_a_late_but_met_deadline_still_never_interrupts():
    """Lateness is preserved by the engine for the record; it is not an interruption."""
    deadline = _deadline(DeadlineKind.PROGRESS_REPORT, date(2026, 10, 1))
    ledger = _ledger(deadlines=[deadline])
    today = date(2026, 11, 1)

    cards = _cards(
        ledger,
        _result(),
        [_in_flight(today)],
        today,
        met={deadline_key(deadline): date(2026, 10, 30)},
    )

    assert cards == []


def test_a_missed_progress_report_is_capped_below_a_missed_annual_review():
    today = date(2026, 11, 1)
    due = date(2026, 10, 1)
    report = _deadline(DeadlineKind.PROGRESS_REPORT, due)
    review = _deadline(DeadlineKind.ANNUAL_REVIEW, due)

    statuses = evaluate_deadlines(_ledger(deadlines=[report]), today)
    # The engine itself would call this imminent; the cap is what lowers it.
    assert urgency_for(statuses[0]) is Urgency.DEADLINE_IMMINENT

    report_card = _card(_cards(_ledger(deadlines=[report]), _result(), [], today), "deadline:")
    review_card = _card(_cards(_ledger(deadlines=[review]), _result(), [], today), "deadline:")

    assert report_card.urgency is Urgency.TIME_SENSITIVE
    assert review_card.urgency is Urgency.DEADLINE_IMMINENT
    assert _RANK[report_card.urgency] > _RANK[review_card.urgency]


def test_the_cap_never_raises_urgency_above_what_the_engine_computed():
    """A due-soon progress report is time-sensitive either way; the cap must not lift it."""
    today = date(2026, 11, 1)
    report = _deadline(DeadlineKind.PROGRESS_REPORT, today + timedelta(days=IMMINENT_DAYS + 3))
    ledger = _ledger(deadlines=[report])

    status = evaluate_deadlines(ledger, today)[0]
    card = _card(_cards(ledger, _result(), [], today), "deadline:")

    assert urgency_for(status) is Urgency.TIME_SENSITIVE
    assert card.urgency is Urgency.TIME_SENSITIVE


def test_two_deadlines_of_one_kind_on_one_date_get_distinct_cards():
    """kind+date is not unique within an IEP; the description is what tells them apart."""
    due = date(2026, 11, 20)
    first = _deadline(DeadlineKind.OTHER, due, "Assistive technology consultation")
    second = _deadline(DeadlineKind.OTHER, due, "Transportation review")
    ledger = _ledger(deadlines=[first, second])
    today = date(2026, 11, 1)

    cards = _cards(ledger, _result(), [_in_flight(today)], today)

    assert len(cards) == 2
    assert len(set(_ids(cards))) == 2


# ---------------------------------------------------------------------------
# Rule 3 — a documented service gap, and the line it must not cross
# ---------------------------------------------------------------------------


def test_an_entirely_undocumented_gap_never_raises_an_accusation_card():
    """THE INTEGRITY TEST. No records is a reason to ask for records, not to accuse."""
    ledger = _ledger()
    # Every promised minute is unaccounted for, and none of it is documented.
    line = _shortfall(owed=600, delivered=0, undocumented=600)
    assert documented_shortfall_minutes(line) == 0

    cards = _cards(ledger, _result(line), [], date(2026, 10, 7))

    assert not [c for c in cards if c.card_id.startswith("shortfall:")]


def test_a_documented_silence_does_not_by_itself_raise_a_shortfall_card():
    """Silence escalates as a records failure (rule 1), never as a service gap."""
    ledger = _ledger()
    silence = EvidenceRef(
        provenance=Provenance.DOCUMENTED_SILENCE,
        source="req-001",
        detail="2026-10-07: no service record produced for this period.",
    )
    line = _shortfall(owed=600, delivered=0, undocumented=600, evidence=[silence])

    cards = _cards(ledger, _result(line), [], date(2026, 10, 7))

    assert not [c for c in cards if c.card_id.startswith("shortfall:")]


def test_the_undocumented_part_of_a_mixed_gap_is_subtracted_before_the_bar():
    ledger = _ledger()
    today = date(2026, 10, 7)

    # 600-minute gap, but all except 59 minutes of it is unrecorded either way.
    under = _shortfall(owed=600, delivered=0, undocumented=600 - (MATERIAL_SHORTFALL_MINUTES - 1))
    assert documented_shortfall_minutes(under) == MATERIAL_SHORTFALL_MINUTES - 1
    assert not [c for c in _cards(ledger, _result(under), [], today) if c.card_id.startswith("shortfall:")]

    # One more documented minute crosses both bars.
    over = _shortfall(owed=600, delivered=0, undocumented=600 - MATERIAL_SHORTFALL_MINUTES)
    assert documented_shortfall_minutes(over) == MATERIAL_SHORTFALL_MINUTES
    assert _card(_cards(ledger, _result(over), [], today), "shortfall:")


def test_the_absolute_floor_holds_even_when_the_percentage_is_enormous():
    """A small service must not fire on less than one session's worth."""
    ledger = _ledger()
    today = date(2026, 10, 7)

    # 59 of 100 minutes — 59%, far past the ratio, but under one session.
    small = _shortfall(owed=100, delivered=100 - (MATERIAL_SHORTFALL_MINUTES - 1))
    assert not [c for c in _cards(ledger, _result(small), [], today) if c.card_id.startswith("shortfall:")]

    crossing = _shortfall(owed=100, delivered=100 - MATERIAL_SHORTFALL_MINUTES)
    assert _card(_cards(ledger, _result(crossing), [], today), "shortfall:")


def test_the_percentage_floor_holds_even_when_the_absolute_gap_is_large():
    """A large service must not fire on a proportionally trivial difference."""
    ledger = _ledger()
    today = date(2026, 10, 7)
    owed = 2000
    bar = owed * MATERIAL_SHORTFALL_PERCENT // 100  # 200 minutes

    under = _shortfall(owed=owed, delivered=owed - (bar - 1))
    assert documented_shortfall_minutes(under) > MATERIAL_SHORTFALL_MINUTES
    assert not [c for c in _cards(ledger, _result(under), [], today) if c.card_id.startswith("shortfall:")]

    at_bar = _shortfall(owed=owed, delivered=owed - bar)
    assert _card(_cards(ledger, _result(at_bar), [], today), "shortfall:")


def test_excused_minutes_are_out_of_the_gap_and_cannot_drive_a_card():
    """Reconcile already removed them; they must not double as a reason to escalate."""
    ledger = _ledger()
    # The whole difference is explained by recorded absences.
    line = _shortfall(owed=600, delivered=400, excused=200)
    assert line.shortfall_minutes == 0

    cards = _cards(ledger, _result(line), [], date(2026, 10, 7))

    assert not [c for c in cards if c.card_id.startswith("shortfall:")]


def test_several_crossing_services_produce_one_card_naming_all_of_them():
    """One letter answers them all, so one card asks for it."""
    ledger = _ledger(obligations=[_obligation(SPEECH), _obligation(OT)])
    speech = _shortfall(SPEECH, owed=600, delivered=400)
    ot = _shortfall(OT, owed=600, delivered=300)

    card = _card(_cards(ledger, _result(speech, ot), [], date(2026, 10, 7)), "shortfall:")

    assert SPEECH in " ".join(card.facts)
    assert OT in " ".join(card.facts)
    # Biggest documented gap first.
    assert card.facts[0].startswith(OT)


def test_a_shortfall_card_says_out_loud_what_it_does_not_rest_on():
    ledger = _ledger()
    line = _shortfall(owed=600, delivered=200, undocumented=100)

    card = _card(_cards(ledger, _result(line), [], date(2026, 10, 7)), "shortfall:")

    assert "300 of the 400-minute difference is covered by records" in " ".join(card.facts)
    assert "The other 100 minutes have no record either way" in " ".join(card.facts)


# ---------------------------------------------------------------------------
# Rule 4 — the discovery cadence
# ---------------------------------------------------------------------------


def test_the_cadence_card_fires_exactly_when_the_window_has_accrued():
    ledger = _ledger()

    day_before = SERVICES_BEGIN + timedelta(days=REQUEST_CADENCE_DAYS - 2)
    assert _cards(ledger, _result(), [], day_before) == []

    on_cadence = SERVICES_BEGIN + timedelta(days=REQUEST_CADENCE_DAYS - 1)
    card = _card(_cards(ledger, _result(), [], on_cadence), "records-due:")
    assert card.card_id == f"records-due:{SERVICES_BEGIN.isoformat()}"
    assert card.urgency is Urgency.ROUTINE


def test_no_cadence_card_while_a_request_is_still_in_flight():
    ledger = _ledger()
    sent = mark_sent(
        RecordsRequest(
            request_id="req-001",
            covers_start=SERVICES_BEGIN,
            covers_end=date(2026, 10, 7),
            services=[SPEECH],
        ),
        date(2026, 10, 8),
    )

    cards = _cards(ledger, _result(), [sent], date(2026, 11, 1))

    assert cards == []


def test_an_ignored_request_does_not_stall_the_cadence():
    """Both rules fire, and they should: different windows, different asks."""
    ledger = _ledger()
    sent = mark_sent(
        RecordsRequest(
            request_id="req-001",
            covers_start=SERVICES_BEGIN,
            covers_end=date(2026, 10, 7),
            services=[SPEECH],
        ),
        date(2026, 10, 8),
    )
    today = sent.response_due + timedelta(days=REQUEST_CADENCE_DAYS)

    cards = _cards(ledger, _result(), [sent], today)

    silence = _card(cards, "records-silence:")
    cadence = _card(cards, "records-due:")
    # The new request covers the period the ignored one did not.
    assert cadence.card_id == "records-due:2026-10-08"
    assert silence.card_id.startswith("records-silence:req-001:")


def test_an_imminent_meeting_makes_the_routine_cadence_card_time_sensitive():
    today = SERVICES_BEGIN + timedelta(days=REQUEST_CADENCE_DAYS)
    near = _deadline(
        DeadlineKind.REEVALUATION, today + timedelta(days=RECORDS_BEFORE_MEETING_DAYS)
    )
    ledger = _ledger(deadlines=[near])

    card = _card(_cards(ledger, _result(), [], today), "records-due:")

    assert card.urgency is Urgency.TIME_SENSITIVE
    assert card.deadline == near.due


# ---------------------------------------------------------------------------
# Drafts
# ---------------------------------------------------------------------------


def test_every_card_that_carries_a_draft_passes_the_integrity_gate():
    """The invariant the product's credibility rests on."""
    today = date(2026, 11, 20)
    ledger = _ledger(
        obligations=[_obligation(SPEECH), _obligation(OT)],
        deadlines=[
            _deadline(DeadlineKind.ANNUAL_REVIEW, date(2026, 10, 1)),
            _deadline(DeadlineKind.PROGRESS_REPORT, date(2026, 11, 25)),
            _deadline(DeadlineKind.REEVALUATION, date(2027, 1, 10)),
            _deadline(DeadlineKind.OTHER, date(2026, 12, 1)),
        ],
    )
    sent = mark_sent(
        RecordsRequest(
            request_id="req-001",
            covers_start=SERVICES_BEGIN,
            covers_end=date(2026, 9, 30),
            services=[SPEECH, OT],
        ),
        date(2026, 10, 1),
    )
    result = _result(
        _shortfall(SPEECH, owed=600, delivered=300, undocumented=60),
        _shortfall(OT, owed=600, delivered=200),
    )

    cards = _cards(ledger, result, [sent], today)

    drafted = [c for c in cards if c.draft is not None]
    assert len(drafted) >= 4, f"expected drafts across the rules, got {_ids(cards)}"
    for card in cards:
        if card.draft is not None:
            assert validate_letter(card.draft) == [], (
                f"{card.card_id} carries a draft that fails the gate"
            )


def test_no_card_promises_a_draft_it_might_not_have():
    """A dropped draft must not leave the recommended action dangling."""
    today = date(2026, 11, 20)
    ledger = _ledger(deadlines=[_deadline(DeadlineKind.ANNUAL_REVIEW, date(2026, 10, 1))])

    cards = _cards(ledger, _result(_shortfall(owed=600, delivered=200)), [], today)

    assert cards
    for card in cards:
        assert "draft below" not in card.recommended_action.lower()
        assert "letter below" not in card.recommended_action.lower()


def test_card_prose_survives_the_statement_gate_unwithheld():
    """Cards are screened again by statement.py; none of ours may be held back."""
    today = date(2026, 11, 20)
    ledger = _ledger(
        obligations=[_obligation(SPEECH), _obligation(OT)],
        deadlines=[
            _deadline(DeadlineKind.ANNUAL_REVIEW, date(2026, 10, 1)),
            _deadline(DeadlineKind.PROGRESS_REPORT, date(2026, 11, 25)),
        ],
    )
    sent = mark_sent(
        RecordsRequest(
            request_id="req-001",
            covers_start=SERVICES_BEGIN,
            covers_end=date(2026, 9, 30),
            services=[SPEECH, OT],
        ),
        date(2026, 10, 1),
    )
    result = _result(
        _shortfall(SPEECH, owed=600, delivered=300),
        _shortfall(OT, owed=600, delivered=200),
    )

    cards = _cards(ledger, result, [sent], today)
    statement = build_statement(
        ledger, result, evaluate_deadlines(ledger, today), [sent], cards
    )

    assert cards
    assert WITHHELD_CARD_NOTE not in render_markdown(statement)


# ---------------------------------------------------------------------------
# card_id — the suppression contract
# ---------------------------------------------------------------------------


def test_card_ids_are_identical_across_repeated_runs_of_the_same_facts():
    today = date(2026, 11, 20)
    ledger = _ledger(deadlines=[_deadline(DeadlineKind.ANNUAL_REVIEW, date(2026, 12, 20))])
    result = _result(_shortfall(owed=600, delivered=200))

    first = _cards(ledger, result, [], today)
    second = _cards(ledger, result, [], today)

    assert _ids(first) == _ids(second)
    assert first == second


def test_card_ids_survive_the_passage_of_a_week():
    """The weekly-run contract: the same fact keeps its id as ``today`` moves."""
    ledger = _ledger(deadlines=[_deadline(DeadlineKind.ANNUAL_REVIEW, date(2026, 12, 20))])
    sent = mark_sent(
        RecordsRequest(
            request_id="req-001",
            covers_start=SERVICES_BEGIN,
            covers_end=date(2026, 9, 30),
            services=[SPEECH],
        ),
        date(2026, 10, 1),
    )
    result = _result(_shortfall(owed=600, delivered=200))

    week_one = _cards(ledger, result, [sent], date(2026, 11, 20))
    week_two = _cards(ledger, result, [sent], date(2026, 11, 27))

    assert set(_ids(week_one)) == set(_ids(week_two))


def test_a_month_to_date_window_does_not_churn_the_shortfall_id():
    """The id carries period_start, never the moving observation end."""
    ledger = _ledger()
    line = _shortfall(owed=600, delivered=200)

    early = _result(line).model_copy(update={"period_end": date(2026, 10, 7)})
    later = _result(line).model_copy(update={"period_end": date(2026, 10, 21)})

    first = _card(_cards(ledger, early, [], date(2026, 10, 7)), "shortfall:")
    second = _card(_cards(ledger, later, [], date(2026, 10, 21)), "shortfall:")

    assert first.card_id == second.card_id


def test_a_deadline_keeps_its_id_when_it_escalates_from_due_soon_to_overdue():
    """State changed, not the fact. Urgency is what carries the escalation."""
    due = date(2026, 12, 1)
    ledger = _ledger(deadlines=[_deadline(DeadlineKind.ANNUAL_REVIEW, due)])

    soon = _card(_cards(ledger, _result(), [], due - timedelta(days=40)), "deadline:")
    passed = _card(_cards(ledger, _result(), [], due + timedelta(days=10)), "deadline:")

    assert soon.card_id == passed.card_id
    assert soon.urgency is Urgency.TIME_SENSITIVE
    assert passed.urgency is Urgency.DEADLINE_IMMINENT


def test_different_facts_get_different_ids():
    today = date(2026, 11, 20)
    ledger = _ledger(
        obligations=[_obligation(SPEECH), _obligation(OT)],
        deadlines=[
            _deadline(DeadlineKind.ANNUAL_REVIEW, date(2026, 12, 20)),
            _deadline(DeadlineKind.PROGRESS_REPORT, date(2026, 11, 25)),
        ],
    )
    sent = mark_sent(
        RecordsRequest(
            request_id="req-001",
            covers_start=SERVICES_BEGIN,
            covers_end=date(2026, 9, 30),
            services=[SPEECH, OT],
        ),
        date(2026, 10, 1),
    )
    result = _result(
        _shortfall(SPEECH, owed=600, delivered=200),
        _shortfall(OT, owed=600, delivered=200),
    )

    cards = _cards(ledger, result, [sent], today)

    assert len(_ids(cards)) == len(set(_ids(cards)))
    assert len({cid.split(":", 1)[0] for cid in _ids(cards)}) == 4


def test_the_shortfall_id_changes_when_the_set_of_short_services_changes():
    """A service joining the gap is a new fact and must reach the parent."""
    ledger = _ledger(obligations=[_obligation(SPEECH), _obligation(OT)])
    today = date(2026, 10, 7)

    speech_only = _result(
        _shortfall(SPEECH, owed=600, delivered=200), _shortfall(OT, owed=600, delivered=600)
    )
    both = _result(
        _shortfall(SPEECH, owed=600, delivered=200), _shortfall(OT, owed=600, delivered=200)
    )

    first = _card(_cards(ledger, speech_only, [], today), "shortfall:")
    second = _card(_cards(ledger, both, [], today), "shortfall:")

    assert first.card_id != second.card_id


def test_the_shortfall_id_folds_a_respelled_service_name():
    """A district export respelling a service must not mint a second card."""
    ledger = _ledger(obligations=[_obligation(OT)])
    today = date(2026, 10, 7)

    plain = _result(_shortfall(OT, owed=600, delivered=200))
    respelled = _result(_shortfall("Occupational Therapy (OT)", owed=600, delivered=200))

    first = _card(_cards(ledger, plain, [], today), "shortfall:")
    second = _card(_cards(ledger, respelled, [], today), "shortfall:")

    assert first.card_id == second.card_id


# ---------------------------------------------------------------------------
# The escalation ladders — what a suppression layer reads that urgency cannot say
# ---------------------------------------------------------------------------


def test_a_deadline_moves_up_a_rung_the_day_it_is_missed():
    """The rung that urgency could never carry, on both kinds it matters for.

    A progress report is capped at time_sensitive for its whole life, so
    due-soon and missed are the same urgency. An annual review is already
    deadline_imminent a week out, so passing the date does not move it either.
    On urgency alone, neither could ever be raised again — the stage is what
    makes the missed date reachable.
    """
    due = date(2026, 12, 1)
    for kind, urgency in (
        (DeadlineKind.PROGRESS_REPORT, Urgency.TIME_SENSITIVE),
        (DeadlineKind.ANNUAL_REVIEW, Urgency.DEADLINE_IMMINENT),
    ):
        ledger = _ledger(deadlines=[_deadline(kind, due)])
        soon = due - timedelta(days=IMMINENT_DAYS - 1)
        passed = due + timedelta(days=10)

        approaching = _card(_cards(ledger, _result(), [], soon), "deadline:")
        missed = _card(_cards(ledger, _result(), [], passed), "deadline:")

        assert approaching.urgency is urgency is missed.urgency, kind
        assert _stage(ledger, _result(), [], soon, "deadline:") == 0, kind
        assert _stage(ledger, _result(), [], passed, "deadline:") == 1, kind


def test_a_deadline_suppresses_on_its_own_id():
    ledger = _ledger(deadlines=[_deadline(DeadlineKind.PROGRESS_REPORT, date(2026, 12, 1))])
    today = date(2026, 11, 25)
    card = _card(_cards(ledger, _result(), [], today), "deadline:")

    assert _key(ledger, _result(), [], today, "deadline:") == card.card_id


@pytest.mark.parametrize(
    ("documented", "stage"),
    [
        (60, 0),  # exactly the material floor: the first rung
        (119, 0),  # not yet doubled
        (120, 1),  # doubled
        (239, 1),
        (240, 2),  # doubled again
        (960, 4),
    ],
)
def test_a_shortfall_climbs_a_rung_every_time_the_documented_gap_doubles(documented, stage):
    """A gap that has doubled is a different conversation; +13 minutes is not.

    Hand-computed against the constants: the floor is 60 minutes and the factor
    is 2, so the rungs sit at 60, 120, 240, 480, 960. owed is 10x documented
    throughout so the percentage bar is met exactly and the minutes bar is what
    the rungs are measured on.
    """
    assert MATERIAL_SHORTFALL_MINUTES == 60
    assert SHORTFALL_ESCALATION_FACTOR == 2

    ledger = _ledger()
    line = _shortfall(owed=documented * 10, delivered=documented * 9)
    assert documented_shortfall_minutes(line) == documented

    assert _stage(ledger, _result(line), [], date(2026, 10, 7), "shortfall:") == stage


def test_another_service_crossing_the_bar_is_a_rung_of_its_own():
    """Breadth worsens the fact as surely as size does — and it can hide in the total.

    Sized so the doubling ladder cannot notice. Speech alone is 1,000 documented
    minutes short, which sits on rung 4 (the rungs are 60, 120, 240, 480, 960).
    Adding OT's 60 takes the total to 1,060 — rung 4 still, because rung 5 needs
    1,920 — so without a term for breadth, a second service joining the gap
    would never reach the parent at all.
    """
    ledger = _ledger(obligations=[_obligation(SPEECH), _obligation(OT)])
    today = date(2026, 10, 7)

    speech_only = _result(
        _shortfall(SPEECH, owed=10_000, delivered=9_000),
        _shortfall(OT, owed=600, delivered=600),
    )
    both = _result(
        _shortfall(SPEECH, owed=10_000, delivered=9_000),
        _shortfall(OT, owed=600, delivered=540),
    )

    assert _stage(ledger, speech_only, [], today, "shortfall:") == 4
    assert _stage(ledger, both, [], today, "shortfall:") == 5


def test_a_shortfall_suppresses_on_the_period_so_good_news_does_not_interrupt():
    """The id names the exact set of short services; the suppression key does not.

    Building the key from the id would mean a service IMPROVING out of the set
    minted a fresh key and interrupted the parent — alert fatigue caused by the
    news getting better. Both renderings belong to one running conversation
    about one period, and the stage is what says which way it moved.
    """
    ledger = _ledger(obligations=[_obligation(SPEECH), _obligation(OT)])
    today = date(2026, 10, 7)

    both = _result(
        _shortfall(SPEECH, owed=600, delivered=540), _shortfall(OT, owed=600, delivered=540)
    )
    recovered = _result(
        _shortfall(SPEECH, owed=600, delivered=540), _shortfall(OT, owed=600, delivered=600)
    )

    assert _card(_cards(ledger, both, [], today), "shortfall:").card_id != _card(
        _cards(ledger, recovered, [], today), "shortfall:"
    ).card_id
    assert (
        _key(ledger, both, [], today, "shortfall:")
        == _key(ledger, recovered, [], today, "shortfall:")
        == "shortfall:2026-09-08"
    )
    # And the stage falls rather than rises, so it reads as recovery.
    assert _stage(ledger, recovered, [], today, "shortfall:") < _stage(
        ledger, both, [], today, "shortfall:"
    )


def test_a_silence_climbs_a_rung_every_further_response_window():
    """Four days late and four months late are not the same fact.

    But "still nothing, a week later" is not news either, so the rung is a whole
    further 45-day window: due 2026-11-15, so day 44 is rung 0 and day 45 is
    rung 1.
    """
    ledger = _ledger()
    sent = mark_sent(
        RecordsRequest(
            request_id="req-001",
            covers_start=SERVICES_BEGIN,
            covers_end=date(2026, 9, 30),
            services=[SPEECH],
        ),
        date(2026, 10, 1),
    )
    due = sent.response_due
    assert due == date(2026, 11, 15)

    for overdue_days, rung in ((1, 0), (RESPONSE_WINDOW_DAYS - 1, 0), (RESPONSE_WINDOW_DAYS, 1)):
        today = due + timedelta(days=overdue_days)
        assert _stage(ledger, _result(), [sent], today, "records-silence:") == rung, overdue_days


def test_a_records_request_coming_due_has_no_ladder():
    """A request is either due or it is not; there is no worse it can become."""
    ledger = _ledger()
    today = SERVICES_BEGIN + timedelta(days=REQUEST_CADENCE_DAYS + 40)

    assert _stage(ledger, _result(), [], today, "records-due:") == 0


def test_staged_and_plain_decisions_return_the_same_cards_in_the_same_order():
    """decisions_for is the staged call with the extra columns dropped."""
    ledger = _ledger(
        deadlines=[
            _deadline(DeadlineKind.ANNUAL_REVIEW, date(2026, 12, 20)),
            _deadline(DeadlineKind.PROGRESS_REPORT, date(2026, 11, 25)),
        ]
    )
    today = date(2026, 11, 20)
    result = _result(_shortfall(owed=600, delivered=200))

    plain = _cards(ledger, result, [], today)
    staged = _staged(ledger, result, [], today)

    assert plain == [item.card for item in staged]


def test_ids_are_unique_within_one_result():
    today = date(2026, 11, 20)
    ledger = _ledger(
        deadlines=[
            _deadline(DeadlineKind.OTHER, date(2026, 12, 1), "Assistive technology consultation"),
            _deadline(DeadlineKind.OTHER, date(2026, 12, 1), "Transportation review"),
            _deadline(DeadlineKind.PROGRESS_REPORT, date(2026, 11, 25)),
        ]
    )

    cards = _cards(ledger, _result(), [], today)

    assert len(_ids(cards)) == len(set(_ids(cards)))


# ---------------------------------------------------------------------------
# Ordering and shape
# ---------------------------------------------------------------------------


def test_cards_come_back_most_urgent_first():
    today = date(2026, 11, 20)
    ledger = _ledger(
        deadlines=[
            _deadline(DeadlineKind.ANNUAL_REVIEW, date(2026, 10, 1)),  # overdue -> imminent
            _deadline(DeadlineKind.PROGRESS_REPORT, date(2026, 11, 1)),  # overdue -> capped
        ]
    )

    cards = _cards(ledger, _result(), [], today)
    urgencies = [card.urgency for card in cards]

    assert urgencies == sorted(urgencies, key=_RANK.__getitem__)
    assert urgencies[0] is Urgency.DEADLINE_IMMINENT
    assert Urgency.TIME_SENSITIVE in urgencies
    assert urgencies[-1] is Urgency.ROUTINE


def test_every_card_is_filled_in_enough_to_act_on():
    today = date(2026, 11, 20)
    ledger = _ledger(
        deadlines=[
            _deadline(DeadlineKind.ANNUAL_REVIEW, date(2026, 10, 1)),
            _deadline(DeadlineKind.PROGRESS_REPORT, date(2026, 11, 25)),
        ]
    )
    sent = mark_sent(
        RecordsRequest(
            request_id="req-001",
            covers_start=SERVICES_BEGIN,
            covers_end=date(2026, 9, 30),
            services=[SPEECH],
        ),
        date(2026, 10, 1),
    )

    cards = _cards(ledger, _result(_shortfall(owed=600, delivered=200)), [sent], today)

    assert cards
    for card in cards:
        assert card.card_id.count(":") >= 1
        assert card.title.strip()
        assert card.why_now.strip()
        assert card.facts and all(fact.strip() for fact in card.facts)
        assert card.recommended_action.strip()
        # why_now must argue for the timing, not restate the headline.
        assert len(card.why_now) > len(card.title)
        assert card.why_now != card.title


def test_an_empty_ledger_and_an_empty_period_stay_silent():
    ledger = _ledger(obligations=[])
    result = ReconciliationResult(
        period_start=SERVICES_BEGIN, period_end=date(2026, 10, 7), shortfalls=[], events_considered=0
    )

    cards = decisions_for(ledger, result, [], [], date(2026, 10, 7))

    assert cards == []
    assert is_quiet(cards) is True
