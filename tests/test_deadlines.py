"""Deadline-clock tests — pure date arithmetic on synthetic fixtures.

No test in this file makes a network call or an LLM call. Every ledger
below is fabricated for the test; none describes a real child or school.

Dates are hand-computed from ``TODAY`` and written as literals wherever a
constant is being pinned, so that changing a module constant breaks a
test instead of quietly moving it.
"""

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from minutes.deadlines import (
    DUE_SOON_DAYS,
    DUE_SOON_DAYS_BY_KIND,
    IMMINENT_DAYS,
    deadline_key,
    derive_deadlines,
    due_soon_window,
    evaluate_deadlines,
    next_action_date,
    urgency_for,
)
from minutes.models import (
    Accommodation,
    Deadline,
    DeadlineKind,
    DeadlineState,
    DeadlineStatus,
    IEPLedger,
    Period,
    ServiceObligation,
    Urgency,
)

CACHE = Path(__file__).resolve().parents[1] / "fixtures" / "cache" / "iep_maya_ledger.json"

TODAY = date(2026, 11, 1)
IEP_DATE = date(2026, 9, 1)


def _deadline(kind: DeadlineKind, due: date, description: str = "") -> Deadline:
    return Deadline(
        kind=kind,
        due=due,
        description=description or f"synthetic {kind.value} obligation",
        source_quote=f"Synthetic IEP sentence stating {due.isoformat()}.",
    )


def _obligation(**overrides) -> ServiceObligation:
    base = dict(
        service="Speech-Language Therapy",
        minutes_per_session=30,
        sessions_per_period=2,
        period=Period.WEEK,
        provider_role="Licensed Speech-Language Pathologist",
        setting="therapy room",
        start_date=date(2026, 9, 8),
        end_date=date(2027, 6, 11),
        source_quote="Speech-Language Therapy: 30 minutes per session, 2 sessions per week, beginning September 8, 2026.",
    )
    base.update(overrides)
    return ServiceObligation(**base)


def _ledger(
    deadlines: list[Deadline] | None = None,
    obligations: list[ServiceObligation] | None = None,
) -> IEPLedger:
    return IEPLedger(
        student_alias="Test Child (synthetic)",
        school_year="2026-2027",
        iep_date=IEP_DATE,
        obligations=[] if obligations is None else obligations,
        deadlines=[] if deadlines is None else deadlines,
        accommodations=[Accommodation(description="synthetic", source_quote="synthetic")],
    )


def _only(statuses: list[DeadlineStatus]) -> DeadlineStatus:
    assert len(statuses) == 1
    return statuses[0]


def _status_on(kind: DeadlineKind, due: date, **kwargs) -> DeadlineStatus:
    """One deadline due on ``due``, evaluated as of TODAY."""
    return _only(evaluate_deadlines(_ledger([_deadline(kind, due)]), TODAY, **kwargs))


def _status(kind: DeadlineKind, days_out: int, **kwargs) -> DeadlineStatus:
    """One deadline ``days_out`` days from TODAY, evaluated."""
    return _status_on(kind, TODAY + timedelta(days=days_out), **kwargs)


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------


def test_far_future_deadline_is_upcoming():
    status = _status(DeadlineKind.OTHER, 365)
    assert status.state is DeadlineState.UPCOMING
    assert status.days_remaining == 365
    assert status.met_on is None


def test_deadline_inside_window_is_due_soon():
    assert _status(DeadlineKind.OTHER, 10).state is DeadlineState.DUE_SOON


def test_past_due_deadline_is_overdue():
    assert _status(DeadlineKind.OTHER, -1).state is DeadlineState.OVERDUE


def test_met_deadline_is_met_with_its_date():
    ledger = _ledger([_deadline(DeadlineKind.PROGRESS_REPORT, date(2026, 11, 6))])
    key = deadline_key(ledger.deadlines[0])
    assert key == "progress_report:2026-11-06:synthetic progress_report obligation"

    status = _only(evaluate_deadlines(ledger, TODAY, met={key: date(2026, 10, 30)}))
    assert status.state is DeadlineState.MET
    assert status.met_on == date(2026, 10, 30)


def test_met_beats_overdue_but_the_lateness_is_still_recorded():
    """Evidence of satisfaction outranks the calendar — and a report that
    arrived 29 days late must not become indistinguishable from an
    on-time one, because that pattern is the evidence."""
    ledger = _ledger([_deadline(DeadlineKind.PROGRESS_REPORT, date(2026, 10, 1))])
    status = _only(
        evaluate_deadlines(ledger, TODAY, met={deadline_key(ledger.deadlines[0]): date(2026, 10, 30)})
    )
    assert status.state is DeadlineState.MET
    assert status.days_remaining == -31  # still reported, negative
    assert status.met_on > status.deadline.due  # 29 days late, and it shows
    assert urgency_for(status) is Urgency.ROUTINE  # satisfied: never interrupt


# ---------------------------------------------------------------------------
# Boundaries and the constants themselves
# ---------------------------------------------------------------------------


def test_exactly_on_the_due_date_is_due_soon_not_overdue():
    status = _status(DeadlineKind.OTHER, 0)
    assert status.state is DeadlineState.DUE_SOON
    assert status.days_remaining == 0


def test_exactly_at_the_due_soon_threshold_is_due_soon():
    assert _status(DeadlineKind.OTHER, DUE_SOON_DAYS).state is DeadlineState.DUE_SOON


def test_one_day_beyond_the_threshold_is_upcoming():
    assert _status(DeadlineKind.OTHER, DUE_SOON_DAYS + 1).state is DeadlineState.UPCOMING


def test_lead_time_windows_are_exactly_the_researched_values():
    """Pinned as literals, with the reasoning, because these numbers are
    the module's substantive claim: read back out of the module they
    would assert nothing."""
    # 34 CFR 300.613(a) gives the school up to 45 days to produce records,
    # and 300.322(a)(1)-(2) requires a mutually agreed meeting time — the
    # 45-day ceiling has to fit inside the warning with room to schedule.
    assert DUE_SOON_DAYS_BY_KIND[DeadlineKind.ANNUAL_REVIEW] == 60
    # Consent under 300.300(c)(1)(i), then assessment, then a meeting.
    assert DUE_SOON_DAYS_BY_KIND[DeadlineKind.REEVALUATION] == 90
    # Quarterly cadence: 30 days would keep a report flagged a third of
    # the year, which is noise, and 300.320(a)(3)(ii) sets no federal
    # minimum frequency to argue otherwise.
    assert DUE_SOON_DAYS_BY_KIND[DeadlineKind.PROGRESS_REPORT] == 14
    assert DUE_SOON_DAYS_BY_KIND[DeadlineKind.OTHER] == 30
    assert DUE_SOON_DAYS == 30
    assert IMMINENT_DAYS == 7


def test_every_deadline_kind_has_an_explicit_window():
    """No silent fallback: a kind added to models.py without a window here
    is a gap in the reasoning, and must fail rather than inherit 30."""
    assert set(DUE_SOON_DAYS_BY_KIND) == set(DeadlineKind)
    assert [due_soon_window(kind) for kind in DeadlineKind] == [60, 90, 14, 30]


@pytest.mark.parametrize(
    "kind, last_due_soon, first_upcoming",
    [
        # Days after TODAY (2026-11-01), counted by hand.
        (DeadlineKind.ANNUAL_REVIEW, date(2026, 12, 31), date(2027, 1, 1)),  # 60 / 61
        (DeadlineKind.REEVALUATION, date(2027, 1, 30), date(2027, 1, 31)),  # 90 / 91
        (DeadlineKind.PROGRESS_REPORT, date(2026, 11, 15), date(2026, 11, 16)),  # 14 / 15
        (DeadlineKind.OTHER, date(2026, 12, 1), date(2026, 12, 2)),  # 30 / 31
    ],
)
def test_each_kind_flips_to_upcoming_on_a_hand_computed_date(kind, last_due_soon, first_upcoming):
    assert _status_on(kind, last_due_soon).state is DeadlineState.DUE_SOON
    assert _status_on(kind, first_upcoming).state is DeadlineState.UPCOMING


def test_meeting_kinds_get_longer_lead_time_than_progress_reports():
    # Scheduling an IEP meeting and requesting records ahead of it needs
    # more runway than noticing a report did not arrive.
    assert DUE_SOON_DAYS_BY_KIND[DeadlineKind.REEVALUATION] > DUE_SOON_DAYS_BY_KIND[DeadlineKind.ANNUAL_REVIEW]
    assert DUE_SOON_DAYS_BY_KIND[DeadlineKind.ANNUAL_REVIEW] > DUE_SOON_DAYS_BY_KIND[DeadlineKind.OTHER]
    assert DUE_SOON_DAYS_BY_KIND[DeadlineKind.OTHER] > DUE_SOON_DAYS_BY_KIND[DeadlineKind.PROGRESS_REPORT]


def test_annual_review_at_forty_five_days_is_due_soon_but_other_is_not():
    """A parent needs the 34 CFR 300.613(a) 45-day records window to fit
    inside the annual-review warning."""
    assert _status(DeadlineKind.ANNUAL_REVIEW, 45).state is DeadlineState.DUE_SOON
    assert _status(DeadlineKind.OTHER, 45).state is DeadlineState.UPCOMING


# ---------------------------------------------------------------------------
# Past-due arithmetic
# ---------------------------------------------------------------------------


def test_days_remaining_goes_negative_by_exactly_the_days_elapsed():
    ledger = _ledger([_deadline(DeadlineKind.ANNUAL_REVIEW, date(2026, 9, 1))])
    status = _only(evaluate_deadlines(ledger, date(2026, 11, 1)))
    assert status.days_remaining == -61
    assert status.state is DeadlineState.OVERDUE


def test_long_overdue_deadline_stays_overdue():
    status = _status(DeadlineKind.REEVALUATION, -900)
    assert status.state is DeadlineState.OVERDUE
    assert status.days_remaining == -900


# ---------------------------------------------------------------------------
# ``met`` key scheme
# ---------------------------------------------------------------------------


def _four_progress_reports() -> IEPLedger:
    shared = "Progress report on all goals provided to parents"
    return _ledger(
        [
            _deadline(DeadlineKind.PROGRESS_REPORT, date(2026, 11, 6), shared),
            _deadline(DeadlineKind.PROGRESS_REPORT, date(2027, 1, 29), shared),
            _deadline(DeadlineKind.PROGRESS_REPORT, date(2027, 4, 9), shared),
            _deadline(DeadlineKind.PROGRESS_REPORT, date(2027, 6, 11), shared),
        ]
    )


def _same_day_deadlines() -> IEPLedger:
    """Two obligations of one kind falling on one date — the case a
    kind+date key cannot tell apart. 'Individual Counseling' sorts first,
    so a key that guessed would pick the wrong one here."""
    return _ledger(
        [
            _deadline(DeadlineKind.OTHER, date(2026, 9, 8), "Speech-Language Therapy assessment delivered"),
            _deadline(DeadlineKind.OTHER, date(2026, 9, 8), "Individual Counseling assessment delivered"),
        ]
    )


def test_deadline_key_tells_apart_two_deadlines_sharing_kind_and_date():
    """The key a caller is told to use must name one deadline. Crediting
    the wrong service is two falsehoods at once: a compliance credit the
    ledger has no evidence for, and an accusation against a service the
    parent documented."""
    ledger = _same_day_deadlines()
    speech, counseling = ledger.deadlines
    assert deadline_key(speech) != deadline_key(counseling)

    statuses = evaluate_deadlines(ledger, TODAY, met={deadline_key(speech): date(2026, 9, 15)})
    by_description = {s.deadline.description: s for s in statuses}
    assert by_description[speech.description].state is DeadlineState.MET
    assert by_description[speech.description].met_on == date(2026, 9, 15)
    assert by_description[counseling.description].state is DeadlineState.OVERDUE
    assert by_description[counseling.description].met_on is None


def test_a_kind_and_date_key_matching_two_deadlines_raises_instead_of_guessing():
    ledger = _same_day_deadlines()
    with pytest.raises(ValueError, match="ambiguous"):
        evaluate_deadlines(ledger, TODAY, met={"other:2026-09-08": date(2026, 9, 15)})


def test_a_kind_and_date_key_still_works_when_it_names_exactly_one_deadline():
    ledger = _four_progress_reports()
    statuses = evaluate_deadlines(
        ledger, date(2026, 11, 10), met={"progress_report:2026-11-06": date(2026, 11, 6)}
    )
    met = [s for s in statuses if s.state is DeadlineState.MET]
    assert len(met) == 1
    assert met[0].deadline.due == date(2026, 11, 6)


def test_kind_key_marks_only_the_latest_already_due_deadline():
    ledger = _four_progress_reports()
    statuses = evaluate_deadlines(ledger, date(2027, 2, 10), met={"progress_report": date(2027, 2, 2)})

    by_due = {s.deadline.due: s for s in statuses}
    assert by_due[date(2027, 1, 29)].state is DeadlineState.MET
    assert by_due[date(2026, 11, 6)].state is DeadlineState.OVERDUE  # one date, one satisfaction
    assert by_due[date(2027, 4, 9)].state is DeadlineState.UPCOMING


def test_shared_description_key_resolves_the_same_way():
    ledger = _four_progress_reports()
    statuses = evaluate_deadlines(
        ledger,
        date(2027, 2, 10),
        met={"Progress report on all goals provided to parents": date(2027, 2, 2)},
    )
    met = [s for s in statuses if s.state is DeadlineState.MET]
    assert len(met) == 1
    assert met[0].deadline.due == date(2027, 1, 29)


def test_kind_key_before_anything_is_due_marks_the_earliest_as_met_early():
    ledger = _four_progress_reports()
    statuses = evaluate_deadlines(ledger, date(2026, 11, 3), met={"progress_report": date(2026, 11, 2)})
    met = [s for s in statuses if s.state is DeadlineState.MET]
    assert len(met) == 1
    assert met[0].deadline.due == date(2026, 11, 6)
    assert met[0].days_remaining == 3


def test_several_qualified_keys_mark_several_deadlines():
    ledger = _four_progress_reports()
    statuses = evaluate_deadlines(
        ledger,
        date(2027, 2, 10),
        met={
            deadline_key(ledger.deadlines[0]): date(2026, 11, 6),
            deadline_key(ledger.deadlines[1]): date(2027, 2, 2),
        },
    )
    assert sum(1 for s in statuses if s.state is DeadlineState.MET) == 2


def test_two_keys_resolving_to_one_deadline_raise_rather_than_dropping_a_date():
    """The caller supplied two satisfaction dates. Recording one and
    discarding the other would under-report the school's compliance —
    the exact harm this module raises ValueError to prevent."""
    ledger = _four_progress_reports()
    met = {
        "progress_report": date(2027, 2, 2),
        deadline_key(ledger.deadlines[1]): date(2027, 1, 20),
    }
    with pytest.raises(ValueError, match="both resolve to the same deadline"):
        evaluate_deadlines(ledger, date(2027, 2, 10), met=met)


def test_a_weaker_key_is_never_silently_superseded_by_a_stronger_one():
    """Two ambiguous keys, two dates, one deadline between them."""
    ledger = _four_progress_reports()
    with pytest.raises(ValueError, match="both resolve to the same deadline"):
        evaluate_deadlines(
            ledger,
            date(2026, 11, 10),
            met={
                "progress_report": date(2026, 11, 5),
                "Progress report on all goals provided to parents": date(2026, 11, 6),
            },
        )


def test_result_does_not_depend_on_met_dict_insertion_order():
    ledger = _four_progress_reports()
    forward = {
        deadline_key(ledger.deadlines[0]): date(2026, 11, 6),
        deadline_key(ledger.deadlines[1]): date(2027, 1, 20),
    }
    reverse = dict(reversed(list(forward.items())))
    assert evaluate_deadlines(ledger, date(2027, 2, 10), met=forward) == evaluate_deadlines(
        ledger, date(2027, 2, 10), met=reverse
    )


def test_a_collision_reads_the_same_whichever_order_the_keys_arrive_in():
    ledger = _four_progress_reports()
    forward = {
        "progress_report": date(2027, 2, 2),
        deadline_key(ledger.deadlines[1]): date(2027, 1, 20),
    }
    reverse = dict(reversed(list(forward.items())))

    messages = []
    for mapping in (forward, reverse):
        with pytest.raises(ValueError) as raised:
            evaluate_deadlines(ledger, date(2027, 2, 10), met=mapping)
        messages.append(str(raised.value))
    assert messages[0] == messages[1]


def test_unmatched_met_key_raises_rather_than_being_dropped():
    ledger = _four_progress_reports()
    with pytest.raises(ValueError, match="matches no deadline"):
        evaluate_deadlines(ledger, TODAY, met={"annual_review": date(2026, 10, 1)})


def test_malformed_met_key_raises():
    ledger = _four_progress_reports()
    with pytest.raises(ValueError, match="matches no deadline"):
        evaluate_deadlines(ledger, TODAY, met={"not-a-key": date(2026, 10, 1)})


def test_a_kind_and_date_key_naming_a_date_no_deadline_falls_on_raises():
    ledger = _four_progress_reports()
    with pytest.raises(ValueError, match="matches no deadline"):
        evaluate_deadlines(ledger, TODAY, met={"progress_report:2026-11-07": date(2026, 10, 1)})


def test_description_key_containing_a_colon_is_not_mistaken_for_an_exact_key():
    ledger = _ledger([_deadline(DeadlineKind.OTHER, date(2026, 12, 1), "Reviews and Reporting: quarterly")])
    status = _only(
        evaluate_deadlines(ledger, TODAY, met={"Reviews and Reporting: quarterly": date(2026, 10, 28)})
    )
    assert status.state is DeadlineState.MET


def test_a_qualified_key_round_trips_a_description_containing_colons():
    ledger = _ledger([_deadline(DeadlineKind.OTHER, date(2026, 12, 1), "Reviews and Reporting: quarterly")])
    key = deadline_key(ledger.deadlines[0])
    assert key == "other:2026-12-01:Reviews and Reporting: quarterly"
    assert _only(evaluate_deadlines(ledger, TODAY, met={key: date(2026, 10, 28)})).state is DeadlineState.MET


# ---------------------------------------------------------------------------
# Implausible satisfaction dates
# ---------------------------------------------------------------------------


def test_a_met_date_in_the_future_raises():
    ledger = _ledger([_deadline(DeadlineKind.ANNUAL_REVIEW, date(2027, 9, 1))])
    with pytest.raises(ValueError, match="future"):
        evaluate_deadlines(ledger, TODAY, met={"annual_review": date(2030, 1, 1)})


def test_a_met_date_before_the_iep_existed_raises():
    ledger = _ledger([_deadline(DeadlineKind.ANNUAL_REVIEW, date(2027, 9, 1))])
    with pytest.raises(ValueError, match="predates"):
        evaluate_deadlines(ledger, TODAY, met={"annual_review": date(1990, 1, 1)})


def test_a_met_date_on_the_iep_date_itself_is_allowed():
    ledger = _ledger([_deadline(DeadlineKind.OTHER, date(2026, 9, 15))])
    assert _only(evaluate_deadlines(ledger, TODAY, met={"other": IEP_DATE})).state is DeadlineState.MET


def test_a_met_date_of_today_is_allowed():
    ledger = _ledger([_deadline(DeadlineKind.PROGRESS_REPORT, date(2026, 11, 6))])
    status = _only(evaluate_deadlines(ledger, TODAY, met={"progress_report": TODAY}))
    assert status.state is DeadlineState.MET
    assert status.met_on == TODAY


# ---------------------------------------------------------------------------
# derive_deadlines: the IEP's stated dates, and nothing else
# ---------------------------------------------------------------------------


def test_stated_deadlines_are_preferred_and_returned_unmodified():
    stated = _deadline(DeadlineKind.ANNUAL_REVIEW, date(2027, 9, 1))
    derived = derive_deadlines(_ledger([stated]))
    assert derived == [stated]


def test_stated_deadlines_come_back_in_canonical_order():
    early = _deadline(DeadlineKind.PROGRESS_REPORT, date(2026, 11, 6))
    late = _deadline(DeadlineKind.REEVALUATION, date(2028, 11, 14))
    middle = _deadline(DeadlineKind.ANNUAL_REVIEW, date(2027, 9, 1))
    assert derive_deadlines(_ledger([late, early, middle])) == [early, middle, late]


def test_projected_service_start_dates_are_not_turned_into_deadlines():
    """34 CFR 300.320(a)(7) requires the IEP to state a *projected* start
    date. A projection is a plan, not a date the school breaches the day
    after, and no verified regulation converts it into one."""
    ledger = _ledger(obligations=[_obligation(), _obligation(service="Occupational Therapy")])
    assert derive_deadlines(ledger) == []


def test_obligations_alone_never_produce_an_accusation_from_silence():
    """The cardinal sin, guarded: a parent who has not reported on a
    service has said nothing, and nothing is not a missed obligation.
    Undelivered minutes are measured in reconciliation, where the absence
    of a record is reported as undocumented rather than as a breach."""
    ledger = _ledger(
        obligations=[
            _obligation(start_date=date(2026, 9, 8)),
            _obligation(service="Occupational Therapy", start_date=date(2026, 9, 8)),
            _obligation(service="Individual Counseling", start_date=date(2026, 9, 8)),
        ]
    )
    statuses = evaluate_deadlines(ledger, TODAY)  # two months past every start date
    assert statuses == []
    assert next_action_date(statuses) is None


def test_nothing_is_invented_beyond_what_the_iep_states():
    """No annual-review anniversary, no triennial, no start date: the
    first two are not federally computable from this ledger and the third
    is a projection."""
    stated = _deadline(DeadlineKind.PROGRESS_REPORT, date(2026, 11, 6))
    assert derive_deadlines(_ledger([stated], obligations=[_obligation()])) == [stated]


def test_deadlines_come_back_in_due_date_order():
    ledger = _ledger(
        [
            _deadline(DeadlineKind.REEVALUATION, date(2028, 11, 14)),
            _deadline(DeadlineKind.PROGRESS_REPORT, date(2026, 11, 6)),
            _deadline(DeadlineKind.ANNUAL_REVIEW, date(2027, 9, 1)),
        ]
    )
    dues = [s.deadline.due for s in evaluate_deadlines(ledger, TODAY)]
    assert dues == sorted(dues)


# ---------------------------------------------------------------------------
# Empty ledger
# ---------------------------------------------------------------------------


def test_empty_ledger_evaluates_to_nothing():
    assert evaluate_deadlines(_ledger(), TODAY) == []


def test_next_action_date_of_nothing_is_none():
    assert next_action_date([]) is None


def test_empty_met_mapping_is_equivalent_to_none():
    ledger = _four_progress_reports()
    assert evaluate_deadlines(ledger, TODAY, met={}) == evaluate_deadlines(ledger, TODAY)


# ---------------------------------------------------------------------------
# Urgency mapping
# ---------------------------------------------------------------------------


def test_upcoming_is_routine():
    assert urgency_for(_status(DeadlineKind.OTHER, 200)) is Urgency.ROUTINE


def test_due_soon_outside_the_imminent_window_is_time_sensitive():
    assert urgency_for(_status(DeadlineKind.OTHER, IMMINENT_DAYS + 1)) is Urgency.TIME_SENSITIVE


def test_due_soon_at_the_imminent_boundary_is_imminent():
    assert urgency_for(_status(DeadlineKind.OTHER, IMMINENT_DAYS)) is Urgency.DEADLINE_IMMINENT


def test_the_imminent_window_is_exactly_seven_days():
    """Hand-computed from TODAY (2026-11-01) so the boundary is pinned
    independently of the constant and of fixtures/cache/."""
    assert urgency_for(_status_on(DeadlineKind.OTHER, date(2026, 11, 8))) is Urgency.DEADLINE_IMMINENT
    assert urgency_for(_status_on(DeadlineKind.OTHER, date(2026, 11, 9))) is Urgency.TIME_SENSITIVE


def test_due_today_is_imminent():
    assert urgency_for(_status(DeadlineKind.OTHER, 0)) is Urgency.DEADLINE_IMMINENT


def test_overdue_is_imminent():
    assert urgency_for(_status(DeadlineKind.OTHER, -3)) is Urgency.DEADLINE_IMMINENT


def test_met_never_interrupts_the_parent():
    ledger = _ledger([_deadline(DeadlineKind.PROGRESS_REPORT, date(2026, 10, 1))])
    status = _only(
        evaluate_deadlines(ledger, TODAY, met={deadline_key(ledger.deadlines[0]): date(2026, 10, 1)})
    )
    assert urgency_for(status) is Urgency.ROUTINE


# ---------------------------------------------------------------------------
# next_action_date
# ---------------------------------------------------------------------------


def test_next_action_date_is_the_soonest_unmet_due_date():
    ledger = _four_progress_reports()
    statuses = evaluate_deadlines(ledger, date(2026, 11, 1))
    assert next_action_date(statuses) == date(2026, 11, 6)


def test_next_action_date_skips_met_deadlines():
    ledger = _four_progress_reports()
    statuses = evaluate_deadlines(
        ledger, date(2026, 11, 10), met={"progress_report:2026-11-06": date(2026, 11, 6)}
    )
    assert next_action_date(statuses) == date(2027, 1, 29)


def test_next_action_date_returns_a_past_date_when_something_is_overdue():
    ledger = _four_progress_reports()
    statuses = evaluate_deadlines(ledger, date(2027, 5, 1))
    assert next_action_date(statuses) == date(2026, 11, 6)


def test_next_action_date_is_none_when_everything_is_met():
    ledger = _ledger([_deadline(DeadlineKind.PROGRESS_REPORT, date(2026, 11, 6))])
    statuses = evaluate_deadlines(ledger, TODAY, met={"progress_report": date(2026, 10, 30)})
    assert next_action_date(statuses) is None


# ---------------------------------------------------------------------------
# The cached real ledger
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CACHE.exists(), reason="run scripts/extract_once.py to build the cache")
def test_cached_ledger_clocks_read_correctly_in_november():
    ledger = IEPLedger.model_validate(json.loads(CACHE.read_text(encoding="utf-8")))
    statuses = evaluate_deadlines(ledger, date(2026, 11, 1))

    # The dates the IEP states, and only those: the four services carry
    # projected start dates, which are not clocked as obligations.
    assert ledger.obligations
    assert len(statuses) == len(ledger.deadlines)

    by_state: dict[DeadlineState, list[DeadlineStatus]] = {}
    for status in statuses:
        by_state.setdefault(status.state, []).append(status)

    # Nothing in this ledger evidences a missed date, and no evidence was
    # supplied, so nothing is overdue. Silence accuses no one.
    assert DeadlineState.OVERDUE not in by_state
    assert DeadlineState.MET not in by_state

    # The first progress report is 5 days out; everything else is further.
    due_soon = by_state[DeadlineState.DUE_SOON]
    assert [s.deadline.due for s in due_soon] == [date(2026, 11, 6)]
    assert due_soon[0].days_remaining == 5
    assert urgency_for(due_soon[0]) is Urgency.DEADLINE_IMMINENT
    assert len(by_state[DeadlineState.UPCOMING]) == len(ledger.deadlines) - 1

    assert next_action_date(statuses) == date(2026, 11, 6)


@pytest.mark.skipif(not CACHE.exists(), reason="run scripts/extract_once.py to build the cache")
def test_cached_ledger_annual_review_warns_before_the_meeting():
    ledger = IEPLedger.model_validate(json.loads(CACHE.read_text(encoding="utf-8")))
    annual = next(d for d in ledger.deadlines if d.kind is DeadlineKind.ANNUAL_REVIEW)

    statuses = evaluate_deadlines(ledger, annual.due - timedelta(days=50))
    review = next(s for s in statuses if s.deadline.kind is DeadlineKind.ANNUAL_REVIEW)
    assert review.state is DeadlineState.DUE_SOON
    assert urgency_for(review) is Urgency.TIME_SENSITIVE
