"""Reconciliation tests — pure arithmetic, no network, no model calls.

Every expected number in this file is computed by hand from the school
calendar and written as a literal. Nothing here calls the code under test to
decide what the answer should be, because a wrong number in this module is a
false accusation against a school.

The windows used throughout, counted by hand:

  2026-10-05 (Mon) .. 2026-10-30 (Fri)  -> four whole school weeks, 20 school days
      Oct 5-9, 12-16, 19-23, 26-30 = 5 + 5 + 5 + 5
      As calendar time: 26 days of a 31-day month = 26/31 of a month.

  2026-09-08 (Tue) .. 2026-09-30 (Wed)  -> 17 school days
      Sep 8-11 (4), 14-18 (5), 21-25 (5), 28-30 (3)

  2027-03-01 (Mon) .. 2027-03-31 (Wed)  -> 23 school days, exactly 1 calendar month
      Mar 1-5, 8-12, 15-19, 22-26, 29-31 = 5 + 5 + 5 + 5 + 3

  2026-09-08 (Tue) .. 2026-12-07 (Mon)  -> 65 school days, exactly 3 calendar months
  2026-08-20 (Thu) .. 2027-08-19 (Thu)  -> 261 school days, exactly 12 calendar months
  2026-08-20 (Thu) .. 2027-06-11 (Fri)  -> 212 school days, 9 + 23/31 calendar months

All students, schools and records below are synthetic.
"""

from datetime import date
from pathlib import Path

import pytest

from minutes import reconcile as reconcile_module
from minutes.correspondence import load_cached_events
from minutes.models import (
    Attribution,
    IEPLedger,
    Period,
    Provenance,
    ServiceEvent,
    ServiceObligation,
)
from minutes.reconcile import (
    ABSENCE_NOTE,
    CALENDAR_MONTHS_PER_PERIOD,
    SCHOOL_DAYS_PER_PERIOD,
    canonical_service,
    expected_sessions,
    reconcile,
    unmatched_events,
)

WINDOW_START = date(2026, 10, 5)
WINDOW_END = date(2026, 10, 30)

# Speech at 2 x 30 min/week over 20 school days = 20 * 2 // 5 = 8 sessions.
SPEECH_SESSIONS = 8
SPEECH_OWED = 240

# The eight weekdays a 2-a-week service would plausibly be scheduled on inside
# the window, used wherever a test needs every promised slot to carry a record.
SESSION_DAYS = [
    date(2026, 10, 5), date(2026, 10, 7),
    date(2026, 10, 12), date(2026, 10, 14),
    date(2026, 10, 19), date(2026, 10, 21),
    date(2026, 10, 26), date(2026, 10, 28),
]


def obligation(**overrides) -> ServiceObligation:
    base = dict(
        service="Speech-Language Therapy",
        minutes_per_session=30,
        sessions_per_period=2,
        period=Period.WEEK,
        provider_role="Licensed Speech-Language Pathologist",
        setting="therapy room",
        start_date=date(2026, 9, 8),
        end_date=date(2027, 6, 11),
        source_quote="Speech-Language Therapy: 30 minutes per session, 2 sessions per week.",
    )
    base.update(overrides)
    return ServiceObligation(**base)


def ledger(*obligations: ServiceObligation) -> IEPLedger:
    return IEPLedger(
        student_alias="Test Child (synthetic)",
        school_year="2026-2027",
        iep_date=date(2026, 9, 1),
        obligations=list(obligations),
        deadlines=[],
        accommodations=[],
    )


def held(
    day: date,
    minutes: int = 30,
    service: str = "Speech-Language Therapy",
    provenance: Provenance = Provenance.SCHOOL_CONFIRMED,
    source: str | None = None,
) -> ServiceEvent:
    return ServiceEvent(
        event_date=day,
        service=service,
        minutes=minutes,
        delivered=True,
        provenance=provenance,
        source=source or f"log-{day.isoformat()}",
    )


def missed(
    day: date,
    service: str = "Speech-Language Therapy",
    provenance: Provenance = Provenance.SCHOOL_CONFIRMED,
    source: str | None = None,
) -> ServiceEvent:
    return ServiceEvent(
        event_date=day,
        service=service,
        minutes=0,
        delivered=False,
        provenance=provenance,
        source=source or f"log-{day.isoformat()}",
    )


def absent(
    day: date,
    service: str = "Speech-Language Therapy",
    provenance: Provenance = Provenance.SCHOOL_CONFIRMED,
    source: str | None = None,
) -> ServiceEvent:
    """A non-delivery the source document attributes to the child being away."""
    return missed(day, service, provenance, source).model_copy(
        update={"attribution": Attribution.STUDENT_ABSENCE}
    )


def silence(day: date, service: str = "Speech-Language Therapy", source: str | None = None) -> ServiceEvent:
    """What discovery.py emits for an overdue, unanswered records request."""
    return ServiceEvent(
        event_date=day,
        service=service,
        minutes=0,
        delivered=False,
        provenance=Provenance.DOCUMENTED_SILENCE,
        source=source or f"req-{day.isoformat()}",
    )


def only(result) -> "object":
    assert len(result.shortfalls) == 1
    return result.shortfalls[0]


# --------------------------------------------------------------------------
# The public surface
# --------------------------------------------------------------------------


def test_the_public_surface_is_pinned():
    assert set(reconcile_module.__all__) == {
        "CALENDAR_MONTHS_PER_PERIOD",
        "SCHOOL_DAYS_PER_PERIOD",
        "canonical_service",
        "expected_sessions",
        "reconcile",
        "unmatched_events",
    }


# --------------------------------------------------------------------------
# expected_sessions
# --------------------------------------------------------------------------


def test_weekly_service_over_four_whole_school_weeks():
    # 20 school days x 2 sessions // 5 school days per week = 8.
    assert expected_sessions(obligation(), WINDOW_START, WINDOW_END) == 8


def test_weekends_are_never_school_days():
    # Sat 2026-10-10 through Sun 2026-10-11: no school days, so nothing owed.
    assert expected_sessions(obligation(), date(2026, 10, 10), date(2026, 10, 11)) == 0


def test_partial_week_floors_rather_than_inflating_the_promise():
    # Mon 2026-10-05 .. Wed 2026-10-07 = 3 school days.
    # 3 x 2 // 5 = 1 session, not 2 and not a rounded-up 1.2.
    assert expected_sessions(obligation(), date(2026, 10, 5), date(2026, 10, 7)) == 1


def test_window_is_clipped_to_the_obligation_start_date():
    # Requested Sep 1 (Tue) .. Sep 30 (Wed) is 22 school days:
    #   Sep 1-4 (4), 7-11 (5), 14-18 (5), 21-25 (5), 28-30 (3).
    # The service does not begin until Sep 8, leaving 17 school days.
    # Unclipped: 22 x 2 // 5 = 8.  Clipped: 17 x 2 // 5 = 6.
    assert expected_sessions(obligation(), date(2026, 9, 1), date(2026, 9, 30)) == 6


def test_window_is_clipped_to_the_obligation_end_date():
    # Service ends Oct 16, so only Oct 5-9 and Oct 12-16 count = 10 school days.
    # 10 x 2 // 5 = 4.
    ended = obligation(end_date=date(2026, 10, 16))
    assert expected_sessions(ended, WINDOW_START, WINDOW_END) == 4


def test_no_overlap_owes_nothing():
    future = obligation(start_date=date(2027, 1, 4), end_date=date(2027, 6, 11))
    assert expected_sessions(future, WINDOW_START, WINDOW_END) == 0


def test_daily_service_counts_every_school_day():
    daily = obligation(
        service="Specialized Academic Instruction",
        minutes_per_session=60,
        sessions_per_period=1,
        period=Period.DAY,
    )
    assert expected_sessions(daily, WINDOW_START, WINDOW_END) == 20


def test_monthly_service_over_one_nominal_school_month():
    # A month is worth 20 school days, and the window is exactly 20. As calendar
    # time the window is 26/31 of a month, and 1 x 26/31 rounds up to 1, so the
    # calendar cap does not bite either.
    monthly = obligation(
        service="Individual Counseling",
        minutes_per_session=30,
        sessions_per_period=1,
        period=Period.MONTH,
    )
    assert expected_sessions(monthly, WINDOW_START, WINDOW_END) == 1


def test_monthly_service_over_two_nominal_school_months():
    # Mon 2026-10-05 .. Fri 2026-11-27 = 20 October school days + 20 November
    # school days (Nov 2-6, 9-13, 16-20, 23-27) = 40 // 20 = 2 sessions.
    # Calendar cap: Oct 5 .. Nov 27 is 1 + 23/30 months, and 1 x that rounds up
    # to 2, so the cap agrees.
    monthly = obligation(
        service="Individual Counseling",
        minutes_per_session=30,
        sessions_per_period=1,
        period=Period.MONTH,
    )
    assert expected_sessions(monthly, WINDOW_START, date(2026, 11, 27)) == 2


def test_a_real_calendar_month_owes_one_month_not_twenty_three_school_days():
    # March 2027 holds 23 weekdays but a month is a month. A 20-a-month service
    # owes 20 sessions there, not the 23 that weekday pro-rata alone would give.
    monthly = obligation(
        service="Specialized Academic Instruction",
        minutes_per_session=30,
        sessions_per_period=20,
        period=Period.MONTH,
    )
    assert expected_sessions(monthly, date(2027, 3, 1), date(2027, 3, 31)) == 20


def test_a_long_calendar_month_does_not_inflate_a_smaller_monthly_promise():
    # October 2026 holds 22 weekdays. 22 x 12 // 20 = 13 by pro-rata alone;
    # one calendar month of a 12-a-month service is 12.
    monthly = obligation(
        service="Individual Counseling",
        minutes_per_session=30,
        sessions_per_period=12,
        period=Period.MONTH,
    )
    assert expected_sessions(monthly, date(2026, 10, 1), date(2026, 10, 31)) == 12


def test_a_partial_calendar_month_still_pro_rates():
    # 4 a month over Oct 5 .. Oct 30: pro-rata is 20 x 4 // 20 = 4, and the
    # calendar cap is 4 x 26/31 = 3.35 rounded up to 4. The cap trims inflation,
    # it does not zero out a window that covers most of a month.
    monthly = obligation(
        service="Individual Counseling",
        minutes_per_session=30,
        sessions_per_period=4,
        period=Period.MONTH,
    )
    assert expected_sessions(monthly, WINDOW_START, WINDOW_END) == 4


def test_quarterly_service_over_exactly_one_calendar_quarter():
    # 2026-09-08 .. 2026-12-07 is exactly three calendar months = one quarter,
    # so a 12-a-quarter service owes exactly 12. The nominal 45-school-day
    # quarter would have said 65 x 12 // 45 = 17, inventing five sessions,
    # because a real quarter holds 65 weekdays and not 45.
    quarterly = obligation(
        service="Individual Counseling",
        minutes_per_session=30,
        sessions_per_period=12,
        period=Period.QUARTER,
        start_date=date(2026, 8, 1),
    )
    assert expected_sessions(quarterly, date(2026, 9, 8), date(2026, 12, 7)) == 12


def test_yearly_service_over_exactly_one_calendar_year():
    # 2026-08-20 .. 2027-08-19 is exactly twelve calendar months, so a
    # 36-a-year service owes exactly 36. The nominal 180-school-day year would
    # have said 261 x 36 // 180 = 52.
    yearly = obligation(
        service="Individual Counseling",
        minutes_per_session=60,
        sessions_per_period=36,
        period=Period.YEAR,
        start_date=date(2026, 8, 1),
        end_date=date(2027, 12, 31),
    )
    assert expected_sessions(yearly, date(2026, 8, 20), date(2027, 8, 19)) == 36


def test_yearly_service_over_a_school_year_under_states_rather_than_inflates():
    # 2026-08-20 .. 2027-06-11 is 9 + 23/31 calendar months = 0.8118 of a year.
    # 36 x 0.8118 = 29.23, rounded up to 30. Weekday pro-rata would have said
    # 212 x 36 // 180 = 42 — six phantom sessions of shortfall against the
    # school. 30 under-states what a school-year service owes, and that is the
    # deliberate direction: under-stating owed under-states the accusation.
    yearly = obligation(
        service="Individual Counseling",
        minutes_per_session=60,
        sessions_per_period=36,
        period=Period.YEAR,
        start_date=date(2026, 8, 1),
        end_date=date(2027, 12, 31),
    )
    assert expected_sessions(yearly, date(2026, 8, 20), date(2027, 6, 11)) == 30


def test_every_period_has_a_school_day_factor():
    assert set(SCHOOL_DAYS_PER_PERIOD) == set(Period)


def test_only_the_approximate_periods_are_calendar_capped():
    # A school day is exactly one weekday and a school week exactly five, so
    # those two factors need no correction. The other three are instructional
    # day counts standing in for calendar time, and are capped.
    assert set(CALENDAR_MONTHS_PER_PERIOD) == {Period.MONTH, Period.QUARTER, Period.YEAR}


def test_expected_sessions_rejects_a_reversed_window():
    with pytest.raises(ValueError):
        expected_sessions(obligation(), WINDOW_END, WINDOW_START)


# --------------------------------------------------------------------------
# reconcile — the core cases
# --------------------------------------------------------------------------


def test_full_delivery_leaves_no_shortfall_and_nothing_undocumented():
    assert len(SESSION_DAYS) == SPEECH_SESSIONS

    result = reconcile(ledger(obligation()), [held(d) for d in SESSION_DAYS], WINDOW_START, WINDOW_END)
    line = only(result)

    assert line.owed_minutes == SPEECH_OWED
    assert line.delivered_minutes == 240
    assert line.shortfall_minutes == 0
    assert line.school_confirmed_minutes == 240
    assert line.parent_observed_minutes == 0
    assert line.undocumented_minutes == 0
    assert result.total_shortfall_minutes == 0
    assert result.events_considered == 8
    assert len(line.evidence) == 8


def test_partial_delivery_splits_the_shortfall_into_documented_and_undocumented():
    events = [
        held(date(2026, 10, 5)),
        held(date(2026, 10, 7)),
        held(date(2026, 10, 12)),
        held(date(2026, 10, 14)),
        held(date(2026, 10, 19)),
        missed(date(2026, 10, 21)),
    ]
    result = reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END)
    line = only(result)

    # 8 sessions owed = 240 min. 5 held x 30 = 150 delivered. 240 - 150 = 90 short.
    # 6 of the 8 promised sessions have a district record (5 held + 1 missed);
    # the remaining 2 x 30 = 60 min have no record either way.
    assert line.owed_minutes == 240
    assert line.delivered_minutes == 150
    assert line.shortfall_minutes == 90
    assert line.undocumented_minutes == 60
    assert line.school_confirmed_minutes == 150
    assert len(line.evidence) == 6


def test_a_window_with_no_evidence_is_entirely_undocumented():
    # Explicit and deliberate: shortfall_minutes IS owed_minutes here, because
    # shortfall is owed minus delivered and nothing was delivered *on the
    # record*. That subtraction is not a finding of non-delivery — which is
    # exactly why the same 240 minutes are also reported as undocumented.
    result = reconcile(ledger(obligation()), [], WINDOW_START, WINDOW_END)
    line = only(result)

    assert line.owed_minutes == 240
    assert line.delivered_minutes == 0
    assert line.shortfall_minutes == 240
    assert line.undocumented_minutes == 240
    assert line.school_confirmed_minutes == 0
    assert line.parent_observed_minutes == 0
    assert line.evidence == []
    assert result.events_considered == 0


def test_documented_misses_are_not_undocumented():
    # All 8 promised sessions are accounted for by the district: 4 held, 4
    # recorded as missed. The 120-minute shortfall is fully evidenced, so
    # nothing is undocumented.
    events = [
        held(date(2026, 10, 5)), held(date(2026, 10, 7)),
        held(date(2026, 10, 12)), held(date(2026, 10, 14)),
        missed(date(2026, 10, 19)), missed(date(2026, 10, 21)),
        missed(date(2026, 10, 26)), missed(date(2026, 10, 28)),
    ]
    line = only(reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END))

    assert line.owed_minutes == 240
    assert line.delivered_minutes == 120
    assert line.shortfall_minutes == 120
    assert line.undocumented_minutes == 0


def test_short_sessions_are_a_documented_gap_not_an_undocumented_one():
    # All 8 sessions held but each ran 20 minutes instead of 30.
    # Delivered = 8 x 20 = 160; shortfall = 240 - 160 = 80; every slot has a
    # district record, so none of that 80 is undocumented.
    line = only(
        reconcile(ledger(obligation()), [held(d, minutes=20) for d in SESSION_DAYS], WINDOW_START, WINDOW_END)
    )

    assert line.delivered_minutes == 160
    assert line.shortfall_minutes == 80
    assert line.undocumented_minutes == 0


def test_over_delivery_absorbs_the_undocumented_slots():
    # Four double-length sessions cover the full 240 promised minutes even
    # though only 4 of 8 slots have a record. Nothing is owed, so nothing is
    # undocumented: undocumented can never exceed the shortfall.
    days = [date(2026, 10, 5), date(2026, 10, 12), date(2026, 10, 19), date(2026, 10, 26)]
    line = only(
        reconcile(ledger(obligation()), [held(d, minutes=60) for d in days], WINDOW_START, WINDOW_END)
    )

    assert line.delivered_minutes == 240
    assert line.shortfall_minutes == 0
    assert line.undocumented_minutes == 0


# --------------------------------------------------------------------------
# Provenance — which records may retire a promised session
# --------------------------------------------------------------------------


def test_a_parent_observed_miss_leaves_the_slot_undocumented():
    # A parent's own note that eight sessions were missed is an allegation, not
    # a district record. Owed 240, delivered 0, shortfall 240 — and all 240 is
    # still undocumented, because the district has produced nothing either way.
    # The family's own unverified observation must never make the accusation
    # look better documented than it is.
    events = [missed(d, provenance=Provenance.PARENT_OBSERVED, source=f"parent-log-{i}")
              for i, d in enumerate(SESSION_DAYS)]
    line = only(reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END))

    assert line.owed_minutes == 240
    assert line.delivered_minutes == 0
    assert line.shortfall_minutes == 240
    assert line.undocumented_minutes == 240
    assert len(line.evidence) == 8


def test_a_school_confirmed_miss_and_a_parent_observed_miss_are_not_the_same_fact():
    # Identical arithmetic — 240 owed, 0 delivered, 240 short — but only the
    # district's own account of the eight sessions documents them.
    by_school = only(reconcile(ledger(obligation()), [missed(d) for d in SESSION_DAYS], WINDOW_START, WINDOW_END))
    by_parent = only(
        reconcile(
            ledger(obligation()),
            [missed(d, provenance=Provenance.PARENT_OBSERVED, source=f"parent-log-{i}")
             for i, d in enumerate(SESSION_DAYS)],
            WINDOW_START,
            WINDOW_END,
        )
    )

    assert by_school.shortfall_minutes == by_parent.shortfall_minutes == 240
    assert by_school.undocumented_minutes == 0
    assert by_parent.undocumented_minutes == 240


def test_delivered_minutes_split_by_evidence_grade():
    events = [
        held(date(2026, 10, 5)),
        held(date(2026, 10, 7)),
        held(date(2026, 10, 12)),
        held(date(2026, 10, 14)),
        held(date(2026, 10, 19), provenance=Provenance.PARENT_OBSERVED, source="parent-log-1"),
        held(date(2026, 10, 21), provenance=Provenance.PARENT_OBSERVED, source="parent-log-2"),
        missed(date(2026, 10, 26), provenance=Provenance.PARENT_OBSERVED, source="parent-log-3"),
    ]
    line = only(reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END))

    # 6 held x 30 = 180 delivered: 4 school-confirmed (120), 2 parent-observed
    # (60). Shortfall is 240 - 180 = 60.
    # Only the 4 district records retire a slot, so 4 of the 8 promised
    # sessions are undocumented at 30 min = 120 — capped at the 60-minute
    # shortfall. The parent's own three notes shrink nothing.
    assert line.delivered_minutes == 180
    assert line.school_confirmed_minutes == 120
    assert line.parent_observed_minutes == 60
    assert line.shortfall_minutes == 60
    assert line.undocumented_minutes == 60


def test_parent_observed_delivery_still_reduces_the_shortfall():
    # A parent-observed delivery is evidence of minutes received, so it counts
    # in delivered_minutes and shrinks the shortfall. It just does not make the
    # remaining gap look documented.
    events = [held(d, provenance=Provenance.PARENT_OBSERVED, source=f"parent-log-{i}")
              for i, d in enumerate(SESSION_DAYS[:4])]
    line = only(reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END))

    # 4 x 30 = 120 delivered, all parent-observed; shortfall 240 - 120 = 120;
    # no district record exists, so all 8 slots are undocumented at 240,
    # capped at the 120-minute shortfall.
    assert line.delivered_minutes == 120
    assert line.parent_observed_minutes == 120
    assert line.school_confirmed_minutes == 0
    assert line.shortfall_minutes == 120
    assert line.undocumented_minutes == 120


# --------------------------------------------------------------------------
# Documented silence — the discovery.py contract
# --------------------------------------------------------------------------


def test_documented_silence_does_not_retire_a_promised_session():
    # Four overdue, unanswered records requests. Silence documents the absence
    # of a record, not the disposition of a session, so exercising the parent's
    # records right must not shrink the undocumented bucket.
    events = [silence(d) for d in SESSION_DAYS[:4]]
    line = only(reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END))

    assert line.owed_minutes == 240
    assert line.delivered_minutes == 0
    assert line.shortfall_minutes == 240
    assert line.undocumented_minutes == 240
    assert len(line.evidence) == 4


def test_a_silence_record_speaks_about_the_record_never_about_the_session():
    line = only(reconcile(ledger(obligation()), [silence(date(2026, 10, 5))], WINDOW_START, WINDOW_END))
    ref = line.evidence[0]

    assert ref.provenance is Provenance.DOCUMENTED_SILENCE
    assert "2026-10-05" in ref.detail
    assert "no Speech-Language Therapy service record was produced by the district" in ref.detail
    assert "recorded as not delivered" not in ref.detail


def test_a_delivery_claimed_on_a_silence_record_is_disregarded():
    # Contradictory input: silence is the absence of a district record, so it
    # can never confirm a delivery. The claim is not credited to the district,
    # and the slot stays undocumented.
    claimed = ServiceEvent(
        event_date=date(2026, 10, 5),
        service="Speech-Language Therapy",
        minutes=30,
        delivered=True,
        provenance=Provenance.DOCUMENTED_SILENCE,
        source="req-001",
    )
    line = only(reconcile(ledger(obligation()), [claimed], WINDOW_START, WINDOW_END))

    assert line.delivered_minutes == 0
    assert line.school_confirmed_minutes == 0
    assert line.parent_observed_minutes == 0
    assert line.shortfall_minutes == 240
    assert line.undocumented_minutes == 240
    assert "not counted" in line.evidence[0].detail
    assert "recorded as delivered" not in line.evidence[0].detail


# --------------------------------------------------------------------------
# Excused non-delivery — the sessions the child's own absence cost
#
# Every figure below is hand-computed against the standard window: 8 promised
# sessions of 30 minutes, 240 minutes owed. An excused session is subtracted
# from the difference and never from the promise, because a letter that
# reported a smaller "owed" could not be checked against the IEP it quotes.
# --------------------------------------------------------------------------


def test_a_session_the_child_was_absent_for_is_not_part_of_the_difference():
    # Six sessions held (180), one missed by the school, one missed because
    # Maya was out. 240 - 180 - 30 = 30 is the difference the letter may claim.
    events = [held(d) for d in SESSION_DAYS[:6]]
    events.append(missed(SESSION_DAYS[6]))
    events.append(absent(SESSION_DAYS[7]))
    line = only(reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END))

    assert line.owed_minutes == 240  # the promise is untouched
    assert line.delivered_minutes == 180
    assert line.excused_minutes == 30
    assert line.shortfall_minutes == 30
    assert line.undocumented_minutes == 0


def test_an_excused_session_is_never_counted_as_delivered():
    # All eight sessions fell on days the child was absent. Nothing was
    # delivered and nothing is short, and the two facts are not the same one.
    line = only(reconcile(ledger(obligation()), [absent(d) for d in SESSION_DAYS], WINDOW_START, WINDOW_END))

    assert line.delivered_minutes == 0
    assert line.school_confirmed_minutes == 0
    assert line.parent_observed_minutes == 0
    assert line.excused_minutes == 240
    assert line.shortfall_minutes == 0


def test_an_excused_session_keeps_its_evidence_and_says_why():
    # Excluded from the arithmetic, still on the record: the footnote is how a
    # reader sees why the difference is smaller than owed minus delivered.
    line = only(reconcile(ledger(obligation()), [absent(date(2026, 10, 14))], WINDOW_START, WINDOW_END))

    assert len(line.evidence) == 1
    assert "2026-10-14" in line.evidence[0].detail
    assert ABSENCE_NOTE in line.evidence[0].detail


def test_an_ordinary_miss_is_not_footnoted_as_an_absence():
    line = only(reconcile(ledger(obligation()), [missed(date(2026, 10, 14))], WINDOW_START, WINDOW_END))

    assert ABSENCE_NOTE not in line.evidence[0].detail
    assert line.excused_minutes == 0


def test_one_absence_recorded_twice_excuses_one_session():
    # The provider's email and the parent's log describe the same absence. One
    # calendar date is one promised slot, so 30 minutes come out, not 60.
    events = [
        absent(date(2026, 10, 14), source="email-2026-10-14-01"),
        absent(date(2026, 10, 14), provenance=Provenance.PARENT_OBSERVED, source="plog-2026-10-14-01"),
    ]
    line = only(reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END))

    assert line.excused_minutes == 30
    assert line.shortfall_minutes == 210
    assert len(line.evidence) == 2


def test_a_parent_logged_absence_excuses_a_miss_the_district_left_unexplained():
    # The district records the session as not held and gives no reason; the
    # family's own log says why. An absence noted at home costs the family
    # minutes it could otherwise have claimed, so it is believed.
    events = [
        missed(date(2026, 10, 14), source="svclog-001"),
        absent(date(2026, 10, 14), provenance=Provenance.PARENT_OBSERVED, source="plog-2026-10-14-01"),
    ]
    line = only(reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END))

    assert line.excused_minutes == 30
    assert line.shortfall_minutes == 210
    # Seven slots carry no record at all; the eighth is accounted for by both.
    assert line.undocumented_minutes == 210


def test_a_date_with_a_delivery_record_is_never_also_excused():
    # Those minutes are already counted as delivered. Excusing them as well
    # would subtract one promised slot twice and understate the difference.
    events = [
        held(date(2026, 10, 14), source="svclog-001"),
        absent(date(2026, 10, 14), provenance=Provenance.PARENT_OBSERVED, source="plog-2026-10-14-01"),
    ]
    line = only(reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END))

    assert line.delivered_minutes == 30
    assert line.excused_minutes == 0
    assert line.shortfall_minutes == 210


def test_excused_minutes_cannot_exceed_what_the_promise_has_left():
    # Four double-length sessions already cover all 240 promised minutes. An
    # absence on a fifth date excuses nothing, because there is nothing left to
    # excuse and a bucket may never manufacture minutes to fill itself.
    days = [date(2026, 10, 5), date(2026, 10, 12), date(2026, 10, 19), date(2026, 10, 26)]
    events = [held(d, minutes=60) for d in days] + [absent(date(2026, 10, 28))]
    line = only(reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END))

    assert line.delivered_minutes == 240
    assert line.excused_minutes == 0
    assert line.shortfall_minutes == 0


def test_an_excused_session_is_never_also_undocumented():
    # The one date anybody wrote anything about is the absence, and it is
    # already out of the difference. Counting it as "no record either way"
    # would put one promised slot in two buckets at once.
    events = [absent(date(2026, 10, 14), provenance=Provenance.PARENT_OBSERVED, source="plog-1")]
    line = only(reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END))

    assert line.excused_minutes == 30
    assert line.shortfall_minutes == 210
    assert line.undocumented_minutes == 210  # the other seven slots, not the eighth


# --------------------------------------------------------------------------
# Several records for one date
# --------------------------------------------------------------------------


def test_one_session_recorded_twice_is_counted_once():
    # The expected state once a records request is answered: the district's
    # service log and the parent's log both describe the same 30-minute
    # session. Delivered is 30, not 60, and one slot is retired, not two.
    events = [
        held(date(2026, 10, 5), source="district-log-1"),
        held(date(2026, 10, 5), provenance=Provenance.PARENT_OBSERVED, source="parent-log-1"),
    ]
    result = reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END)
    line = only(result)

    assert line.delivered_minutes == 30
    assert line.school_confirmed_minutes == 30
    assert line.parent_observed_minutes == 0
    assert line.shortfall_minutes == 210
    # 7 of 8 slots carry no district record: 7 x 30 = 210.
    assert line.undocumented_minutes == 210
    # Both records are still footnotable.
    assert len(line.evidence) == 2
    assert result.events_considered == 2


def test_a_contradiction_on_one_date_is_resolved_by_the_district_record():
    # The district says the Oct 5 session was held; the parent says it was
    # missed. The district's own record decides delivery, and the date is one
    # slot however many records describe it.
    events = [
        held(date(2026, 10, 5), source="district-log-1"),
        missed(date(2026, 10, 5), provenance=Provenance.PARENT_OBSERVED, source="parent-log-1"),
    ]
    line = only(reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END))

    assert line.delivered_minutes == 30
    assert line.shortfall_minutes == 210
    assert line.undocumented_minutes == 210


def test_a_district_miss_outranks_a_parent_delivery_on_the_same_date():
    # The other direction: the district records Oct 5 as missed while the
    # parent remembers a session. Delivered is 0, the slot is documented, and
    # the remaining 7 x 30 = 210 minutes are undocumented.
    events = [
        missed(date(2026, 10, 5), source="district-log-1"),
        held(date(2026, 10, 5), provenance=Provenance.PARENT_OBSERVED, source="parent-log-1"),
    ]
    line = only(reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END))

    assert line.delivered_minutes == 0
    assert line.shortfall_minutes == 240
    assert line.undocumented_minutes == 210


def test_two_genuine_district_sessions_on_one_date_both_count():
    # Two separate district log lines for Oct 5, 30 minutes each: both are the
    # district's own record, so both minutes count. They still retire one slot,
    # which is the conservative direction.
    events = [
        held(date(2026, 10, 5), source="district-log-1"),
        held(date(2026, 10, 5), source="district-log-2"),
    ]
    line = only(reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END))

    assert line.delivered_minutes == 60
    assert line.school_confirmed_minutes == 60
    assert line.shortfall_minutes == 180
    # 7 slots without a district record = 210, capped at the 180 shortfall.
    assert line.undocumented_minutes == 180


# --------------------------------------------------------------------------
# Evidence refs
# --------------------------------------------------------------------------


def test_evidence_refs_carry_provenance_source_and_a_dated_fact():
    events = [
        held(date(2026, 10, 5), source="district-log-88"),
        missed(date(2026, 10, 7), provenance=Provenance.PARENT_OBSERVED, source="parent-log-4"),
    ]
    line = only(reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END))

    assert [ref.source for ref in line.evidence] == ["district-log-88", "parent-log-4"]

    delivered_ref, missed_ref = line.evidence
    assert delivered_ref.provenance is Provenance.SCHOOL_CONFIRMED
    assert "2026-10-05" in delivered_ref.detail
    assert "30 minutes" in delivered_ref.detail
    assert "recorded as delivered" in delivered_ref.detail

    assert missed_ref.provenance is Provenance.PARENT_OBSERVED
    assert "2026-10-07" in missed_ref.detail
    assert "recorded as not delivered" in missed_ref.detail


def test_evidence_is_ordered_by_date():
    events = [
        held(date(2026, 10, 21)),
        held(date(2026, 10, 5)),
        missed(date(2026, 10, 14)),
    ]
    line = only(reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END))

    assert [ref.detail[:10] for ref in line.evidence] == ["2026-10-05", "2026-10-14", "2026-10-21"]


# --------------------------------------------------------------------------
# Service-name matching
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "written",
    [
        "Speech-Language Therapy",
        "speech language therapy",
        "  SPEECH   THERAPY  ",
        "Speech/Language Therapy",
        "Speech Therapy",
        "Speech-Language Services",
        "SLP",
        # Spellings a district export or an IEP writes routinely. Each one is a
        # fully phantom 180-minute shortfall if it fails to fold.
        "Speech-Language Therapy (Direct)",
        "speech-language therapy (direct)",
        "Speech and Language Pathology",
        "Individual Speech Therapy",
        "Speech Therapy - Individual",
        "Group Speech Therapy",
        "Push-In Speech Therapy",
        "Speech Therapy - Pull Out",
        "Speech Therapy (Push-In)",
    ],
)
def test_service_aliases_reconcile_against_the_same_obligation(written):
    line = only(
        reconcile(
            ledger(obligation()),
            [held(date(2026, 10, 5), service=written)],
            WINDOW_START,
            WINDOW_END,
        )
    )
    assert line.delivered_minutes == 30


def test_a_parenthetical_abbreviation_does_not_manufacture_a_shortfall():
    # 'Occupational Therapy (OT)' is the normal way a district logs the
    # service. Before the parenthetical was stripped this produced 180 owed,
    # 0 delivered and a 180-minute shortfall out of thin air.
    ot = obligation(
        service="Occupational Therapy",
        minutes_per_session=45,
        sessions_per_period=1,
        period=Period.WEEK,
    )
    events = [held(date(2026, 10, 6), minutes=45, service="Occupational Therapy (OT)", source="ot-log-1")]
    result = reconcile(ledger(ot), events, WINDOW_START, WINDOW_END)
    line = only(result)

    # 20 school days x 1 // 5 = 4 sessions x 45 = 180 owed, 45 delivered.
    assert line.owed_minutes == 180
    assert line.delivered_minutes == 45
    assert line.shortfall_minutes == 135
    # 3 of 4 slots carry no district record: 3 x 45 = 135.
    assert line.undocumented_minutes == 135
    assert unmatched_events(ledger(ot), events, WINDOW_START, WINDOW_END) == []


def test_canonical_service_folds_known_spellings_and_leaves_unknowns_alone():
    assert canonical_service("Speech-Language Therapy") == canonical_service("Speech Therapy")
    assert canonical_service("Occupational Therapy") == canonical_service("OT")
    assert canonical_service("Occupational Therapy") == canonical_service("Occupational Therapy (OT)")
    assert canonical_service("Physical Therapy") == canonical_service("Physical Therapy (PT)")
    assert canonical_service("Individual Counseling") == canonical_service("counseling")
    assert canonical_service("Specialized Academic Instruction") == canonical_service("SAI")
    assert canonical_service("Occupational Therapy") != canonical_service("Physical Therapy")
    # An unrecognized name is its own key rather than a guess at a neighbour.
    assert canonical_service("Music Therapy") == "music therapy"
    assert canonical_service("Music Therapy") != canonical_service("Speech Therapy")
    # 'in' and 'out' are dropped only as the tail of a delivery model, never
    # from a service that happens to contain the word.
    assert canonical_service("Push-In Music Therapy") == "music therapy"
    assert canonical_service("Music In Classroom") == "music in classroom"


def test_events_for_an_unpromised_service_are_counted_not_dropped():
    events = [
        held(date(2026, 10, 5)),
        held(date(2026, 10, 7), service="Physical Therapy", source="pt-log-1"),
        held(date(2026, 10, 12), service="Music Therapy", source="music-log-1"),
    ]
    result = reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END)
    line = only(result)

    assert result.events_considered == 3
    assert line.delivered_minutes == 30

    stray = unmatched_events(ledger(obligation()), events, WINDOW_START, WINDOW_END)
    assert [event.source for event in stray] == ["pt-log-1", "music-log-1"]


def test_the_unmatched_count_is_recoverable_from_the_result_alone():
    # ReconciliationResult has no unmatched field, so this subtraction is how a
    # caller holding only the result learns that in-window records bound to
    # nothing and the row it is about to send may overstate the shortfall.
    events = [
        held(date(2026, 10, 5)),
        held(date(2026, 10, 7), service="Physical Therapy", source="pt-log-1"),
        held(date(2026, 10, 12), service="Music Therapy", source="music-log-1"),
    ]
    result = reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END)

    cited = sum(len(line.evidence) for line in result.shortfalls)
    assert result.events_considered == 3
    assert cited == 1
    assert result.events_considered - cited == 2
    assert len(unmatched_events(ledger(obligation()), events, WINDOW_START, WINDOW_END)) == 2


def test_an_event_outside_the_obligation_dates_binds_to_nothing():
    ended = obligation(end_date=date(2026, 10, 16))
    events = [held(date(2026, 10, 7)), held(date(2026, 10, 26), source="late-log")]

    result = reconcile(ledger(ended), events, WINDOW_START, WINDOW_END)
    line = only(result)

    # Oct 5-16 = 10 school days -> 10 x 2 // 5 = 4 sessions = 120 min owed.
    # Only the Oct 7 session falls inside the obligation.
    assert line.owed_minutes == 120
    assert line.delivered_minutes == 30
    assert line.period_start == date(2026, 10, 5)
    assert line.period_end == date(2026, 10, 16)
    assert result.events_considered == 2

    stray = unmatched_events(ledger(ended), events, WINDOW_START, WINDOW_END)
    assert [event.source for event in stray] == ["late-log"]
    assert result.events_considered - len(line.evidence) == len(stray)


def test_events_outside_the_window_are_ignored_entirely():
    events = [
        held(date(2026, 9, 14), source="september-log"),
        held(date(2026, 10, 5)),
        held(date(2026, 11, 9), source="november-log"),
    ]
    result = reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END)

    assert result.events_considered == 1
    assert only(result).delivered_minutes == 30
    assert unmatched_events(ledger(obligation()), events, WINDOW_START, WINDOW_END) == []


# --------------------------------------------------------------------------
# Windows, multiple services, and degenerate input
# --------------------------------------------------------------------------


def test_partial_overlap_owes_only_the_clipped_period():
    # Requested Sep 1 .. Sep 30 (22 school days) but the service starts Sep 8,
    # leaving 17 school days -> 17 x 2 // 5 = 6 sessions = 180 min.
    line = only(reconcile(ledger(obligation()), [], date(2026, 9, 1), date(2026, 9, 30)))

    assert line.owed_minutes == 180
    assert line.period_start == date(2026, 9, 8)
    assert line.period_end == date(2026, 9, 30)


def test_a_service_not_in_force_reports_a_zero_line_over_the_requested_window():
    future = obligation(start_date=date(2027, 1, 4), end_date=date(2027, 6, 11))
    line = only(reconcile(ledger(future), [], WINDOW_START, WINDOW_END))

    assert line.owed_minutes == 0
    assert line.shortfall_minutes == 0
    assert line.undocumented_minutes == 0
    assert line.period_start == WINDOW_START
    assert line.period_end == WINDOW_END


def test_each_service_reconciles_independently():
    speech = obligation()
    ot = obligation(
        service="Occupational Therapy",
        minutes_per_session=45,
        sessions_per_period=1,
        period=Period.WEEK,
    )
    daily = obligation(
        service="Specialized Academic Instruction",
        minutes_per_session=60,
        sessions_per_period=1,
        period=Period.DAY,
    )
    events = [
        held(date(2026, 10, 5)),
        held(date(2026, 10, 7)),
        held(date(2026, 10, 6), minutes=45, service="OT", source="ot-log-1"),
    ]
    result = reconcile(ledger(speech, ot, daily), events, WINDOW_START, WINDOW_END)

    assert [line.service for line in result.shortfalls] == [
        "Speech-Language Therapy",
        "Occupational Therapy",
        "Specialized Academic Instruction",
    ]
    speech_line, ot_line, daily_line = result.shortfalls

    # Speech: 8 x 30 = 240 owed, 60 delivered, 180 short, 6 slots unrecorded = 180.
    assert (speech_line.owed_minutes, speech_line.delivered_minutes) == (240, 60)
    assert speech_line.shortfall_minutes == 180
    assert speech_line.undocumented_minutes == 180

    # OT: 20 school days x 1 // 5 = 4 sessions x 45 = 180 owed, 45 delivered.
    assert (ot_line.owed_minutes, ot_line.delivered_minutes) == (180, 45)
    assert ot_line.shortfall_minutes == 135
    assert ot_line.undocumented_minutes == 135

    # SAI: 20 school days x 60 = 1200 owed, nothing recorded.
    assert (daily_line.owed_minutes, daily_line.delivered_minutes) == (1200, 0)
    assert daily_line.undocumented_minutes == 1200

    # 240 + 180 + 1200 owed; 180 + 135 + 1200 short.
    assert result.total_shortfall_minutes == 1515


def test_two_ledger_lines_for_one_service_reconcile_as_a_single_row():
    # An amendment mid-window: 30 min sessions through Oct 16, 45 min after.
    original = obligation(start_date=date(2026, 9, 8), end_date=date(2026, 10, 16))
    amended = obligation(
        minutes_per_session=45,
        start_date=date(2026, 10, 19),
        end_date=date(2027, 6, 11),
        source_quote="Amended: 45 minutes per session, 2 sessions per week.",
    )
    events = [held(date(2026, 10, 5)), held(date(2026, 10, 7))]

    result = reconcile(ledger(original, amended), events, WINDOW_START, WINDOW_END)
    line = only(result)

    # Oct 5-16 = 10 school days -> 4 sessions x 30 = 120.
    # Oct 19-30 = 10 school days -> 4 sessions x 45 = 180.
    # 8 sessions, 300 min owed. 2 slots recorded, 60 min delivered.
    # 6 unrecorded slots priced at the blended 300/8 = 37.5 -> 225.
    assert line.owed_minutes == 300
    assert line.delivered_minutes == 60
    assert line.shortfall_minutes == 240
    assert line.undocumented_minutes == 225
    assert line.period_start == date(2026, 10, 5)
    assert line.period_end == date(2026, 10, 30)


def test_a_blended_rate_rounds_undocumented_minutes_up():
    # Same amendment, one recorded slot: 7 unrecorded x 37.5 = 262.5 minutes.
    # That rounds to 263, not 262 — a fraction of a minute belongs in "we
    # cannot tell", never in the shortfall stated without a hedge.
    original = obligation(start_date=date(2026, 9, 8), end_date=date(2026, 10, 16))
    amended = obligation(
        minutes_per_session=45,
        start_date=date(2026, 10, 19),
        end_date=date(2027, 6, 11),
        source_quote="Amended: 45 minutes per session, 2 sessions per week.",
    )
    line = only(
        reconcile(ledger(original, amended), [held(date(2026, 10, 5))], WINDOW_START, WINDOW_END)
    )

    assert line.owed_minutes == 300
    assert line.delivered_minutes == 30
    assert line.shortfall_minutes == 270
    assert line.undocumented_minutes == 263


def test_empty_ledger_and_empty_events():
    result = reconcile(ledger(), [], WINDOW_START, WINDOW_END)

    assert result.shortfalls == []
    assert result.events_considered == 0
    assert result.total_shortfall_minutes == 0
    assert result.period_start == WINDOW_START
    assert result.period_end == WINDOW_END


def test_events_with_no_obligations_at_all_are_all_unmatched():
    events = [held(date(2026, 10, 5)), held(date(2026, 10, 7))]
    result = reconcile(ledger(), events, WINDOW_START, WINDOW_END)

    assert result.shortfalls == []
    assert result.events_considered == 2
    assert len(unmatched_events(ledger(), events, WINDOW_START, WINDOW_END)) == 2


def test_a_single_day_window():
    # Mon 2026-10-05 alone: 1 school day, 1 x 2 // 5 = 0 sessions owed.
    result = reconcile(ledger(obligation()), [held(date(2026, 10, 5))], WINDOW_START, WINDOW_START)
    line = only(result)

    assert line.owed_minutes == 0
    assert line.delivered_minutes == 30
    assert line.shortfall_minutes == 0
    assert line.undocumented_minutes == 0


def test_reconcile_rejects_a_reversed_window():
    with pytest.raises(ValueError):
        reconcile(ledger(obligation()), [], WINDOW_END, WINDOW_START)


def test_unmatched_events_rejects_a_reversed_window():
    with pytest.raises(ValueError):
        unmatched_events(ledger(obligation()), [], WINDOW_END, WINDOW_START)


# --------------------------------------------------------------------------
# The reported invariant, over every provenance
# --------------------------------------------------------------------------


# Each case is (events, owed, delivered, shortfall, undocumented, school, parent),
# every figure hand-computed against the standard window: 8 promised sessions,
# 30 minutes each, 240 minutes owed. Slots are retired only by SCHOOL_CONFIRMED
# records, so the undocumented figure is 30 x (8 - distinct district dates),
# capped at the shortfall.
@pytest.mark.parametrize(
    "events, owed, delivered, shortfall, undocumented, school, parent",
    [
        ([], 240, 0, 240, 240, 0, 0),
        ([held(date(2026, 10, 5))], 240, 30, 210, 210, 30, 0),
        ([held(date(2026, 10, 5)), missed(date(2026, 10, 7))], 240, 30, 210, 180, 30, 0),
        (
            [held(d, minutes=90) for d in (date(2026, 10, 5), date(2026, 10, 7), date(2026, 10, 12))],
            240, 270, 0, 0, 270, 0,
        ),
        (
            [held(date(2026, 10, 5), provenance=Provenance.PARENT_OBSERVED)],
            240, 30, 210, 210, 0, 30,
        ),
        ([silence(date(2026, 10, 5))], 240, 0, 240, 240, 0, 0),
        (
            [ServiceEvent(
                event_date=date(2026, 10, 5),
                service="Speech-Language Therapy",
                minutes=30,
                delivered=True,
                provenance=Provenance.DOCUMENTED_SILENCE,
                source="req-002",
            )],
            240, 0, 240, 240, 0, 0,
        ),
    ],
    ids=[
        "no-evidence",
        "one-district-delivery",
        "delivery-and-district-miss",
        "over-delivery",
        "parent-observed-delivery",
        "documented-silence",
        "silence-claiming-delivery",
    ],
)
def test_the_reported_invariant_holds(events, owed, delivered, shortfall, undocumented, school, parent):
    line = only(reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END))

    assert line.owed_minutes == owed
    assert line.delivered_minutes == delivered
    assert line.shortfall_minutes == shortfall
    assert line.undocumented_minutes == undocumented
    assert line.school_confirmed_minutes == school
    assert line.parent_observed_minutes == parent

    # The guards the module publishes, restated over the same cases.
    assert 0 <= line.undocumented_minutes <= line.shortfall_minutes <= line.owed_minutes
    assert line.school_confirmed_minutes + line.parent_observed_minutes == line.delivered_minutes


# --------------------------------------------------------------------------
# The bucket invariant
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "events, delivered, excused, undocumented, shortfall",
    [
        # Nothing on file: the whole promise is short and none of it is
        # documented either way.
        ([], 0, 0, 240, 240),
        # Eight sessions held.
        ([held(d) for d in SESSION_DAYS], 240, 0, 0, 0),
        # Eight sessions the district's own log records as not held: short in
        # full, and documented in full.
        ([missed(d) for d in SESSION_DAYS], 0, 0, 0, 240),
        # Eight sessions the child was away for: nothing delivered, nothing
        # short, and the two facts are not the same one.
        ([absent(d) for d in SESSION_DAYS], 0, 240, 0, 0),
        # Four held (120), four excused (4 x 30 = 120).
        (
            [held(d) for d in SESSION_DAYS[:4]] + [absent(d) for d in SESSION_DAYS[4:]],
            120,
            120,
            0,
            0,
        ),
        # One of each grade. 30 delivered, 30 excused, so 180 short. Three
        # slots are accounted for (two district records plus the excused
        # session; the silence accounts for nothing by contract), leaving five
        # of the eight at 30 minutes = 150 undocumented, and 30 of the
        # shortfall evidenced by the district's own missed-session record.
        (
            [held(SESSION_DAYS[0]), missed(SESSION_DAYS[1]), absent(SESSION_DAYS[2]),
             silence(SESSION_DAYS[3])],
            30,
            30,
            150,
            180,
        ),
        # The district records the miss and gives no reason; the family's log
        # says why. One slot, one excused session, seven slots unrecorded.
        (
            [absent(SESSION_DAYS[0], provenance=Provenance.PARENT_OBSERVED, source="plog-1"),
             missed(SESSION_DAYS[0], source="svclog-1")],
            0,
            30,
            210,
            210,
        ),
        # A 60-minute session held on the same date the family logged an
        # absence. Those minutes are already delivered, so excusing them too
        # would subtract one slot twice. One slot is documented, the other
        # seven are worth 210 and the shortfall caps them at 180.
        (
            [held(SESSION_DAYS[0], minutes=60),
             absent(SESSION_DAYS[0], provenance=Provenance.PARENT_OBSERVED, source="plog-1")],
            60,
            0,
            180,
            180,
        ),
        # Over-delivery: 270 minutes across three long sessions covers the
        # whole 240 promise, so the absence on a fourth date has nothing left
        # to excuse.
        ([held(d, minutes=90) for d in SESSION_DAYS[:3]] + [absent(SESSION_DAYS[3])], 270, 0, 0, 0),
    ],
    ids=[
        "nothing",
        "everything-delivered",
        "everything-missed",
        "everything-excused",
        "half-delivered-half-excused",
        "one-of-each-grade",
        "parent-absence-over-district-miss",
        "delivery-and-absence-on-one-date",
        "over-delivery-with-an-absence",
    ],
)
def test_the_four_buckets_partition_the_promise(events, delivered, excused, undocumented, shortfall):
    """No minute is in two buckets, and none goes missing between them.

    Delivered, excused, undocumented and evidenced-short are the four things a
    promised minute can be, and the letter, the Statement and the parent all
    read the difference as owed minus the first two. If they did not add up,
    some minutes would be silently dropped or silently counted twice --- and
    counted twice, in the only direction that matters, means asked for twice.

    Every figure is a hand-computed literal, and that is the point. The
    addition ``delivered + excused + undocumented + (shortfall - undocumented)``
    is an algebraic identity over whatever this module returns: it holds just
    as well when a date is double-excused, when five absence dates excuse five
    sessions of a weekly service, or when an excused session is priced at ten
    times its rate. An invariant that cannot fail is not an invariant, so the
    identity is checked *underneath* five pinned numbers rather than instead of
    them, and two bounds the identity cannot absorb are checked beside it: an
    excused session is never worth more than a session, and there are never
    more of them than there are dates a record notes an absence on.

    Over-delivery is the one case where the sum exceeds the promise rather
    than equalling it: minutes delivered beyond what was owed are still real,
    the shortfall is floored at zero, and nothing is short.
    """
    line = only(reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END))

    assert line.owed_minutes == SPEECH_OWED  # never reduced by any of the others
    assert line.delivered_minutes == delivered
    assert line.excused_minutes == excused
    assert line.undocumented_minutes == undocumented
    assert line.shortfall_minutes == shortfall

    # Slot-level bounds. Neither follows from the arithmetic above, which is
    # why they are the two that catch an over-large excused bucket.
    minutes_per_session = SPEECH_OWED // SPEECH_SESSIONS
    absence_dates = {e.event_date for e in events if e.attribution is Attribution.STUDENT_ABSENCE}
    assert line.excused_minutes <= len(absence_dates) * minutes_per_session
    assert line.excused_minutes <= SPEECH_SESSIONS * minutes_per_session

    evidenced = line.shortfall_minutes - line.undocumented_minutes
    assert evidenced >= 0
    buckets = (
        line.delivered_minutes + line.excused_minutes + line.undocumented_minutes + evidenced
    )
    assert buckets == max(line.owed_minutes, line.delivered_minutes + line.excused_minutes)


@pytest.mark.parametrize(
    "events",
    [
        [absent(d) for d in SESSION_DAYS],
        [held(d) for d in SESSION_DAYS[:4]] + [absent(d) for d in SESSION_DAYS[4:]],
        [absent(SESSION_DAYS[0], provenance=Provenance.PARENT_OBSERVED, source="plog-1"),
         missed(SESSION_DAYS[0], source="svclog-1")],
        [held(SESSION_DAYS[0], minutes=60), absent(SESSION_DAYS[1])],
    ],
    ids=["all", "half", "parent-log-only", "one-of-eight"],
)
def test_excused_minutes_never_arrive_without_the_record_that_excused_them(events):
    """The contract ``letters.py`` rests on when it footnotes the exclusion.

    A line reporting excused minutes with no absence record on it would leave
    the letter with an uncitable claim about the child --- and the letter drops
    that claim rather than assert it, so the exclusion would go unexplained
    beside a difference smaller than owed minus delivered. Excused minutes come
    from matched absence events, and every matched event becomes a footnote, so
    the two cannot come apart here.
    """
    line = only(reconcile(ledger(obligation()), events, WINDOW_START, WINDOW_END))

    assert line.excused_minutes > 0
    assert [ref for ref in line.evidence if ABSENCE_NOTE in ref.detail]


# --------------------------------------------------------------------------
# What an absence is allowed to be worth
# --------------------------------------------------------------------------


def test_a_week_of_absences_excuses_only_the_sessions_that_week_held():
    """Five absence dates against a once-a-week service excuse ONE session.

    The child is out sick for a week and the parent logs all five school days.
    Pricing each date at a full session would excuse 150 minutes of a promise
    that only ever put one session inside that week --- erasing the three
    sessions owed in the weeks either side of it. Erasing a shortfall is the
    same failure as inventing one, pointed the other way.
    """
    weekly = obligation(sessions_per_period=1)
    week = [date(2026, 10, 12), date(2026, 10, 13), date(2026, 10, 14),
            date(2026, 10, 15), date(2026, 10, 16)]
    events = [
        absent(day, provenance=Provenance.PARENT_OBSERVED, source=f"plog-{day.isoformat()}")
        for day in week
    ]
    line = only(reconcile(ledger(weekly), events, WINDOW_START, WINDOW_END))

    assert line.owed_minutes == 120  # 20 school days x 1 per week = 4 sessions
    assert line.excused_minutes == 30
    assert line.shortfall_minutes == 90
    # The three sessions in the weeks either side are still slots nobody
    # recorded anything about --- the absence week does not document them.
    assert line.undocumented_minutes == 90


def test_a_five_day_absence_of_a_daily_service_excuses_all_five():
    """The same rule the other way: a daily service really did lose five."""
    daily = obligation(minutes_per_session=60, sessions_per_period=1, period=Period.DAY)
    week = [date(2026, 10, 12), date(2026, 10, 13), date(2026, 10, 14),
            date(2026, 10, 15), date(2026, 10, 16)]
    line = only(reconcile(ledger(daily), [absent(day) for day in week], WINDOW_START, WINDOW_END))

    assert line.owed_minutes == 1200  # 20 school days x 60 minutes
    assert line.excused_minutes == 300
    assert line.shortfall_minutes == 900


def test_one_absence_date_is_worth_at_most_one_session():
    daily = obligation(minutes_per_session=60, sessions_per_period=1, period=Period.DAY)
    line = only(reconcile(ledger(daily), [absent(date(2026, 10, 14))], WINDOW_START, WINDOW_END))

    assert line.excused_minutes == 60
    assert line.shortfall_minutes == 1140


def test_an_absence_is_priced_at_the_obligation_in_force_that_day():
    """Two IEP lines for one service, of different lengths.

    A mid-year amendment leaves 30-minute sessions in force until 2026-10-20
    and 90-minute sessions after it. Pricing an absence at the group's blended
    average (150 owed over 3 sessions = 50) would excuse 50 minutes for a day
    on which only the 30-minute line was in force.
    """
    before = obligation(
        service="Speech Therapy",
        minutes_per_session=30,
        sessions_per_period=1,
        start_date=date(2026, 9, 8),
        end_date=date(2026, 10, 20),
    )
    after = obligation(
        minutes_per_session=90,
        sessions_per_period=1,
        start_date=date(2026, 10, 21),
        end_date=date(2027, 6, 11),
    )
    line = only(
        reconcile(ledger(before, after), [absent(date(2026, 10, 14))], WINDOW_START, WINDOW_END)
    )

    assert line.owed_minutes == 150  # 2 x 30 before the amendment, 1 x 90 after
    assert line.excused_minutes == 30
    assert line.shortfall_minutes == 120


def test_a_stretch_with_no_provider_is_never_excused_by_anything():
    """The shortfall this whole mechanism must not be able to erase."""
    line = only(reconcile(ledger(obligation()), [missed(d) for d in SESSION_DAYS], WINDOW_START, WINDOW_END))

    assert line.excused_minutes == 0
    assert line.shortfall_minutes == 240


# --------------------------------------------------------------------------
# The committed fixture semester
#
# Read off disk, never off the network: fixtures/cache holds the ledger the
# extractor produced and the events the classifier produced, both committed.
# Three of those events are non-deliveries the correspondence attributes to
# Maya's own absence, and they are the reason this module has an excused
# bucket at all — so they are pinned here against the arithmetic, not only
# against the classifier that found them.
# --------------------------------------------------------------------------

CACHE = Path(__file__).resolve().parents[1] / "fixtures" / "cache"
SEMESTER_START = date(2026, 9, 8)
SEMESTER_END = date(2027, 1, 29)

# Read off the fixtures by hand: the OT absence is written twice (the
# provider's email and the parent's log of the same day), the SAI absence once.
FIXTURE_ABSENCES = {
    ("email-2026-10-14-01", date(2026, 10, 14), "Occupational Therapy"),
    ("plog-2026-10-14-01", date(2026, 10, 14), "Occupational Therapy"),
    ("email-2027-01-15-01", date(2027, 1, 12), "Specialized Academic Instruction"),
}


def _fixture_semester() -> tuple[IEPLedger, list[ServiceEvent]]:
    ledger_json = (CACHE / "iep_maya_ledger.json").read_text(encoding="utf-8")
    return IEPLedger.model_validate_json(ledger_json), load_cached_events()


def test_the_three_recorded_absences_are_the_ones_excused():
    ledger_, events = _fixture_semester()

    attributed = {
        (e.source, e.event_date, e.service)
        for e in events
        if e.attribution is Attribution.STUDENT_ABSENCE
    }
    assert attributed == FIXTURE_ABSENCES

    lines = {line.service: line for line in reconcile(ledger_, events, SEMESTER_START, SEMESTER_END).shortfalls}
    ot = lines["Occupational Therapy"]
    sai = lines["Specialized Academic Instruction"]

    # OT is 45 minutes once a week and SAI 60 minutes a day, so one excused
    # session each — and the two records of 2026-10-14 excuse one OT session
    # between them, not two.
    assert (ot.owed_minutes, ot.delivered_minutes, ot.excused_minutes, ot.shortfall_minutes) == (900, 630, 45, 225)
    assert (sai.owed_minutes, sai.delivered_minutes, sai.excused_minutes, sai.shortfall_minutes) == (6240, 0, 60, 6180)
    assert lines["Speech-Language Therapy"].excused_minutes == 0
    assert lines["Individual Counseling"].excused_minutes == 0


def test_excusing_the_absences_moves_those_minutes_and_no_others():
    """The whole semester, before and after, as literals.

    7,155 minutes is what this semester reported while an absent child's
    sessions were still counted in the difference; 7,050 is what it reports
    now. The 105 between them are the OT session on 2026-10-14 and the SAI
    session on 2027-01-12, and nothing else moves — least of all what the IEP
    promised.
    """
    ledger_, events = _fixture_semester()
    unattributed = [
        e.model_copy(update={"attribution": Attribution.SCHOOL_OR_UNRECORDED}) for e in events
    ]

    before = reconcile(ledger_, unattributed, SEMESTER_START, SEMESTER_END)
    after = reconcile(ledger_, events, SEMESTER_START, SEMESTER_END)

    assert before.total_shortfall_minutes == 7155
    assert after.total_shortfall_minutes == 7050
    assert sum(line.excused_minutes for line in after.shortfalls) == 105

    for old, new in zip(before.shortfalls, after.shortfalls):
        assert old.service == new.service
        assert old.owed_minutes == new.owed_minutes
        assert old.delivered_minutes == new.delivered_minutes
        assert old.shortfall_minutes - new.shortfall_minutes == new.excused_minutes
        assert len(old.evidence) == len(new.evidence)  # nothing was dropped to get there
