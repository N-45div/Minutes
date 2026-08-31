"""Active-discovery tests — cadence, request state machine, and silence facts.

Every fixture here is synthetic. "Maya P." and "Ridgeview" are invented; no
real child, school, or district is described. No test in this file makes a
network call or an LLM call.
"""

from datetime import date, timedelta

import pytest

from minutes import discovery
from minutes.discovery import (
    RESPONSE_WINDOW_DAYS,
    due_requests,
    mark_answered,
    mark_sent,
    refresh_states,
    silence_events,
)
from minutes.models import (
    IEPLedger,
    Period,
    Provenance,
    RecordsRequest,
    RequestState,
    ServiceObligation,
)

SPEECH = "Speech-Language Therapy"
OT = "Occupational Therapy"
COUNSELING = "Counseling Services"


def _obligation(service: str, **overrides) -> ServiceObligation:
    base = dict(
        service=service,
        minutes_per_session=30,
        sessions_per_period=2,
        period=Period.WEEK,
        provider_role="Licensed Speech-Language Pathologist",
        setting="therapy room",
        start_date=date(2026, 9, 8),
        end_date=date(2027, 6, 11),
        source_quote="30 minutes per session, 2 sessions per week",
    )
    base.update(overrides)
    return ServiceObligation(**base)


def _ledger(*obligations: ServiceObligation) -> IEPLedger:
    return IEPLedger(
        student_alias="Maya P.",
        school_year="2026-2027",
        iep_date=date(2026, 9, 1),
        obligations=list(obligations) or [_obligation(SPEECH), _obligation(OT)],
        deadlines=[],
        accommodations=[],
    )


def _request(**overrides) -> RecordsRequest:
    base = dict(
        request_id="req-001",
        covers_start=date(2026, 9, 8),
        covers_end=date(2026, 10, 7),
        services=[SPEECH, OT],
        state=RequestState.DRAFT,
    )
    base.update(overrides)
    return RecordsRequest(**base)


# ---------------------------------------------------------------------------
# due_requests — when the agent asks
# ---------------------------------------------------------------------------


def test_first_request_covers_from_the_earliest_obligation_start():
    ledger = _ledger(
        _obligation(SPEECH, start_date=date(2026, 9, 8)),
        _obligation(OT, start_date=date(2026, 9, 21)),
    )

    (request,) = due_requests(ledger, [], today=date(2026, 11, 1))

    assert request.state is RequestState.DRAFT
    assert request.covers_start == date(2026, 9, 8)
    assert request.covers_end == date(2026, 11, 1)
    assert request.services == [SPEECH, OT]
    assert request.sent_on is None
    assert request.response_due is None


def test_no_request_until_the_cadence_window_has_accrued():
    ledger = _ledger()
    start = date(2026, 9, 8)

    # Day 29 of the window: not yet 30 days of uncovered period.
    assert due_requests(ledger, [], today=start + timedelta(days=28)) == []
    # Day 30, counted inclusively.
    assert len(due_requests(ledger, [], today=start + timedelta(days=29))) == 1


def test_cadence_is_measured_from_the_end_of_the_last_covered_window():
    ledger = _ledger()
    answered = mark_answered(
        mark_sent(_request(covers_end=date(2026, 10, 7)), date(2026, 10, 8)),
        date(2026, 10, 20),
    )

    assert due_requests(ledger, [answered], today=date(2026, 11, 5)) == []

    (request,) = due_requests(ledger, [answered], today=date(2026, 11, 6))
    # Windows tile: the new one starts the day after the last one ended.
    assert request.covers_start == date(2026, 10, 8)
    assert request.covers_end == date(2026, 11, 6)


def test_custom_cadence_is_respected():
    ledger = _ledger()
    start = date(2026, 9, 8)

    assert due_requests(ledger, [], today=start + timedelta(days=12), cadence_days=14) == []
    assert len(due_requests(ledger, [], today=start + timedelta(days=13), cadence_days=14)) == 1


def test_open_draft_blocks_a_duplicate_request():
    ledger = _ledger()
    draft = _request(state=RequestState.DRAFT)

    assert due_requests(ledger, [draft], today=date(2027, 1, 15)) == []


def test_sent_request_still_inside_its_window_blocks_a_duplicate():
    ledger = _ledger()
    sent = mark_sent(_request(), date(2026, 10, 8))

    assert due_requests(ledger, [sent], today=date(2027, 1, 15)) == []


def test_overdue_unanswered_request_does_not_stall_the_cadence():
    """A school that never answers must not be able to switch discovery off."""
    ledger = _ledger()
    overdue = refresh_states(
        [mark_sent(_request(), date(2026, 10, 8))], today=date(2026, 12, 20)
    )[0]
    assert overdue.state is RequestState.UNANSWERED_OVERDUE

    (request,) = due_requests(ledger, [overdue], today=date(2026, 12, 20))
    assert request.covers_start == date(2026, 10, 8)


def test_request_ids_do_not_collide_with_existing_ids():
    ledger = _ledger()
    answered = mark_answered(
        mark_sent(_request(request_id="req-002"), date(2026, 10, 8)), date(2026, 10, 20)
    )

    (request,) = due_requests(ledger, [answered], today=date(2026, 11, 20))
    assert request.request_id == "req-003"


def test_request_ids_stay_above_a_sparse_history():
    """Ids order the audit trail, so a new one must never sort below an old one."""
    ledger = _ledger()
    answered = mark_answered(
        mark_sent(_request(request_id="req-005"), date(2026, 10, 8)), date(2026, 10, 20)
    )

    (request,) = due_requests(ledger, [answered], today=date(2026, 11, 20))

    assert request.request_id == "req-006"


def test_a_non_positive_cadence_is_refused():
    """cadence_days=0 would mint a fresh request against the district every run."""
    ledger = _ledger()

    for cadence in (0, -5):
        with pytest.raises(ValueError):
            due_requests(ledger, [], today=date(2026, 11, 1), cadence_days=cadence)


def test_only_services_overlapping_the_window_are_requested():
    ledger = _ledger(
        _obligation(SPEECH),
        _obligation(OT, start_date=date(2026, 9, 8), end_date=date(2026, 10, 31)),
        _obligation(COUNSELING, start_date=date(2027, 2, 1)),
    )
    covered = mark_answered(
        mark_sent(_request(covers_end=date(2026, 11, 30)), date(2026, 12, 1)),
        date(2026, 12, 10),
    )

    (request,) = due_requests(ledger, [covered], today=date(2027, 1, 5))

    # OT ended before the window; counseling has not started yet.
    assert request.services == [SPEECH]


def test_no_request_when_the_ledger_promises_nothing():
    empty = IEPLedger(
        student_alias="Maya P.",
        school_year="2026-2027",
        iep_date=date(2026, 9, 1),
        obligations=[],
        deadlines=[],
        accommodations=[],
    )

    assert due_requests(empty, [], today=date(2027, 1, 5)) == []


def test_no_request_before_services_have_begun():
    ledger = _ledger(_obligation(SPEECH, start_date=date(2026, 9, 8)))

    assert due_requests(ledger, [], today=date(2026, 8, 20)) == []


# ---------------------------------------------------------------------------
# mark_sent / mark_answered — the state machine
# ---------------------------------------------------------------------------


def test_mark_sent_computes_the_45_day_statutory_deadline():
    sent = mark_sent(_request(), date(2026, 10, 8))

    assert RESPONSE_WINDOW_DAYS == 45
    assert sent.state is RequestState.SENT
    assert sent.sent_on == date(2026, 10, 8)
    assert sent.response_due == date(2026, 11, 22)  # 45 calendar days, no tolling


def test_mark_sent_returns_a_copy_and_leaves_the_draft_untouched():
    draft = _request()
    sent = mark_sent(draft, date(2026, 10, 8))

    assert draft.state is RequestState.DRAFT
    assert draft.response_due is None
    assert sent is not draft


def test_an_upcoming_iep_meeting_shortens_the_deadline():
    """34 CFR 300.613(a) stacks three deadlines; the earliest governs."""
    sent = mark_sent(_request(), date(2026, 10, 8), before_meeting_on=date(2026, 10, 20))

    assert sent.response_due == date(2026, 10, 20)


def test_a_distant_meeting_does_not_extend_the_deadline():
    sent = mark_sent(_request(), date(2026, 10, 8), before_meeting_on=date(2027, 3, 4))

    assert sent.response_due == date(2026, 11, 22)


def test_a_meeting_on_the_45th_day_leaves_the_statutory_deadline_intact():
    sent = mark_sent(_request(), date(2026, 10, 8), before_meeting_on=date(2026, 11, 22))

    assert sent.response_due == date(2026, 11, 22)


def test_a_meeting_already_held_is_refused_rather_than_back_dating_the_deadline():
    """A stale meeting date would allege a breach that predates the request."""
    with pytest.raises(ValueError):
        mark_sent(_request(), date(2026, 10, 8), before_meeting_on=date(2026, 10, 1))


def test_a_meeting_on_the_day_of_sending_is_refused():
    with pytest.raises(ValueError):
        mark_sent(_request(), date(2026, 10, 8), before_meeting_on=date(2026, 10, 8))


def test_response_due_can_never_precede_the_send(monkeypatch):
    """The invariant every dated silence fact rests on, guarded independently."""
    monkeypatch.setattr(discovery, "RESPONSE_WINDOW_DAYS", -1)

    with pytest.raises(ValueError):
        mark_sent(_request(), date(2026, 10, 8))


def test_mark_sent_refuses_to_restart_a_running_clock():
    sent = mark_sent(_request(), date(2026, 10, 8))

    with pytest.raises(ValueError):
        mark_sent(sent, date(2026, 11, 1))


def test_mark_answered_records_the_response():
    answered = mark_answered(mark_sent(_request(), date(2026, 10, 8)), date(2026, 10, 30))

    assert answered.state is RequestState.ANSWERED
    assert answered.answered_on == date(2026, 10, 30)
    # Preserved so lateness stays computable.
    assert answered.response_due == date(2026, 11, 22)


def test_a_late_answer_closes_an_overdue_request():
    overdue = refresh_states(
        [mark_sent(_request(), date(2026, 10, 8))], today=date(2026, 12, 1)
    )[0]

    answered = mark_answered(overdue, date(2026, 12, 3))

    assert answered.state is RequestState.ANSWERED


def test_mark_answered_refuses_a_request_that_was_never_sent():
    draft = _request()

    with pytest.raises(ValueError):
        mark_answered(draft, date(2026, 10, 30))


# ---------------------------------------------------------------------------
# refresh_states — the overdue boundary
# ---------------------------------------------------------------------------


def test_still_compliant_on_the_deadline_day_itself():
    """Producing on day 45 is compliance: "in no case more than 45 days".

    The boundary date is written out rather than read back off ``sent``, so
    this test fails if mark_sent's arithmetic drifts instead of moving with it.
    """
    sent = mark_sent(_request(), date(2026, 10, 8))

    (refreshed,) = refresh_states([sent], today=date(2026, 11, 22))

    assert refreshed.state is RequestState.SENT


def test_overdue_on_the_first_day_past_the_deadline():
    sent = mark_sent(_request(), date(2026, 10, 8))

    (refreshed,) = refresh_states([sent], today=date(2026, 11, 23))

    assert refreshed.state is RequestState.UNANSWERED_OVERDUE
    assert sent.state is RequestState.SENT  # original untouched


def test_overdue_stays_overdue_and_other_states_are_left_alone():
    sent = mark_sent(_request(), date(2026, 10, 8))
    draft = _request(request_id="req-002")
    answered = mark_answered(
        mark_sent(_request(request_id="req-003"), date(2026, 10, 8)), date(2026, 10, 20)
    )

    once = refresh_states([sent, draft, answered], today=date(2027, 1, 30))
    twice = refresh_states(once, today=date(2027, 2, 28))

    assert [r.state for r in twice] == [
        RequestState.UNANSWERED_OVERDUE,
        RequestState.DRAFT,
        RequestState.ANSWERED,
    ]


def test_refresh_states_on_an_empty_list():
    assert refresh_states([], today=date(2027, 1, 30)) == []


# ---------------------------------------------------------------------------
# silence_events — what silence is allowed to assert
# ---------------------------------------------------------------------------


def _overdue_request(**overrides) -> RecordsRequest:
    sent = mark_sent(_request(**overrides), date(2026, 10, 8))
    return refresh_states([sent], today=date(2027, 1, 30))[0]


def test_silence_event_per_service_dated_at_the_missed_deadline():
    ledger = _ledger()
    overdue = _overdue_request()

    events = silence_events([overdue], ledger, today=date(2027, 1, 30))

    assert [e.service for e in events] == [SPEECH, OT]
    for event in events:
        assert event.event_date == overdue.response_due
        assert event.minutes == 0
        assert event.delivered is False
        assert event.provenance is Provenance.DOCUMENTED_SILENCE
        assert event.source == overdue.request_id


def test_silence_events_use_the_ledgers_spelling_and_skip_unknown_services():
    ledger = _ledger(_obligation(SPEECH))
    overdue = _overdue_request(services=["speech-language therapy", "Adapted PE"])

    events = silence_events([overdue], ledger, today=date(2027, 1, 30))

    assert [e.service for e in events] == [SPEECH]


def test_answered_requests_never_become_silence():
    ledger = _ledger()
    late_answer = mark_answered(_overdue_request(), date(2027, 1, 20))

    assert silence_events([late_answer], ledger, today=date(2027, 1, 30)) == []


def test_a_request_inside_its_window_is_not_silence():
    ledger = _ledger()
    sent = mark_sent(_request(), date(2026, 10, 8))

    assert silence_events([sent], ledger, today=date(2026, 11, 1)) == []


def test_silence_is_never_asserted_before_the_deadline_arrives():
    """Guards a hand-built or unrefreshed list from back-dating a future breach."""
    ledger = _ledger()
    premature = _overdue_request().model_copy(
        update={"response_due": date(2027, 6, 1)}
    )

    assert silence_events([premature], ledger, today=date(2027, 1, 30)) == []


def test_silence_events_on_empty_inputs():
    assert silence_events([], _ledger(), today=date(2027, 1, 30)) == []


def test_silence_is_not_asserted_when_the_deadline_precedes_the_request():
    """A contradictory record cannot date a breach before the request existed."""
    ledger = _ledger()
    impossible = _overdue_request().model_copy(
        update={"sent_on": date(2026, 10, 8), "response_due": date(2026, 10, 1)}
    )

    assert silence_events([impossible], ledger, today=date(2027, 1, 30)) == []


def test_silence_events_are_a_recomputation_not_an_increment():
    """Called on two different days, the same overdue request yields one event.

    Callers must replace their silence set with this result, never append to
    it; this pins the property that makes replacing safe.
    """
    ledger = _ledger(_obligation(SPEECH))
    overdue = _overdue_request(services=[SPEECH])

    first = silence_events([overdue], ledger, today=date(2026, 11, 23))
    second = silence_events([overdue], ledger, today=date(2026, 11, 24))

    assert len(first) == 1
    assert first == second
    assert first[0].event_date == date(2026, 11, 22)


def test_one_silence_fact_per_service_however_the_request_spells_it():
    """Two spellings of one promised service are one missing record, not two."""
    ledger = _ledger(_obligation(SPEECH))
    overdue = _overdue_request(
        services=[SPEECH, "speech-language therapy", "  Speech-Language Therapy  "]
    )

    events = silence_events([overdue], ledger, today=date(2027, 1, 30))

    assert [e.service for e in events] == [SPEECH]


def test_silence_never_reports_delivered_minutes():
    """The only shape a silence fact may take: zero minutes, nothing delivered.

    Anything else would let a consumer read a silence event as a record of
    what happened in the session rather than a record of the missing document.
    """
    ledger = _ledger()
    overdue = _overdue_request()

    events = silence_events([overdue], ledger, today=date(2027, 1, 30))

    assert events
    for event in events:
        assert event.provenance is Provenance.DOCUMENTED_SILENCE
        assert event.delivered is False
        assert event.minutes == 0


# ---------------------------------------------------------------------------
# the whole mechanism, end to end
# ---------------------------------------------------------------------------


def test_discovery_cycle_from_draft_to_dated_silence():
    ledger = _ledger()
    requests: list[RecordsRequest] = []

    (first,) = due_requests(ledger, requests, today=date(2026, 10, 8))
    requests = [mark_sent(first, date(2026, 10, 8))]

    # Inside the statutory window: nothing to say, nothing new to send.
    requests = refresh_states(requests, today=date(2026, 11, 1))
    assert silence_events(requests, ledger, today=date(2026, 11, 1)) == []
    assert due_requests(ledger, requests, today=date(2026, 11, 1)) == []

    # Past it: the school's silence is now dated evidence, and the next
    # request comes due for the period the first one did not cover.
    overdue_day = date(2026, 11, 23)
    requests = refresh_states(requests, today=overdue_day)
    events = silence_events(requests, ledger, today=overdue_day)
    assert len(events) == 2
    assert {e.event_date for e in events} == {date(2026, 11, 22)}

    (second,) = due_requests(ledger, requests, today=overdue_day)
    assert second.request_id == "req-002"
    assert second.covers_start == date(2026, 10, 9)
