"""Correspondence fixture and classifier tests.

No test in this file makes a network call. The deterministic layer is exercised
with inline drafts, and everything that depends on the model is read from the
cached fixture written by scripts/classify_once.py.

Two rules this file tries to keep to. Fixtures state the thing their test
names: a test called "keeps its stated minutes" runs against a body that
states minutes. And assertions about the cached artifact are hand-computed
wherever a re-invocation of the code under test would move with it — a test
that recomputes a guard against the artifact that guard produced cannot fail.
"""

import inspect
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from minutes import correspondence
from minutes.correspondence import (
    BATCH_CHAR_BUDGET,
    BATCH_MAX_ITEMS,
    CORRESPONDENCE_DIR,
    MAX_SESSION_FACTOR,
    PROVENANCE_BY_KIND,
    SYNTHETIC_MARKER,
    VOTE_QUORUM,
    EventDraft,
    _batches,
    _drafts_to_events,
    attributed,
    cache_payload,
    date_is_grounded,
    load_cached_events,
    load_correspondence,
    reports_student_absence,
    stated_minutes,
    student_absence_events,
)
from minutes.models import (
    Attribution,
    Correspondence,
    CorrespondenceKind,
    IEPLedger,
    Period,
    Provenance,
    ServiceObligation,
)

CACHE = Path(__file__).resolve().parents[1] / "fixtures" / "cache" / "maya_fall_2026_events.json"

SEMESTER_START = date(2026, 9, 1)
SEMESTER_END = date(2027, 1, 29)

# The stretch where the SLP post sat vacant and speech quietly stopped.
VACANCY_START = date(2026, 12, 8)
VACANCY_END = date(2027, 1, 22)

SPEECH = "Speech-Language Therapy"


def _ledger(start_date: date = date(2026, 9, 8), end_date: date = date(2027, 6, 11)) -> IEPLedger:
    return IEPLedger(
        student_alias="Test S.",
        school_year="2026-2027",
        iep_date=date(2026, 9, 1),
        obligations=[
            ServiceObligation(
                service=SPEECH,
                minutes_per_session=30,
                sessions_per_period=2,
                period=Period.WEEK,
                provider_role="Licensed Speech-Language Pathologist",
                setting="therapy room",
                start_date=start_date,
                end_date=end_date,
                source_quote="30 minutes per session, 2 sessions per week",
            )
        ],
        deadlines=[],
        accommodations=[],
    )


def _item(**overrides) -> Correspondence:
    base = dict(
        item_id="email-2026-11-10-01",
        received=date(2026, 11, 10),
        kind=CorrespondenceKind.SCHOOL_EMAIL,
        sender="Ms. R. Renner, M.A., CCC-SLP",
        subject="Maya",
        body="Saw Maya today. Good session.",
    )
    base.update(overrides)
    return Correspondence(**base)


def _draft(**overrides) -> EventDraft:
    base = dict(
        item_id="email-2026-11-10-01",
        event_date=date(2026, 11, 10),
        service=SPEECH,
        delivered=True,
        minutes=30,
    )
    base.update(overrides)
    return EventDraft(**base)


def _events(drafts, items=None, ledger=None, min_readings=1):
    items = items if items is not None else [_item()]
    return _drafts_to_events(drafts, items, ledger or _ledger(), min_readings=min_readings)


# ---------------------------------------------------------------------------
# The declared contract
# ---------------------------------------------------------------------------


def test_the_public_entry_point_takes_exactly_the_declared_arguments():
    """Other modules call this by its published signature; it does not drift."""
    assert list(inspect.signature(correspondence.classify_to_events).parameters) == ["items", "ledger"]
    assert list(inspect.signature(correspondence.load_correspondence).parameters) == ["name"]
    assert list(inspect.signature(correspondence.load_cached_events).parameters) == ["name"]


# ---------------------------------------------------------------------------
# The synthetic semester
# ---------------------------------------------------------------------------


def test_semester_is_a_realistic_size_and_sits_inside_the_term():
    items = load_correspondence()

    assert 45 <= len(items) <= 70
    assert all(SEMESTER_START <= i.received <= SEMESTER_END for i in items)
    assert items == sorted(items, key=lambda i: (i.received, i.item_id))
    assert len({i.item_id for i in items}) == len(items)


def test_no_correspondence_lands_on_a_weekend():
    assert [i.item_id for i in load_correspondence() if i.received.weekday() >= 5] == []


def test_semester_carries_every_kind_of_evidence():
    kinds = {i.kind for i in load_correspondence()}
    assert kinds == set(CorrespondenceKind)


def test_every_fixture_file_is_marked_synthetic():
    paths = sorted(CORRESPONDENCE_DIR.glob("maya_fall_2026_*.json"))
    assert paths

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "SYNTHETIC" in payload[SYNTHETIC_MARKER].upper()


def test_load_refuses_a_file_that_is_not_marked_synthetic(tmp_path, monkeypatch):
    unmarked = {"items": [json.loads(_item().model_dump_json())]}
    (tmp_path / "anon_fall_2026_school_email.json").write_text(json.dumps(unmarked), encoding="utf-8")
    monkeypatch.setattr(correspondence, "CORRESPONDENCE_DIR", tmp_path)

    with pytest.raises(ValueError, match="not marked synthetic"):
        load_correspondence("anon_fall_2026")


def test_load_rejects_a_duplicate_item_id(tmp_path, monkeypatch):
    duplicated = {
        SYNTHETIC_MARKER: "SYNTHETIC TEST DATA",
        "items": [json.loads(_item().model_dump_json()), json.loads(_item().model_dump_json())],
    }
    (tmp_path / "dupe_fall_2026_school_email.json").write_text(json.dumps(duplicated), encoding="utf-8")
    monkeypatch.setattr(correspondence, "CORRESPONDENCE_DIR", tmp_path)

    with pytest.raises(ValueError, match="duplicate item_id"):
        load_correspondence("dupe_fall_2026")


def test_missing_fixture_names_itself():
    with pytest.raises(FileNotFoundError):
        load_correspondence("no_such_semester")


# ---------------------------------------------------------------------------
# Provenance is derived, never generated
# ---------------------------------------------------------------------------


def test_every_kind_has_a_provenance_and_none_is_documented_silence():
    assert set(PROVENANCE_BY_KIND) == set(CorrespondenceKind)
    assert Provenance.DOCUMENTED_SILENCE not in PROVENANCE_BY_KIND.values()


def test_school_sources_are_school_confirmed():
    # A compiled document is issued after the sessions it records, so the log
    # and the report are received later than the day they are reporting on.
    received_by_kind = {
        CorrespondenceKind.SCHOOL_EMAIL: date(2026, 11, 10),
        CorrespondenceKind.SERVICE_LOG: date(2027, 1, 8),
        CorrespondenceKind.PROGRESS_REPORT: date(2027, 1, 8),
    }

    for kind, received in received_by_kind.items():
        item = _item(item_id="x-1", kind=kind, received=received, body="2026-11-10  30  Held  speech")
        events = _events([_draft(item_id="x-1")], [item])
        assert events[0].provenance is Provenance.SCHOOL_CONFIRMED


def test_a_parent_log_is_only_ever_parent_observed():
    item = _item(item_id="plog-1", kind=CorrespondenceKind.PARENT_LOG)
    events = _events([_draft(item_id="plog-1")], [item])

    assert events[0].provenance is Provenance.PARENT_OBSERVED
    assert events[0].source == "plog-1"


# ---------------------------------------------------------------------------
# The deterministic guard around the model
# ---------------------------------------------------------------------------


def test_a_service_the_iep_does_not_promise_is_dropped():
    assert _events([_draft(service="Music Therapy")]) == []


def test_a_draft_from_an_unknown_item_is_dropped():
    assert _events([_draft(item_id="email-that-does-not-exist")]) == []


def test_service_name_is_snapped_to_the_ledger_spelling():
    events = _events([_draft(service="  speech-language therapy  ")])
    assert events[0].service == SPEECH


def test_a_missed_session_carries_no_minutes():
    events = _events([_draft(delivered=False, minutes=30)])

    assert events[0].delivered is False
    assert events[0].minutes == 0


def test_an_item_cannot_report_a_session_that_had_not_happened_yet():
    item = _item(received=date(2026, 11, 10), body="Speech on 11/19 will go ahead.")
    assert _events([_draft(event_date=date(2026, 11, 19))], [item]) == []


def test_a_session_the_day_after_the_item_was_written_is_refused():
    """The edge of the same rule: received + 1, not received + 9."""
    wednesday = date(2026, 11, 11)
    item = _item(received=date(2026, 11, 10), body="Speech on 11/11 is the plan.")

    assert wednesday == item.received + timedelta(days=1)
    assert _events([_draft(event_date=wednesday)], [item]) == []


def test_a_same_day_email_may_report_its_own_date():
    events = _events([_draft()])
    assert events[0].event_date == date(2026, 11, 10)


@pytest.mark.parametrize("weekend_day", [date(2026, 11, 14), date(2026, 11, 15)])
def test_a_weekend_session_is_refused(weekend_day):
    item = _item(received=date(2026, 11, 16), body=f"{weekend_day.month}/{weekend_day.day} no speech")

    assert weekend_day.weekday() >= 5
    assert _events([_draft(event_date=weekend_day, delivered=False)], [item]) == []


def test_the_first_day_of_the_service_term_is_admissible():
    start = date(2026, 9, 8)
    item = _item(received=start, body="9/8 no speech")

    events = _events([_draft(event_date=start, delivered=False)], [item], _ledger(start_date=start))
    assert [e.event_date for e in events] == [start]


def test_the_day_before_the_service_term_starts_is_refused():
    start = date(2026, 9, 8)
    before = start - timedelta(days=1)
    item = _item(received=start, body="9/7 no speech")

    assert before.weekday() < 5  # not passing on the weekend rule instead
    assert _events([_draft(event_date=before, delivered=False)], [item], _ledger(start_date=start)) == []


def test_the_last_day_of_the_service_term_is_admissible():
    end = date(2027, 6, 10)
    item = _item(received=end, body="6/10 no speech")

    events = _events([_draft(event_date=end, delivered=False)], [item], _ledger(end_date=end))
    assert [e.event_date for e in events] == [end]


def test_the_day_after_the_service_term_ends_is_refused():
    end = date(2027, 6, 10)
    after = end + timedelta(days=1)
    item = _item(received=after, body="6/11 no speech")

    assert after.weekday() < 5
    assert _events([_draft(event_date=after, delivered=False)], [item], _ledger(end_date=end)) == []


def test_a_date_the_document_never_names_is_refused():
    vague = _item(body="Services are being provided as outlined in the IEP.")
    assert _events([_draft()], [vague]) == []


def test_a_compiled_document_does_not_report_a_session_on_its_own_issue_date():
    issued = date(2027, 1, 29)
    report = _item(
        item_id="progrpt-1",
        received=issued,
        kind=CorrespondenceKind.PROGRESS_REPORT,
        body="IEP PROGRESS REPORT\nDate issued: 2027-01-29\nMaya's speech session goals continue.",
    )
    assert _events([_draft(item_id="progrpt-1", event_date=issued)], [report]) == []


# ---------------------------------------------------------------------------
# Durations must be grounded in the item that claims them
# ---------------------------------------------------------------------------


def test_minutes_are_backfilled_from_the_iep_when_the_item_states_none():
    assert _events([_draft(minutes=0)])[0].minutes == 30


def test_a_shortened_session_keeps_the_duration_the_item_states():
    short = _item(body="Saw Maya today. Her session ran about fifteen minutes.")

    assert stated_minutes(short) == frozenset({15})
    assert _events([_draft(minutes=15)], [short])[0].minutes == 15


def test_a_duration_the_item_never_states_falls_back_to_the_iep():
    """The unit-level twin of the plog-2026-11-12-01 defect.

    The default fixture body states no duration at all, so a 15 arriving from
    the model came from somewhere else — in the shipped run, from a different
    email in the same batch. A number nobody wrote down is not evidence.
    """
    plain = _item()

    assert stated_minutes(plain) == frozenset()
    assert _events([_draft(minutes=15)], [plain])[0].minutes == 30


def test_a_duration_stated_in_a_neighbouring_item_does_not_attach_to_this_one():
    """Verbatim from the semester: the parent log states no duration at all.

    The provider's own email that afternoon says "about fifteen minutes", and
    the two items sit in the same batch. Reading 15 onto this item deletes 15
    delivered minutes from the record on evidence that does not contain them.
    """
    parent_log = _item(
        item_id="plog-2026-11-12-01",
        received=date(2026, 11, 12),
        kind=CorrespondenceKind.PARENT_LOG,
        sender="Ms. D. Reyes (parent)",
        subject="quick log",
        body="11/12 she says speech was 'only for a little bit' today",
    )
    drafts = [_draft(item_id="plog-2026-11-12-01", event_date=date(2026, 11, 12), minutes=15)]

    assert _events(drafts, [parent_log])[0].minutes == 30


def test_a_make_up_session_longer_than_the_iep_promises_is_kept():
    """A doubled-up make-up is 60 real minutes; clamping it to 30 loses 30."""
    long_session = _item(body="Speech today - a make-up, so we ran sixty minutes.")

    assert stated_minutes(long_session) == frozenset({60})
    assert _events([_draft(minutes=60)], [long_session])[0].minutes == 60


def test_an_absurd_duration_is_clamped_to_something_a_school_day_could_hold():
    absurd = _item(body="Speech today. The note says 600 minutes, which cannot be right.")

    assert stated_minutes(absurd) == frozenset({600})
    assert _events([_draft(minutes=600)], [absurd])[0].minutes == MAX_SESSION_FACTOR * 30


@pytest.mark.parametrize(
    "body,expected",
    [
        ("I had Maya for our first OT session today, the full 45 minutes.", {45}),
        ("We got started with Maya today. Thirty minutes, in a group of three.", {30}),
        ("Maya's session ran short today, about fifteen minutes.", {15}),
        ("2026-09-22  10:15-10:45   30   G    3    Therapy room   Held", {30}),
        ("2026-09-16  13:00-13:45   45   I    1    OT room   Held", {45}),
        ("No OT today. These minutes will be recovered when the schedule permits.", set()),
        ("Saw Maya today. Good session.", set()),
    ],
)
def test_stated_minutes_reads_only_durations_the_item_writes_down(body, expected):
    assert stated_minutes(_item(body=body)) == frozenset(expected)


# ---------------------------------------------------------------------------
# Voting across passes
# ---------------------------------------------------------------------------


def test_readings_that_contradict_each_other_establish_nothing():
    split = [_draft(delivered=True), _draft(delivered=False)]
    assert _events(split) == []


def test_a_lone_bad_reading_loses_to_the_majority():
    outvoted = [_draft(delivered=True), _draft(delivered=False), _draft(delivered=True)]
    events = _events(outvoted)

    assert len(events) == 1
    assert events[0].delivered is True


def test_disagreeing_durations_are_voted_not_minimised():
    """Duration is decided the same way delivery is, or the guard is a sham.

    Taking min() here would let the single 15 lose the question of whether the
    session happened and win the question of how long it ran — and it loses 15
    delivered minutes in the direction that overstates a shortfall.
    """
    stated = _item(body="Session today. It ran the full thirty minutes, not the fifteen minutes I first wrote.")
    disagreeing = [_draft(minutes=30), _draft(minutes=15), _draft(minutes=30)]

    assert stated_minutes(stated) == frozenset({15, 30})
    assert _events(disagreeing, [stated])[0].minutes == 30


def test_durations_that_split_evenly_fall_back_to_the_iep():
    stated = _item(body="Session today, either fifteen minutes or thirty minutes - the note is unclear.")
    tied = [_draft(minutes=15), _draft(minutes=30)]

    assert _events(tied, [stated])[0].minutes == 30


def test_a_fact_only_one_pass_ever_read_does_not_meet_the_quorum():
    """Three passes only protect a fact all three passes had a chance to see.

    A key one pass emitted and the others omitted has been read once. Silence
    is not a vote against it, but it is not a vote for it either, so it does
    not establish a missed session against a school on its own.
    """
    lone = [_draft(delivered=False)]

    assert _events(lone, min_readings=VOTE_QUORUM) == []
    assert len(_events(lone + [_draft(delivered=False)], min_readings=VOTE_QUORUM)) == 1


def test_the_quorum_is_what_the_classifier_actually_applies():
    assert VOTE_QUORUM == 2
    assert correspondence.CLASSIFIER_PASSES >= VOTE_QUORUM
    # The unit-level default replays already-established facts; the quorum is
    # applied where the ballots exist, in classify_to_events.
    assert len(_events([_draft()])) == 1


def test_two_documents_about_the_same_day_stay_two_facts():
    school = _item(item_id="email-1", body="Speech today, good session.")
    parent = _item(item_id="plog-1", kind=CorrespondenceKind.PARENT_LOG, body="no speech today")
    events = _events(
        [_draft(item_id="email-1"), _draft(item_id="plog-1", delivered=False)],
        [school, parent],
    )

    assert {e.source for e in events} == {"email-1", "plog-1"}


# ---------------------------------------------------------------------------
# Date grounding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body,day",
    [
        ("Saw Maya today.", date(2026, 11, 10)),
        ("No speech yesterday.", date(2026, 11, 9)),
        ("2026-11-12  30 min  Held", date(2026, 11, 12)),
        ("11/12 she says speech was short", date(2026, 11, 12)),
        ("I had Maya down for November 12 and was not able to see her.", date(2026, 11, 12)),
        ("I had Maya down for Nov 12 and was not able to see her.", date(2026, 11, 12)),
        ("I could not fit her group in on Thursday.", date(2026, 11, 5)),
    ],
)
def test_a_date_written_any_of_the_usual_ways_is_grounded(body, day):
    assert date_is_grounded(_item(body=body), day)


def test_a_short_date_does_not_match_inside_a_longer_one():
    item = _item(body="11/5 no speech")

    assert date_is_grounded(item, date(2026, 11, 5))
    assert not date_is_grounded(item, date(2027, 1, 5))


def test_a_date_named_in_a_sentence_about_something_else_grounds_nothing():
    """Grounding is claim attachment, not token presence.

    Both of these name a date the school year contains, and neither says
    anything about a service. If a bare token search were enough, either would
    ground a session the document never mentions.
    """
    picture_day = _item(
        item_id="email-2026-09-11-01",
        received=date(2026, 9, 11),
        subject="Picture Day - September 24",
        body="Picture Day is Thursday, September 24. Order envelopes went home in backpacks today.",
    )
    report_cards = _item(
        item_id="email-2026-11-06-01",
        received=date(2026, 11, 6),
        subject="Report cards go home today",
        body="First-quarter report cards go home in backpacks this afternoon.",
    )

    assert not date_is_grounded(picture_day, date(2026, 9, 24))
    assert not date_is_grounded(report_cards, date(2026, 11, 6))


def test_a_note_that_no_session_has_ever_happened_grounds_no_dated_miss():
    """Verbatim from the semester. This says speech has NEVER begun.

    It asserts no session on any day, so it must produce no dated fact — even
    though the word "today" appears in it, and even though the sentence that
    word sits in also says "speech". This is the sentence pattern the system
    prompt names as emit-nothing; the guard is what makes that binding.
    """
    log = _item(
        item_id="plog-2026-09-14-01",
        received=date(2026, 9, 14),
        kind=CorrespondenceKind.PARENT_LOG,
        sender="Ms. D. Reyes (parent)",
        subject="quick log",
        body=(
            "no speech yet. asked maya again today and she says she hasn't met "
            "the speech teacher. school started a week ago"
        ),
    )

    assert not date_is_grounded(log, date(2026, 9, 14))
    assert _events([_draft(item_id="plog-2026-09-14-01", event_date=date(2026, 9, 14), delivered=False)], [log]) == []


def test_a_closure_notice_grounds_nothing_however_many_dates_it_names():
    snow = _item(
        item_id="email-2027-01-14-01",
        received=date(2027, 1, 14),
        subject="SCHOOL CLOSED - snow",
        body=(
            "All Unified School District schools and offices are closed today, "
            "Thursday January 14, due to snow. All after-school activities are cancelled."
        ),
    )
    assert not date_is_grounded(snow, date(2027, 1, 14))


def test_a_weekday_name_reaches_only_this_week_and_last_week():
    item = _item(received=date(2026, 11, 10), body="We missed her group on Monday last week.")
    last_week_monday = date(2026, 11, 2)
    the_monday_before_that = date(2026, 10, 26)

    assert date_is_grounded(item, last_week_monday)
    assert not date_is_grounded(item, the_monday_before_that)


def test_this_week_and_last_week_pick_out_different_weekdays():
    this_week = _item(received=date(2026, 11, 13), body="We missed her group on Monday this week.")
    last_week = _item(received=date(2026, 11, 13), body="We missed her group on Monday last week.")

    assert date_is_grounded(this_week, date(2026, 11, 9))
    assert not date_is_grounded(this_week, date(2026, 11, 2))
    assert date_is_grounded(last_week, date(2026, 11, 2))
    assert not date_is_grounded(last_week, date(2026, 11, 9))


def test_a_weekday_that_could_mean_either_of_two_weeks_grounds_neither():
    """Written on a Friday, "Tuesday" is two admissible Tuesdays.

    The guard cannot read the writer's mind, and picking the nearer one would
    fabricate a dated miss against a school on a day the writer never meant.
    """
    friday = _item(received=date(2026, 11, 13), body="We missed her group on Tuesday.")

    assert not date_is_grounded(friday, date(2026, 11, 10))
    assert not date_is_grounded(friday, date(2026, 11, 3))


def test_a_weekday_is_unambiguous_when_only_one_of_the_two_has_happened():
    """Written on a Tuesday, only last week's Thursday exists yet."""
    tuesday = _item(received=date(2026, 11, 10), body="I could not fit her group in on Thursday.")

    assert date_is_grounded(tuesday, date(2026, 11, 5))
    assert not date_is_grounded(tuesday, date(2026, 11, 12))


# ---------------------------------------------------------------------------
# Excusable non-delivery, carried forward rather than folded in
# ---------------------------------------------------------------------------


def test_a_session_the_child_was_absent_for_is_flagged_as_such():
    absent = _item(
        item_id="email-2026-10-14-01",
        received=date(2026, 10, 14),
        body="Maya was absent today so we missed OT. We will pick back up next Wednesday.",
    )
    assert reports_student_absence(absent)


def test_a_provider_out_sick_is_not_the_child_being_absent():
    """Same sentence shape, opposite fact, opposite consequence."""
    provider_out = _item(
        item_id="email-2027-01-20-01",
        received=date(2027, 1, 20),
        body="I am out sick today so there is no OT. I have a make-up slot Friday at 1:15.",
    )
    assert not reports_student_absence(provider_out)


# Ordinary school prose in which SOMEBODY ELSE was away. Every one of these
# carries an absence marker and a third-person word, which is why a rule
# looking only for those two things read all of them as the child being out.
# A staffing vacancy read as an absence is a real shortfall excused to zero by
# one sentence, so the subject of the absence decides: in each of these, the
# last person named before the absence is not the child.
@pytest.mark.parametrize(
    "body",
    [
        "The therapist was absent last week, so her speech sessions did not take place.",
        "Our OT was out sick this week and could not see her.",
        "Coverage note: the speech-language pathologist has been absent since 10/1 and "
        "the child has received no services.",
        "He was absent for the entire grading period; the student's speech minutes were "
        "not delivered.",
        "Her one-to-one aide was absent, so the resource room session was cancelled.",
        "In the absence of a provider we could not run the group.",
        "The substitute was absent on 10/14 so nothing ran.",
    ],
    ids=[
        "therapist",
        "ot-provider",
        "pathologist-vacancy",
        "unresolved-pronoun",
        "one-to-one-aide",
        "absence-of-a-provider",
        "substitute",
    ],
)
def test_somebody_elses_absence_is_never_the_childs(body):
    assert not reports_student_absence(_item(body=body))


@pytest.mark.parametrize(
    "body",
    [
        "Maya was absent today so we missed OT.",
        "10/14 - maya home sick. no OT obviously. my fault",
        "Maya missed resource room on Tuesday and again on Wednesday this week - she was "
        "in the nurse's office both days with a stomach ache.",
        "I kept her home today so there was no speech.",
        "The student was not at school today.",
    ],
    ids=["named", "parent-log", "pronoun-after-the-name", "kept-home", "the-student"],
)
def test_the_childs_own_absence_is_still_recognized(body):
    """The rule narrowed on WHO was away, not on how anybody writes it."""
    assert reports_student_absence(_item(body=body))


def test_an_absence_in_one_row_of_a_service_log_stays_in_that_row():
    """A service log is ONE item covering a semester.

    Spreading one absence row across the whole document is how a stretch of
    provider vacancy becomes an excused semester: six sessions the district
    should answer for, struck on the strength of one day the child was out.
    """
    log = _item(
        item_id="svclog-2026-11-02-01",
        received=date(2026, 11, 2),
        kind=CorrespondenceKind.SERVICE_LOG,
        body="\n".join(
            [
                "DATE        STATUS    NOTE",
                "2026-09-08  Not held  no provider assigned",
                "2026-09-10  Not held  no provider assigned",
                "2026-09-15  Not held  no provider assigned",
                "2026-09-17  Not held  no provider assigned",
                "2026-10-14  Not held  Maya was absent",
                "2026-10-21  Not held  no provider assigned",
                "2026-10-28  Not held  no provider assigned",
            ]
        ),
    )
    rows = [
        date(2026, 9, 8),
        date(2026, 9, 10),
        date(2026, 9, 15),
        date(2026, 9, 17),
        date(2026, 10, 14),
        date(2026, 10, 21),
        date(2026, 10, 28),
    ]
    events = _events(
        [_draft(item_id=log.item_id, event_date=day, delivered=False) for day in rows], [log]
    )
    stamped = attributed(events, [log])

    assert len(stamped) == 7
    assert [e.event_date for e in stamped if e.attribution is Attribution.STUDENT_ABSENCE] == [
        date(2026, 10, 14)
    ]


def test_an_absence_no_sentence_dates_excuses_nothing():
    """The document says she was out; it never says which day.

    An exclusion the district cannot check against its own attendance calendar
    is worth less than the minutes it gives up, so those minutes stay in the
    difference — where the letter's standing request to compare the figures
    against the district's own records is what surfaces them.
    """
    undated = _item(
        item_id="email-2026-10-20-01",
        received=date(2026, 10, 20),
        body="Maya has been out a lot lately and was absent again recently. We missed her group on 10/13.",
    )
    events = _events(
        [_draft(item_id=undated.item_id, event_date=date(2026, 10, 13), delivered=False)], [undated]
    )

    assert reports_student_absence(undated)  # the document does say it
    assert [e.attribution for e in attributed(events, [undated])] == [
        Attribution.SCHOOL_OR_UNRECORDED
    ]


def test_a_stamp_the_document_does_not_support_is_cleared_rather_than_kept():
    """Attribution is derived on every load, so it is authoritative both ways.

    An event arriving already stamped — out of a cache, or from a caller — is
    re-derived against the document it came from. A stamp that outlives its own
    rule is a cached grade wearing a different name.
    """
    ordinary = _item(
        item_id="email-2026-11-05-01",
        received=date(2026, 11, 5),
        body="Parent conferences are today so specialists are covering classrooms. No speech groups ran.",
    )
    events = _events(
        [_draft(item_id=ordinary.item_id, event_date=date(2026, 11, 5), delivered=False)], [ordinary]
    )
    presumed = [e.model_copy(update={"attribution": Attribution.STUDENT_ABSENCE}) for e in events]

    assert [e.attribution for e in attributed(presumed, [ordinary])] == [
        Attribution.SCHOOL_OR_UNRECORDED
    ]


def test_student_absence_events_names_the_misses_a_letter_must_qualify():
    absent = _item(
        item_id="email-2026-10-14-01",
        received=date(2026, 10, 14),
        body="Maya was absent today so we missed OT.",
    )
    cancelled = _item(
        item_id="email-2026-11-05-01",
        received=date(2026, 11, 5),
        body="Parent conferences are today so specialists are covering classrooms. No speech groups ran.",
    )
    events = _events(
        [
            _draft(item_id="email-2026-10-14-01", event_date=date(2026, 10, 14), delivered=False),
            _draft(item_id="email-2026-11-05-01", event_date=date(2026, 11, 5), delivered=False),
        ],
        [absent, cancelled],
    )
    flagged = student_absence_events(events, [absent, cancelled])

    assert [e.source for e in flagged] == ["email-2026-10-14-01"]
    assert len(events) == 2  # the fact itself is still recorded, only annotated


def test_attribution_marks_the_absence_and_leaves_the_school_s_own_miss_alone():
    absent = _item(
        item_id="email-2026-10-14-01",
        received=date(2026, 10, 14),
        body="Maya was absent today so we missed OT.",
    )
    cancelled = _item(
        item_id="email-2026-11-05-01",
        received=date(2026, 11, 5),
        body="Parent conferences are today so specialists are covering classrooms. No speech groups ran.",
    )
    events = _events(
        [
            _draft(item_id="email-2026-10-14-01", event_date=date(2026, 10, 14), delivered=False),
            _draft(item_id="email-2026-11-05-01", event_date=date(2026, 11, 5), delivered=False),
        ],
        [absent, cancelled],
    )
    stamped = {e.source: e.attribution for e in attributed(events, [absent, cancelled])}

    assert stamped["email-2026-10-14-01"] is Attribution.STUDENT_ABSENCE
    assert stamped["email-2026-11-05-01"] is Attribution.SCHOOL_OR_UNRECORDED


def test_a_delivered_session_is_never_attributed_to_an_absence():
    """A document can report an absence and a session in the same breath — the
    parent's log of a short week, say. Attribution answers why a session did
    not happen, so a session that did have nothing to answer for."""
    mixed = _item(
        item_id="plog-2026-10-14-01",
        received=date(2026, 10, 14),
        kind=CorrespondenceKind.PARENT_LOG,
        body="Maya was absent today. She did have speech on 10/13 though, 30 minutes.",
    )
    events = _events(
        [_draft(item_id="plog-2026-10-14-01", event_date=date(2026, 10, 13), delivered=True)],
        [mixed],
    )
    stamped = attributed(events, [mixed])

    assert [e.delivered for e in stamped] == [True]
    assert stamped[0].attribution is Attribution.SCHOOL_OR_UNRECORDED


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


def test_batching_keeps_every_item_exactly_once_and_in_order():
    items = load_correspondence()
    batched = [item for batch in _batches(items) for item in batch]

    assert batched == items


def test_batches_respect_both_caps():
    for batch in _batches(load_correspondence()):
        assert len(batch) <= BATCH_MAX_ITEMS
        weight = sum(len(i.subject) + len(i.body) for i in batch)
        assert weight <= BATCH_CHAR_BUDGET or len(batch) == 1


def test_batching_is_far_cheaper_than_one_call_per_item():
    items = load_correspondence()
    assert len(list(_batches(items))) <= len(items) // 4


# ---------------------------------------------------------------------------
# The cached classification
# ---------------------------------------------------------------------------

# Hand-written, not recomputed: (source, date) -> the words in that document
# that put the session on that day. Re-running date_is_grounded over the cache
# would pass no matter how far the guard was loosened, because the guard is
# what produced the cache. These pairs were read off the fixture by hand.
GROUNDING_SAMPLE = {
    ("email-2026-09-24-01", date(2026, 9, 24)): "No speech for Maya today",
    ("plog-2026-10-06-01", date(2026, 10, 6)): "10/6 she says no speech today",
    ("plog-2026-11-12-01", date(2026, 11, 12)): "11/12 she says speech was",
    ("plog-2026-12-17-01", date(2026, 12, 17)): "12/17 nothing",
    ("email-2026-12-18-02", date(2026, 12, 9)): "I had Maya down for December 9",
    ("svclog-2027-01-08-01", date(2026, 12, 3)): "2026-12-03  10:15-10:45",
    ("email-2027-01-15-01", date(2027, 1, 12)): "on Tuesday and again on Wednesday this week",
}

# The routine inbox traffic the classifier has to read past. Every one of these
# names a date; none of them says anything about a service session.
NOISE_ITEMS = {
    "email-2026-09-04-01",  # welcome back, first day is Tuesday
    "email-2026-09-11-01",  # picture day, September 24
    "email-2026-09-18-01",  # PTA fundraiser
    "email-2026-09-25-01",  # bus route change
    "email-2026-10-08-01",  # staff development day, no school Friday
    "email-2026-10-21-01",  # field trip permission slip
    "email-2026-11-06-01",  # report cards go home today
    "email-2026-12-18-01",  # winter break dates
    "email-2027-01-08-01",  # records cover note, no dates of service
    "email-2027-01-14-01",  # snow closure
    "email-2027-01-19-01",  # winter benchmark testing window
    "email-2027-01-29-01",  # book fair
}


def test_the_cached_classification_is_committed():
    """A missing cache must be a failure, not a quiet skip.

    Everything below this line is the honesty surface of the module — the
    provenance grades, the documented silence, the vacancy. If the cache can
    go missing and the suite still goes green, none of it is verified.
    """
    assert CACHE.exists(), f"{CACHE} is missing; run scripts/classify_once.py and commit the result"


class TestCachedEvents:
    def test_every_event_is_traceable_to_a_real_item(self):
        events = load_cached_events()
        item_ids = {i.item_id for i in load_correspondence()}

        assert events
        assert all(e.source in item_ids for e in events)

    def test_the_classifier_never_claims_documented_silence(self):
        assert all(e.provenance is not Provenance.DOCUMENTED_SILENCE for e in load_cached_events())

    def test_parent_logs_and_only_parent_logs_are_parent_observed(self):
        """Graded off the id prefix, not off the table that produced the cache."""
        for event in load_cached_events():
            expected = (
                Provenance.PARENT_OBSERVED
                if event.source.startswith("plog-")
                else Provenance.SCHOOL_CONFIRMED
            )
            assert event.provenance is expected, event

    def test_both_evidence_grades_are_represented(self):
        grades = {e.provenance for e in load_cached_events()}
        assert grades == {Provenance.SCHOOL_CONFIRMED, Provenance.PARENT_OBSERVED}

    def test_minutes_agree_with_delivery(self):
        for event in load_cached_events():
            assert (event.minutes > 0) == event.delivered

    def test_every_event_names_a_service_the_iep_promises(self):
        ledger_path = CACHE.parent / "iep_maya_ledger.json"
        ledger = IEPLedger.model_validate_json(ledger_path.read_text(encoding="utf-8"))
        promised = {o.service for o in ledger.obligations}

        for event in load_cached_events():
            assert event.service in promised

    def test_no_cached_session_was_recorded_longer_than_the_iep_promises(self):
        """Not a clamp — the guard allows a make-up to run long. No item in
        this semester states one, so nothing in the artifact does either."""
        ledger_path = CACHE.parent / "iep_maya_ledger.json"
        ledger = IEPLedger.model_validate_json(ledger_path.read_text(encoding="utf-8"))
        promised = {o.service: o.minutes_per_session for o in ledger.obligations}

        for event in load_cached_events():
            assert event.minutes <= promised[event.service], event

    def test_sampled_events_stand_on_words_the_document_actually_contains(self):
        events = {(e.source, e.event_date) for e in load_cached_events()}
        items = {i.item_id: i for i in load_correspondence()}

        for (source, day), quote in GROUNDING_SAMPLE.items():
            assert (source, day) in events, f"{source} {day} missing from the cache"
            assert quote in f"{items[source].subject}\n{items[source].body}", quote

    def test_no_cached_minutes_were_borrowed_from_another_document(self):
        """Every delivered figure is either written in its own source or is
        the IEP's own per-session number. Nothing in between."""
        items = {i.item_id: i for i in load_correspondence()}
        ledger_path = CACHE.parent / "iep_maya_ledger.json"
        ledger = IEPLedger.model_validate_json(ledger_path.read_text(encoding="utf-8"))
        promised = {o.service: o.minutes_per_session for o in ledger.obligations}

        for event in load_cached_events():
            if not event.delivered:
                continue
            source_text = stated_minutes(items[event.source])
            assert event.minutes in source_text or event.minutes == promised[event.service], event

    def test_the_parent_log_of_a_short_session_does_not_invent_the_number(self):
        """The single defect this artifact was regenerated to remove.

        plog-2026-11-12-01 says the session was "only for a little bit". The
        provider's email the same afternoon says "about fifteen minutes" — and
        it is that email, not the parent, that documents fifteen.
        """
        by_key = {(e.source, e.event_date): e for e in load_cached_events()}

        assert by_key[("plog-2026-11-12-01", date(2026, 11, 12))].minutes == 30
        assert by_key[("email-2026-11-12-01", date(2026, 11, 12))].minutes == 15

    def test_no_event_is_dated_before_speech_ever_began(self):
        """plog-2026-09-14-01 says speech has not started. It dates nothing."""
        assert "plog-2026-09-14-01" not in {e.source for e in load_cached_events()}

    def test_events_fall_on_school_days_inside_the_semester(self):
        for event in load_cached_events():
            assert event.event_date.weekday() < 5
            assert SEMESTER_START <= event.event_date <= SEMESTER_END

    def test_no_event_lands_on_an_announced_closure(self):
        """Hand-read off the announcement emails, not derived from anything."""
        closed = {
            date(2026, 10, 9),  # staff development day
            date(2026, 11, 26),  # Thanksgiving
            date(2026, 11, 27),
            date(2027, 1, 14),  # snow
        }
        closed |= {date(2026, 12, 21) + timedelta(days=n) for n in range(12)}  # winter break

        assert closed.isdisjoint({e.event_date for e in load_cached_events()})

    def test_events_are_sorted(self):
        events = load_cached_events()
        assert events == sorted(events, key=lambda e: (e.event_date, e.service, e.source))

    def test_routine_school_announcements_produce_no_events(self):
        item_ids = {i.item_id for i in load_correspondence()}
        assert NOISE_ITEMS <= item_ids  # every id named here is really in the semester
        assert NOISE_ITEMS.isdisjoint({e.source for e in load_cached_events()})

    def test_a_progress_report_yields_no_delivery_evidence(self):
        """The district's own on-time compliance artifact documents no minutes.

        This is the product's thesis, asserted as a test: a progress report
        narrates goals and says only that services 'are being provided as
        outlined in the IEP', so it cannot answer whether they were.
        """
        reports = {
            i.item_id
            for i in load_correspondence()
            if i.kind is CorrespondenceKind.PROGRESS_REPORT
        }
        assert reports
        assert reports.isdisjoint({e.source for e in load_cached_events()})

    def test_a_records_deflection_yields_no_delivery_evidence(self):
        assert "email-2026-11-30-01" not in {e.source for e in load_cached_events()}

    def test_the_district_produced_a_partial_log_covering_only_held_sessions(self):
        from_logs = [e for e in load_cached_events() if e.source.startswith("svclog-")]

        assert from_logs
        assert all(e.delivered for e in from_logs)

    def test_during_the_vacancy_only_the_parent_documents_the_missing_speech(self):
        """The demo's whole point, pinned down.

        For six weeks the district said nothing about speech at all, so the
        only dated record that sessions were missed is the parent's log — which
        is why that evidence grade has to survive all the way into the letters
        rather than being discarded as hearsay.
        """
        during = [
            e
            for e in load_cached_events()
            if e.service == SPEECH and VACANCY_START <= e.event_date <= VACANCY_END
        ]
        parent_misses = [e for e in during if e.provenance is Provenance.PARENT_OBSERVED]

        assert len(parent_misses) >= 5
        assert all(not e.delivered for e in during)
        assert [e for e in during if e.provenance is Provenance.SCHOOL_CONFIRMED] == []

    def test_the_service_log_stops_dead_when_the_provider_resigned(self):
        logged_speech = [
            e.event_date
            for e in load_cached_events()
            if e.source == "svclog-2027-01-08-01" and e.service == SPEECH
        ]
        assert logged_speech
        assert max(logged_speech) == date(2026, 12, 3)

    def test_the_absences_the_school_could_not_have_delivered_are_named(self):
        """Handed to reconciliation and the letters explicitly.

        Both of these are non-deliveries the child caused. ServiceEvent has no
        field for that, and SCHOOL_CONFIRMED facts may be cited in a letter
        without qualification, so a compensatory demand could otherwise be
        built on a session nobody could have held.
        """
        items = load_correspondence()
        flagged = student_absence_events(load_cached_events(), items)

        assert {(e.event_date, e.service) for e in flagged} == {
            (date(2026, 10, 14), "Occupational Therapy"),
            (date(2027, 1, 12), "Specialized Academic Instruction"),
        }

    def test_no_cancellation_the_school_caused_is_mistaken_for_an_absence(self):
        """The 2026-12-08 vacancy misses are the school's; they stay unflagged."""
        items = load_correspondence()
        flagged = {e.source for e in student_absence_events(load_cached_events(), items)}

        assert "plog-2026-12-10-01" not in flagged
        assert "email-2026-11-05-01" not in flagged
        assert "email-2027-01-20-01" not in flagged

    def test_exactly_the_three_absences_come_back_attributed(self):
        """The three facts reconciliation must keep out of a compensatory demand.

        Read off the fixture by hand: the provider's email and the parent's log
        of the same 2026-10-14 OT session, and the January email reporting that
        Maya was in the nurse's office. Anything else stamped here would be a
        session excused out of a letter that the school should have to answer
        for; any of these three missed would be a session billed to a school
        that could not have held it.
        """
        stamped = {
            (e.source, e.event_date, e.service)
            for e in load_cached_events()
            if e.attribution is Attribution.STUDENT_ABSENCE
        }

        assert stamped == {
            ("email-2026-10-14-01", date(2026, 10, 14), "Occupational Therapy"),
            ("plog-2026-10-14-01", date(2026, 10, 14), "Occupational Therapy"),
            ("email-2027-01-15-01", date(2027, 1, 12), "Specialized Academic Instruction"),
        }

    def test_attribution_is_derived_on_load_rather_than_read_out_of_the_cache(self):
        """The cached JSON has no attribution field, and must never need one.

        A cached grade is a grade frozen on the day the classifier ran: tighten
        the absence rule and the source improves while the shipped artifact
        keeps claiming the same minutes. Deriving it on the way out means the
        committed cache — written before this rule existed — is correct now.
        """
        raw = json.loads(CACHE.read_text(encoding="utf-8"))

        assert raw
        assert all("attribution" not in event for event in raw)
        assert any(e.attribution is Attribution.STUDENT_ABSENCE for e in load_cached_events())

    def test_the_next_classifier_run_will_not_start_caching_it_either(self):
        """The rule has to survive the next regeneration of this fixture.

        ``scripts/classify_once.py`` serializes through ``cache_payload``, and a
        plain model dump would quietly start writing the derived field back —
        at which point the committed artifact freezes whatever the absence rule
        happened to be that day, and the test above only notices a fixture
        regeneration later.
        """
        events = load_cached_events()
        stamped = [e for e in events if e.attribution is Attribution.STUDENT_ABSENCE]

        assert stamped  # the payload really is dropping something
        assert all("attribution" not in row for row in cache_payload(events))
        assert all("cause" not in row for row in cache_payload(events)), (
            "the stated reason is derived on read too, and freezing it would ship "
            "whatever the cause patterns happened to be the day the classifier ran"
        )
        assert cache_payload(events) == json.loads(CACHE.read_text(encoding="utf-8"))

    def test_the_cache_is_refused_without_the_correspondence_it_was_read_from(self, tmp_path, monkeypatch):
        """Two files ship together now, so a missing one is named, not inferred.

        Attribution is derived from the correspondence on every read, so the
        cached events alone are no longer enough. A deployment that shipped one
        without the other used to fail deep inside a function the caller never
        called; it now says which directory is empty.
        """
        monkeypatch.setattr(correspondence, "CORRESPONDENCE_DIR", tmp_path)

        with pytest.raises(FileNotFoundError, match="correspondence"):
            load_cached_events()

    def test_every_delivered_event_comes_back_unattributed(self):
        for event in load_cached_events():
            if event.delivered:
                assert event.attribution is Attribution.SCHOOL_OR_UNRECORDED, event
