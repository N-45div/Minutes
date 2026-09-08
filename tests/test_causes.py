"""The reason a record gives for a missed session.

Two properties are load-bearing here and everything in this file is about one
of them.

**It describes, it does not decide.** No minute in any bucket moves because of
a cause. There is no blanket rule in IDEA that a missed session must be made
up — OSEP's framing is whether the interruption denied the child a free
appropriate public education, an individual determination — so a document
assembler that let a reason change an amount would be asserting a legal
conclusion it has no business asserting.

**It stays on its own date, and with its own author.** A service log is one
document spanning a semester; a reason written in one row must not re-caption
the forty rows around it. And a reason a parent wrote is not a reason the
district wrote, which is the difference between a report and an admission.
"""

from datetime import date

import pytest

from minutes.correspondence import stated_cause_on
from minutes.models import (
    Attribution,
    Correspondence,
    CorrespondenceKind,
    IEPLedger,
    MissCause,
    Period,
    Provenance,
    ServiceEvent,
    ServiceObligation,
)
from minutes.reconcile import reconcile

SPEECH = "Speech-Language Therapy"
MONDAY = date(2026, 11, 9)
TUESDAY = date(2026, 11, 10)


def _item(body: str, *, received: date = TUESDAY, kind=CorrespondenceKind.SCHOOL_EMAIL) -> Correspondence:
    return Correspondence(
        item_id="email-2026-11-10-01",
        received=received,
        kind=kind,
        sender="Ms. R. Renner, M.A., CCC-SLP",
        subject="Maya",
        body=body,
    )


def _ledger() -> IEPLedger:
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
                start_date=date(2026, 11, 9),
                end_date=date(2026, 11, 13),
                source_quote="30 minutes per session, 2 sessions per week",
            )
        ],
        deadlines=[],
        accommodations=[],
    )


def _miss(day: date, *, provenance=Provenance.SCHOOL_CONFIRMED, source="email-2026-11-10-01") -> ServiceEvent:
    return ServiceEvent(
        event_date=day,
        service=SPEECH,
        minutes=0,
        delivered=False,
        provenance=provenance,
        source=source,
    )


# ---------------------------------------------------------------------------
# What the patterns read.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            "The speech-language pathologist position is still vacant, so there was no session today.",
            MissCause.PROVIDER_VACANCY,
        ),
        (
            "No speech today - we have not been able to fill the SLP role.",
            MissCause.PROVIDER_VACANCY,
        ),
        (
            "The therapist was out sick today, so speech did not happen.",
            MissCause.PROVIDER_ABSENT,
        ),
        (
            "No speech today — there was no coverage for the SLP.",
            MissCause.PROVIDER_ABSENT,
        ),
        (
            "No session today; state testing ran all morning.",
            MissCause.TESTING,
        ),
        (
            "School was closed today for a snow day, so there was no session.",
            MissCause.SCHOOL_CLOSURE,
        ),
        (
            "She was pulled for an assembly today, so we missed the speech session.",
            MissCause.SCHOOL_ACTIVITY,
        ),
        (
            "Maya was absent today so we did not have speech.",
            MissCause.STUDENT_ABSENT,
        ),
    ],
)
def test_a_reason_written_beside_a_date_is_read(body, expected):
    assert stated_cause_on(_item(body), TUESDAY) is expected


@pytest.mark.parametrize(
    "body",
    [
        # No reason at all — the ordinary case, and the one a wide pattern
        # would quietly invent an answer for.
        "We were not able to see her today.",
        "No speech today.",
        "Speech was held today for the full 30 minutes.",
        # A reason with no date attached to it.
        "The SLP position has been vacant since the autumn.",
        # Sentences that mention the vocabulary without giving a reason.
        "Please return the testing consent form today.",
        "Today's assembly schedule is attached for your reference.",
        # A known and accepted miss. The reason and the date are in separate
        # clauses of a semicolon-joined line, and the line does not open with
        # its date, so neither the sentence rule nor the row rule reaches it.
        # Narrow is the deliberate direction: a reason this misses costs a
        # sentence the letter would have been stronger for, and a reason it
        # invented would cost the letter's credibility.
        "We have not been able to fill the SLP role; no speech today.",
    ],
)
def test_a_reason_that_is_not_actually_given_stays_unstated(body):
    assert stated_cause_on(_item(body), TUESDAY) is MissCause.UNSTATED


def test_the_childs_absence_wins_over_any_other_reason():
    """Otherwise the printed reason could contradict the excusal arithmetic."""
    body = "Maya was absent today, and the therapist was out sick as well."
    assert stated_cause_on(_item(body), TUESDAY) is MissCause.STUDENT_ABSENT


# ---------------------------------------------------------------------------
# Where a reason is allowed to reach.
# ---------------------------------------------------------------------------


LOG = """11/03 speech - session held, 30 min.
11/10 speech - not held. The SLP position is vacant.
11/17 speech - not held.
11/24 speech - not held. Therapist out sick."""


def test_a_reason_in_a_log_row_belongs_to_that_row_and_no_other():
    """The failure that would let one line re-caption a whole semester."""
    log = _item(LOG, received=date(2026, 12, 1), kind=CorrespondenceKind.SERVICE_LOG)

    assert stated_cause_on(log, date(2026, 11, 10)) is MissCause.PROVIDER_VACANCY
    assert stated_cause_on(log, date(2026, 11, 24)) is MissCause.PROVIDER_ABSENT
    assert stated_cause_on(log, date(2026, 11, 17)) is MissCause.UNSTATED, (
        "this row gives no reason, and the row above it is not its reason"
    )
    assert stated_cause_on(log, date(2026, 11, 3)) is MissCause.UNSTATED


def test_a_reason_dated_to_one_day_does_not_reach_a_miss_on_another():
    """Straight from the shipped semester: a snow day on the 14th, a miss on the 12th."""
    note = _item(
        "snow day today, school closed. also tuesday 1/12 - no speech again",
        received=date(2027, 1, 14),
        kind=CorrespondenceKind.PARENT_LOG,
    )

    assert stated_cause_on(note, date(2027, 1, 14)) is MissCause.SCHOOL_CLOSURE
    assert stated_cause_on(note, date(2027, 1, 12)) is MissCause.UNSTATED, (
        "the school being shut on Thursday does not account for Tuesday"
    )


def test_a_delivery_carries_no_reason():
    """There is nothing to explain about a session that happened."""
    delivered = ServiceEvent(
        event_date=TUESDAY,
        service=SPEECH,
        minutes=30,
        delivered=True,
        provenance=Provenance.SCHOOL_CONFIRMED,
        source="email-2026-11-10-01",
    )
    from minutes.correspondence import attributed

    body = "The therapist was out sick today, so speech did not happen."
    (stamped,) = attributed([delivered], [_item(body)])
    assert stamped.cause is MissCause.UNSTATED


# ---------------------------------------------------------------------------
# What it must not do: move a number.
# ---------------------------------------------------------------------------


def test_a_stated_reason_moves_no_minutes():
    """The property the whole feature is built around.

    The same two missed sessions, read once with the district explaining them
    as a staffing vacancy and once with the district explaining nothing. Every
    figure is identical. Only the description differs, because whether missed
    minutes are owed back is a FAPE determination and not a document
    assembler's to make.
    """
    ledger = _ledger()
    silent = [
        _miss(MONDAY).model_copy(update={"cause": MissCause.UNSTATED}),
        _miss(TUESDAY).model_copy(update={"cause": MissCause.UNSTATED}),
    ]
    explained = [
        _miss(MONDAY).model_copy(update={"cause": MissCause.PROVIDER_VACANCY}),
        _miss(TUESDAY).model_copy(update={"cause": MissCause.PROVIDER_VACANCY}),
    ]

    a = reconcile(ledger, silent, MONDAY, date(2026, 11, 13)).shortfalls[0]
    b = reconcile(ledger, explained, MONDAY, date(2026, 11, 13)).shortfalls[0]

    assert a.model_dump(exclude={"stated_reasons"}) == b.model_dump(exclude={"stated_reasons"})
    assert a.stated_reasons == []
    assert [(r.cause, r.sessions) for r in b.stated_reasons] == [(MissCause.PROVIDER_VACANCY, 2)]


def test_a_student_absence_still_excuses_exactly_as_it_did():
    """The one reason that does move minutes moves them through Attribution."""
    ledger = _ledger()
    excused = [
        _miss(TUESDAY).model_copy(
            update={"attribution": Attribution.STUDENT_ABSENCE, "cause": MissCause.STUDENT_ABSENT}
        )
    ]

    line = reconcile(ledger, excused, MONDAY, date(2026, 11, 13)).shortfalls[0]
    assert line.excused_minutes == 30
    assert [(r.cause, r.sessions) for r in line.stated_reasons] == [(MissCause.STUDENT_ABSENT, 1)]


# ---------------------------------------------------------------------------
# Who wrote the reason.
# ---------------------------------------------------------------------------


def test_reasons_are_grouped_by_who_wrote_them():
    """A parent reporting a vacancy is not the district admitting one.

    Collapsing the two would let the weaker record be presented as the
    stronger, which is the single most expensive sentence a letter can carry.
    """
    ledger = _ledger()
    events = [
        _miss(MONDAY, provenance=Provenance.SCHOOL_CONFIRMED, source="email-1").model_copy(
            update={"cause": MissCause.PROVIDER_VACANCY}
        ),
        _miss(TUESDAY, provenance=Provenance.PARENT_OBSERVED, source="plog-1").model_copy(
            update={"cause": MissCause.PROVIDER_VACANCY}
        ),
    ]

    line = reconcile(ledger, events, MONDAY, date(2026, 11, 13)).shortfalls[0]
    assert [(r.cause, r.evidence_grade, r.sessions) for r in line.stated_reasons] == [
        (MissCause.PROVIDER_VACANCY, Provenance.SCHOOL_CONFIRMED, 1),
        (MissCause.PROVIDER_VACANCY, Provenance.PARENT_OBSERVED, 1),
    ]


def test_one_date_recorded_twice_within_a_grade_is_one_session():
    ledger = _ledger()
    events = [
        _miss(TUESDAY, source="email-1").model_copy(update={"cause": MissCause.TESTING}),
        _miss(TUESDAY, source="email-2").model_copy(update={"cause": MissCause.TESTING}),
    ]

    (reason,) = reconcile(ledger, events, MONDAY, date(2026, 11, 13)).shortfalls[0].stated_reasons
    assert reason.sessions == 1 and reason.dates == [TUESDAY]


# ---------------------------------------------------------------------------
# The shipped semester.
# ---------------------------------------------------------------------------


def test_the_sample_semester_reads_the_reasons_it_actually_contains():
    """Hand-checked against the fixture text, not recomputed from the code."""
    from minutes.tools import load_case_record

    case = load_case_record()
    result = reconcile(case.ledger, case.events, date(2026, 9, 1), date(2027, 1, 29))
    found = {
        (line.service, reason.cause, reason.evidence_grade): reason.sessions
        for line in result.shortfalls
        for reason in line.stated_reasons
    }

    # The parent's December notes say there is no speech teacher. Hers is the
    # only dated record of those weeks, which is the point of the fixture.
    assert found[(SPEECH, MissCause.PROVIDER_VACANCY, Provenance.PARENT_OBSERVED)] == 2
    # The district's own email explains 24 September as an assembly.
    assert found[(SPEECH, MissCause.SCHOOL_ACTIVITY, Provenance.SCHOOL_CONFIRMED)] == 1
    # No district record ever gives a vacancy as the reason for a dated session.
    assert (SPEECH, MissCause.PROVIDER_VACANCY, Provenance.SCHOOL_CONFIRMED) not in found
