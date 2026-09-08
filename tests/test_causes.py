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

from datetime import date, timedelta

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


# ---------------------------------------------------------------------------
# Make-ups: minutes are credited to the promise they were owed against.
# ---------------------------------------------------------------------------

from minutes.correspondence import MAKE_UP_LOOKBACK_DAYS, make_up_target_on  # noqa: E402

OCT_6 = date(2026, 10, 6)
NOV_12 = date(2026, 11, 12)


def _term_ledger() -> IEPLedger:
    """One speech session a week, October through November."""
    return IEPLedger(
        student_alias="Test S.",
        school_year="2026-2027",
        iep_date=date(2026, 9, 1),
        obligations=[
            ServiceObligation(
                service=SPEECH,
                minutes_per_session=30,
                sessions_per_period=1,
                period=Period.WEEK,
                provider_role="Licensed Speech-Language Pathologist",
                setting="therapy room",
                start_date=OCT_6,
                end_date=date(2026, 11, 30),
                source_quote="30 minutes per session, once per week",
            )
        ],
        deadlines=[],
        accommodations=[],
    )


def _delivery(day: date, *, makes_up_for: date | None = None, source="email-1") -> ServiceEvent:
    return ServiceEvent(
        event_date=day,
        service=SPEECH,
        minutes=30,
        delivered=True,
        provenance=Provenance.SCHOOL_CONFIRMED,
        source=source,
        makes_up_for=makes_up_for,
    )


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("We made up the speech session she missed on 10/6 today.", OCT_6),
        ("Today's session was a make-up for Tuesday 10/6.", OCT_6),
        ("We rescheduled the 10/6 session and ran it today.", OCT_6),
    ],
)
def test_a_delivered_make_up_names_the_session_it_makes_good(body, expected):
    item = _item(body, received=NOV_12)
    assert make_up_target_on(item, NOV_12) == expected


@pytest.mark.parametrize(
    "body",
    [
        # Every one of these is in the shipped semester, and none of them is a
        # session. A promise read as a delivery erases a real shortfall with a
        # sentence the district never had to honour.
        "I have a make-up slot Friday at 1:15 - let me know if that works for her schedule.",
        "I will try to make up Maya's session next week if the schedule allows.",
        "I do not have a make-up slot this week - my Thursday and Friday are evaluations.",
        # A make-up with no session named: credits nothing, because crediting
        # the wrong miss is worse than crediting none.
        "We made up a missed session today.",
    ],
)
def test_an_offered_make_up_is_not_a_delivered_one(body):
    assert make_up_target_on(_item(body, received=NOV_12), NOV_12) is None


def test_two_candidate_dates_in_one_sentence_credit_nothing():
    body = "Today we made up the sessions missed on 10/6 and 10/13."
    assert make_up_target_on(_item(body, received=NOV_12), NOV_12) is None


def test_a_make_up_beyond_the_lookback_is_not_read():
    """A session made good five months later is a different conversation."""
    long_ago = NOV_12 - timedelta(days=MAKE_UP_LOOKBACK_DAYS + 7)
    body = f"Today we made up the session missed on {long_ago.month}/{long_ago.day}."
    assert make_up_target_on(_item(body, received=NOV_12), NOV_12) is None


def test_a_make_up_pays_back_the_period_it_was_owed_to():
    """The failure this exists to prevent.

    Three October sessions missed, three extra run in November to make them
    good. Reported on October, the naive ledger still demands ninety minutes --
    a demand the district answers by forwarding its own November log, after
    which every other figure in the letter is in doubt.
    """
    ledger = _term_ledger()
    october = [_miss(OCT_6), _miss(date(2026, 10, 13)), _miss(date(2026, 10, 20))]
    make_ups = [
        _delivery(NOV_12, makes_up_for=OCT_6, source="email-a"),
        _delivery(date(2026, 11, 13), makes_up_for=date(2026, 10, 13), source="email-b"),
        _delivery(date(2026, 11, 14), makes_up_for=date(2026, 10, 20), source="email-c"),
    ]

    line = reconcile(ledger, october + make_ups, OCT_6, date(2026, 10, 31)).shortfalls[0]
    assert line.delivered_minutes == 90
    assert line.shortfall_minutes == 0, "the minutes came back; nothing is owed for October"


def test_a_make_up_is_not_also_counted_in_the_month_it_was_run():
    """Otherwise the same thirty minutes pay for two different weeks."""
    ledger = _term_ledger()
    events = [_miss(OCT_6), _delivery(NOV_12, makes_up_for=OCT_6)]

    november = reconcile(ledger, events, date(2026, 11, 1), date(2026, 11, 30)).shortfalls[0]
    assert november.delivered_minutes == 0, (
        "November's own promise was not met by a session that paid off October"
    )
    assert november.shortfall_minutes == november.owed_minutes


def test_an_ordinary_delivery_still_counts_where_it_happened():
    """The credit is opt-in and record-driven; nothing else moves."""
    ledger = _term_ledger()
    events = [_miss(OCT_6), _delivery(NOV_12)]

    october = reconcile(ledger, events, OCT_6, date(2026, 10, 12)).shortfalls[0]
    november = reconcile(ledger, events, date(2026, 11, 9), date(2026, 11, 15)).shortfalls[0]
    assert october.owed_minutes == 30
    assert october.delivered_minutes == 0 and october.shortfall_minutes == 30
    assert november.delivered_minutes == 30, "an unlabelled session pays for the week it was run"


def test_a_make_up_retires_the_slot_it_filled_rather_than_leaving_it_undocumented():
    """The district's own record decides the date, and a make-up is that record."""
    ledger = _term_ledger()
    events = [_miss(OCT_6), _delivery(NOV_12, makes_up_for=OCT_6)]

    line = reconcile(ledger, events, OCT_6, date(2026, 10, 12)).shortfalls[0]
    assert line.owed_minutes == 30
    assert line.delivered_minutes == 30
    assert line.shortfall_minutes == 0
    assert line.undocumented_minutes == 0


# ---------------------------------------------------------------------------
# Where a reason is allowed to be seen.
#
# The engine knowing something and a parent seeing it are different facts, and
# for most of a day this feature had the first without the second.
# ---------------------------------------------------------------------------

from minutes.deadlines import evaluate_deadlines  # noqa: E402
from minutes.letters import compile_shortfall_notice, validate_letter  # noqa: E402
from minutes.statement import build_statement, render_markdown  # noqa: E402
from minutes.tools import load_case_record  # noqa: E402

TERM_START = date(2026, 9, 1)
TERM_END = date(2027, 1, 29)


def _sample_reconciliation():
    case = load_case_record()
    return case.ledger, reconcile(case.ledger, case.events, TERM_START, TERM_END)


def test_the_letter_states_a_district_reason_as_the_districts_own():
    ledger, result = _sample_reconciliation()
    body = compile_shortfall_notice(ledger, result, today=date(2027, 2, 1)).body

    assert (
        "District records give as the reason for 1 session not held on 2026-09-24 "
        "that a school activity displaced the session." in body
    )


def test_the_letter_attributes_a_family_reason_to_the_family():
    """The sentence that would put words in the district's mouth if it did not.

    Every record explaining the December weeks is the parent's own note, so
    this is the exact case a letter must not overstate.
    """
    ledger, result = _sample_reconciliation()
    body = compile_shortfall_notice(ledger, result, today=date(2027, 2, 1)).body

    assert (
        "We recorded at home that the position was vacant for 2 sessions not held on "
        "2026-12-08 and 2026-12-10; that is our observation and not a district record." in body
    )
    assert "District records give as the reason" in body
    assert "District records give as the reason for 2 sessions" not in body


def test_the_letter_draws_no_conclusion_from_a_reason():
    ledger, result = _sample_reconciliation()
    letter = compile_shortfall_notice(ledger, result, today=date(2027, 2, 1))

    assert "I am not drawing a conclusion from those reasons." in letter.body
    for forbidden in ("must be made up", "is owed", "required to make up", "entitled to"):
        assert forbidden not in letter.body, f"a reason may not become a legal conclusion: {forbidden!r}"
    assert validate_letter(letter) == [], "every reason sentence carries its own footnote"


def test_a_reason_sentence_cites_only_the_dates_it_names():
    """A footnote has to support the sentence it hangs from, and no more."""
    from minutes.letters import _reason_refs
    from minutes.reconcile import evidence_date

    _, result = _sample_reconciliation()
    line = next(line for line in result.shortfalls if line.service.startswith("Speech"))
    reason = next(r for r in line.stated_reasons if r.cause is MissCause.PROVIDER_VACANCY)

    refs = _reason_refs(line, reason)
    assert refs, "the sentence would not compile without them"
    assert {evidence_date(ref) for ref in refs} == set(reason.dates)
    assert all(ref.provenance is reason.evidence_grade for ref in refs)


def test_the_statement_shows_the_reasons_and_who_wrote_them():
    ledger, result = _sample_reconciliation()
    statement = build_statement(
        ledger, result, evaluate_deadlines(ledger, date(2027, 2, 1)), [], []
    )
    rendered = render_markdown(statement)

    assert "What the records give as the reason" in rendered
    assert "| Speech-Language Therapy | The position was vacant | 2 | you |" in rendered
    assert "A school activity displaced the session | 1 | the district |" in rendered


def test_a_statement_with_nothing_explained_shows_no_reason_section():
    """The common case. A section of nothing is noise in a forwarded document."""
    ledger = _term_ledger()
    result = reconcile(ledger, [_miss(OCT_6)], OCT_6, date(2026, 10, 12))
    statement = build_statement(ledger, result, [], [], [])

    assert "What the records give as the reason" not in render_markdown(statement)


def test_the_caseworker_can_see_the_reasons_and_that_they_are_only_reasons():
    from minutes.tools import reconcile_services

    out = reconcile_services(start=TERM_START.isoformat(), end=TERM_END.isoformat())
    speech = next(row for row in out["services"] if row["service"].startswith("Speech"))
    reasons = {r["reason"]: r for r in speech["reasons_the_records_give"]}

    assert reasons["the position was vacant"]["written_by"] == "the family"
    assert reasons["the position was vacant"]["sessions"] == 2
    assert reasons["a school activity displaced the session"]["written_by"] == "the district"


def test_a_make_up_footnote_says_which_session_it_paid_for():
    """Otherwise a November delivery inside an October total reads as an error."""
    ledger = _term_ledger()
    events = [_miss(OCT_6), _delivery(NOV_12, makes_up_for=OCT_6)]

    line = reconcile(ledger, events, OCT_6, date(2026, 10, 12)).shortfalls[0]
    details = " ".join(ref.detail for ref in line.evidence)
    assert "recorded as making up the session of 2026-10-06 and counted against that date" in details
