"""Letter integrity tests.

Minutes' central promise is that a compiled letter structurally cannot
fabricate an accusation: a claim it cannot footnote is dropped, a parent
observation is never dressed up as a district record, an owed minute with no
evidence either way is called undocumented and nothing more, and a citation
that is not in the verified table cannot appear at all. Those are the
properties this file tests, on both the happy path (every compiled letter
validates) and the adversarial one (hand-built letters that break each rule
must fail).

Every fixture below is synthetic. No test in this file makes a network call or
an LLM call.
"""

import re
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path

import pytest

from minutes.letters import (
    ACCUSATORY_PHRASES,
    ALLOWED_CITATIONS,
    PARENT_ATTRIBUTION_PHRASES,
    _Draft,
    _reconciliation_blocks,
    compile_compensatory_request,
    compile_deadline_reminder,
    compile_records_request,
    compile_shortfall_notice,
    placeholders,
    validate_letter,
)
from minutes.models import (
    Accommodation,
    Attribution,
    Deadline,
    DeadlineKind,
    DeadlineState,
    DeadlineStatus,
    EvidenceRef,
    IEPLedger,
    Letter,
    LetterCitation,
    LetterKind,
    Period,
    Provenance,
    ReconciliationResult,
    RecordsRequest,
    RequestState,
    ServiceEvent,
    ServiceObligation,
    ServiceShortfall,
)
from minutes.reconcile import ABSENCE_NOTE, reconcile

TODAY = date(2026, 11, 2)

# What a sender's UI would substitute once the parent has filled the blank.
_PLACEHOLDER_SUB = re.compile(r"\[\[PARENT:.*?\]\]", re.DOTALL)

# The verified citation table, transcribed here by hand from the build's legal
# research rather than imported from the module under test.
#
# This is the point of the file. Every other citation assertion compares the
# module against itself and so cannot fail if a section number is fabricated
# straight into ALLOWED_CITATIONS — the single most damaging thing this product
# could do. Adding, removing, or editing an entry in letters.py must therefore
# require a deliberate matching edit here, in front of a reviewer.
EXPECTED_CITATION_TABLE = frozenset(
    {
        # Records access — IDEA Part B and FERPA.
        "34 CFR 300.501(a)",
        "34 CFR 300.611(b)",
        "34 CFR 300.611(c)",
        "34 CFR 300.613(a)",
        "34 CFR 300.613(b)(1)-(3)",
        "34 CFR 300.613(c)",
        "34 CFR 300.614",
        "34 CFR 300.615",
        "34 CFR 300.616",
        "34 CFR 300.617(a) and (b)",
        "34 CFR 300.618 and 300.619",
        "34 CFR 300.624(a) and (b)",
        "34 CFR 99.3 (definition of 'Education records'), para. (a)(1)-(2)",
        "34 CFR 99.3 (definition of 'Education records'), para. (b)(1)",
        "34 CFR 99.3 (definition of 'Education records'), para. (b)(4)(i)-(iii)",
        "34 CFR 99.10(a)",
        "34 CFR 99.10(b)",
        "34 CFR 99.10(c)",
        "34 CFR 99.10(d)",
        "34 CFR 99.10(e)",
        "34 CFR 99.11(a) and (b)",
        # Service delivery.
        "20 U.S.C. 1401(9)(D)",
        "20 U.S.C. 1414(d)(1)(A)(i)(III)",
        "20 U.S.C. 1414(d)(4)(A)(i)",
        "34 CFR 300.17(d)",
        "34 CFR 300.34(a)",
        "34 CFR 300.101(a)",
        "34 CFR 300.320(a)(3)(i)-(ii)",
        "34 CFR 300.320(a)(4)",
        "34 CFR 300.320(a)(7)",
        "34 CFR 300.323(a)",
        "34 CFR 300.323(c)(2)",
        "34 CFR 300.323(d)(1)-(2)",
        "34 CFR 300.324(a)(4)(i)",
        "34 CFR 300.324(a)(6)",
        "34 CFR 300.324(b)(1)(i)",
        "34 CFR 300.324(b)(1)(ii)(A)",
        # Timelines and notice.
        "20 U.S.C. 1414(a)(2)(B)",
        "20 U.S.C. 1415(b)(3)",
        "20 U.S.C. 1415(c)(1)",
        "34 CFR 300.11(a)",
        "34 CFR 300.11(b)",
        "34 CFR 300.301(c)(1)(i)",
        "34 CFR 300.301(c)(1)(ii)",
        "34 CFR 300.303(a)(1)-(2)",
        "34 CFR 300.303(b)(1)",
        "34 CFR 300.303(b)(2)",
        "34 CFR 300.322(a)(1)-(2) and (b)(1)(i)",
        "34 CFR 300.322(f)",
        "34 CFR 300.323(c)(1)",
        "34 CFR 300.503(a)(1)-(2)",
        "34 CFR 300.503(b)(1)-(7)",
        "34 CFR 300.503(c)(1)-(2)",
        "34 CFR 300.504(a)(1)-(4)",
        # Remedies and dispute resolution.
        "20 U.S.C. 1415(f)(3)(E)(i)-(ii)",
        "34 CFR 300.151(b)(1)-(2)",
        "34 CFR 300.152(a) and (b)(1)",
        "34 CFR 300.153(a) and (b)",
        "34 CFR 300.153(c) and (d)",
        "34 CFR 300.506",
        "34 CFR 300.507(a)(2)",
        "34 CFR 300.507(b)",
        "34 CFR 300.510(a)(1) and (b)(1)",
        "34 CFR 300.513(a)(2)(i)-(iii)",
        "34 CFR 300.513(a)(2)(ii)",
        "34 CFR 300.515(a)",
        # Case and agency authority. Held for validation only; no compiled
        # letter places any of these in its legal_basis.
        "Amanda J. v. Clark County School District, 267 F.3d 877 (9th Cir. 2001)",
        "Endrew F. v. Douglas Cnty. Sch. Dist. RE-1, 580 U.S. 386, 399 (2017)",
        "Gonzaga University v. Doe, 536 U.S. 273 (2002)",
        "Houston Indep. Sch. Dist. v. Bobby R., 200 F.3d 341, 349 (5th Cir. 2000)",
        "L.J. ex rel. N.N.J. v. Sch. Bd. of Broward Cnty., 927 F.3d 1203, 1211-13 (11th Cir. 2019)",
        "Letter to Clarke, 48 IDELR 77 (OSEP Mar. 8, 2007)",
        "M.C. v. Central Regional School District, 81 F.3d 389, 397 (3d Cir. 1996)",
        "Reid ex rel. Reid v. District of Columbia, 401 F.3d 516 (D.C. Cir. 2005)",
        "Schaffer ex rel. Schaffer v. Weast, 546 U.S. 49 (2005)",
        "Sumter Cnty. Sch. Dist. 17 v. Heffernan ex rel. TH, 642 F.3d 478, 484 (4th Cir. 2011)",
        "Van Duyn ex rel. Van Duyn v. Baker Sch. Dist. 5J, 502 F.3d 811, 822 (9th Cir. 2007)",
    }
)

# The legal_basis each compiler must emit, in order, written out here rather
# than read back from the letter. Drift in either direction fails.
EXPECTED_RECORDS_REQUEST_BASIS = [
    "34 CFR 300.613(a)",
    "34 CFR 300.613(b)(1)-(3)",
    "34 CFR 300.611(b)",
    "34 CFR 300.611(c)",
    "34 CFR 300.614",
    "34 CFR 300.616",
    "34 CFR 300.617(a) and (b)",
    "34 CFR 300.501(a)",
    "34 CFR 99.3 (definition of 'Education records'), para. (a)(1)-(2)",
    "34 CFR 99.10(a)",
    "34 CFR 99.10(b)",
    "34 CFR 99.10(c)",
    "34 CFR 99.10(d)",
    "34 CFR 99.10(e)",
    "34 CFR 99.11(a) and (b)",
]

EXPECTED_SHORTFALL_NOTICE_BASIS = [
    "34 CFR 300.17(d)",
    "20 U.S.C. 1401(9)(D)",
    "34 CFR 300.101(a)",
    "34 CFR 300.323(a)",
    "34 CFR 300.323(c)(2)",
    "34 CFR 300.320(a)(7)",
    "34 CFR 300.320(a)(4)",
    "34 CFR 300.323(d)(1)-(2)",
    "34 CFR 300.324(a)(4)(i)",
    "34 CFR 300.324(a)(6)",
    "34 CFR 300.503(a)(1)-(2)",
    "34 CFR 300.613(a)",
]

EXPECTED_COMPENSATORY_BASIS = [
    "34 CFR 300.323(c)(2)",
    "34 CFR 300.17(d)",
    "20 U.S.C. 1401(9)(D)",
    "34 CFR 300.320(a)(7)",
    "34 CFR 300.320(a)(3)(i)-(ii)",
    "34 CFR 300.324(a)(4)(i)",
    "34 CFR 300.324(a)(6)",
    "34 CFR 300.324(b)(1)(ii)(A)",
    "34 CFR 300.503(a)(1)-(2)",
    "34 CFR 300.151(b)(1)-(2)",
    "34 CFR 300.613(a)",
]

EXPECTED_ANNUAL_REVIEW_BASIS = [
    "34 CFR 300.324(b)(1)(i)",
    "20 U.S.C. 1414(d)(4)(A)(i)",
    "34 CFR 300.323(a)",
    "34 CFR 300.322(a)(1)-(2) and (b)(1)(i)",
    "34 CFR 300.503(a)(1)-(2)",
]


# ---------------------------------------------------------------------------
# Synthetic fixtures. Maya R. is fictional, as is every school named here.
# ---------------------------------------------------------------------------


def _ledger() -> IEPLedger:
    return IEPLedger(
        student_alias="Maya R.",
        school_year="2026-2027",
        iep_date=date(2026, 9, 1),
        obligations=[
            ServiceObligation(
                service="Speech-Language Therapy",
                minutes_per_session=30,
                sessions_per_period=2,
                period=Period.WEEK,
                provider_role="Licensed Speech-Language Pathologist",
                setting="therapy room",
                start_date=date(2026, 9, 8),
                end_date=date(2027, 6, 11),
                source_quote=(
                    "Speech-Language Therapy: 30 minutes per session, 2 sessions per week, provided by a "
                    "Licensed Speech-Language Pathologist in the therapy room."
                ),
            ),
            ServiceObligation(
                service="Occupational Therapy",
                minutes_per_session=45,
                sessions_per_period=1,
                period=Period.WEEK,
                provider_role="Licensed Occupational Therapist",
                setting="OT room",
                start_date=date(2026, 9, 8),
                end_date=date(2027, 6, 11),
                source_quote=(
                    "Occupational Therapy: 45 minutes per session, 1 session per week, provided by a "
                    "Licensed Occupational Therapist in the OT room."
                ),
            ),
        ],
        deadlines=[
            Deadline(
                kind=DeadlineKind.ANNUAL_REVIEW,
                due=date(2027, 9, 1),
                description="IEP team reconvenes for the annual review",
                source_quote="The IEP team will reconvene for the annual review no later than September 1, 2027.",
            ),
            Deadline(
                kind=DeadlineKind.PROGRESS_REPORT,
                due=date(2026, 11, 6),
                description="Progress report on all goals provided to parents",
                source_quote=(
                    "Progress reports on all goals will be provided to the parents at each grading period: "
                    "November 6, 2026; January 29, 2027; April 9, 2027; and June 11, 2027."
                ),
            ),
        ],
        accommodations=[
            Accommodation(
                description="Scheduled sensory breaks every 45 minutes during instruction",
                source_quote="Scheduled sensory breaks every 45 minutes during instruction.",
            )
        ],
    )


def _school_ref(source: str = "log-2026-10-31") -> EvidenceRef:
    return EvidenceRef(
        provenance=Provenance.SCHOOL_CONFIRMED,
        source=source,
        detail="District service log produced 2026-10-31 records 12 speech sessions of 30 minutes.",
    )


def _parent_ref(source: str = "parentlog-2026-10") -> EvidenceRef:
    return EvidenceRef(
        provenance=Provenance.PARENT_OBSERVED,
        source=source,
        detail="10/14 - Maya said no speech today again",
    )


def _silence_ref(source: str = "req-001") -> EvidenceRef:
    return EvidenceRef(
        provenance=Provenance.DOCUMENTED_SILENCE,
        source=source,
        detail="Records request req-001 sent 2026-09-21, response due 2026-11-05, no response received.",
    )


def _shortfall(**overrides) -> ServiceShortfall:
    base = dict(
        service="Speech-Language Therapy",
        period_start=date(2026, 9, 8),
        period_end=date(2026, 10, 30),
        owed_minutes=480,
        delivered_minutes=360,
        shortfall_minutes=120,
        school_confirmed_minutes=300,
        parent_observed_minutes=60,
        undocumented_minutes=120,
        evidence=[_school_ref(), _parent_ref(), _silence_ref()],
    )
    base.update(overrides)
    return ServiceShortfall(**base)


def _result(**overrides) -> ReconciliationResult:
    base = dict(
        period_start=date(2026, 9, 8),
        period_end=date(2026, 10, 30),
        shortfalls=[_shortfall()],
        events_considered=31,
    )
    base.update(overrides)
    return ReconciliationResult(**base)


def _clean_result() -> ReconciliationResult:
    """A period in which everything owed is documented as delivered."""
    return _result(
        shortfalls=[
            _shortfall(
                owed_minutes=480,
                delivered_minutes=480,
                shortfall_minutes=0,
                school_confirmed_minutes=480,
                parent_observed_minutes=0,
                undocumented_minutes=0,
                evidence=[_school_ref()],
            )
        ]
    )


def _excused_result() -> ReconciliationResult:
    """A period holding one non-delivery the records blame on the child's absence.

    Built by running the real reconciliation over synthetic events rather than
    by hand: the letter has to find the absence footnotes that
    :mod:`minutes.reconcile` actually writes, and a hand-typed
    ``ServiceShortfall`` would test the letter against a fixture instead of
    against the module that feeds it.

    Hand-computed: 2026-09-08 to 2026-10-30 is 39 school days, so weekly OT
    owes 7 sessions of 45 minutes = 315. Four were held (180) and the fifth
    fell on 2026-10-14, which both the provider and the parent record as a day
    Maya was out: 45 excused, leaving a difference of 90.
    """
    def occupational(day: date, **overrides) -> ServiceEvent:
        base = dict(
            event_date=day,
            service="Occupational Therapy",
            minutes=45,
            delivered=True,
            provenance=Provenance.SCHOOL_CONFIRMED,
            source=f"svclog-{day.isoformat()}",
        )
        base.update(overrides)
        return ServiceEvent(**base)

    absence = dict(minutes=0, delivered=False, attribution=Attribution.STUDENT_ABSENCE)
    events = [
        occupational(date(2026, 9, 15)),
        occupational(date(2026, 9, 22)),
        occupational(date(2026, 9, 29)),
        occupational(date(2026, 10, 6)),
        occupational(date(2026, 10, 14), source="email-2026-10-14-01", **absence),
        occupational(
            date(2026, 10, 14),
            source="plog-2026-10-14-01",
            provenance=Provenance.PARENT_OBSERVED,
            **absence,
        ),
    ]
    return reconcile(_ledger(), events, date(2026, 9, 8), date(2026, 10, 30))


def _fully_excused_result() -> ReconciliationResult:
    """Two weeks in which the child was away for every session the IEP promised.

    Hand-computed: 2026-09-08 to 2026-09-18 is 9 school days, so speech owes
    3 sessions of 30 minutes (90) and OT 1 session of 45. Every one of them
    falls on a date the district's own email records as an absence, so 135
    minutes are excused and nothing is left short — which is not the same
    thing as a period that reconciles, and the letters must not say it is.
    """
    def away(day: date, service: str) -> ServiceEvent:
        return ServiceEvent(
            event_date=day,
            service=service,
            minutes=0,
            delivered=False,
            provenance=Provenance.SCHOOL_CONFIRMED,
            source=f"email-{day.isoformat()}-{service[:2].lower()}",
            attribution=Attribution.STUDENT_ABSENCE,
        )

    speech_days = [date(2026, 9, 8), date(2026, 9, 10), date(2026, 9, 15)]
    events = [away(day, "Speech-Language Therapy") for day in speech_days]
    events.append(away(date(2026, 9, 8), "Occupational Therapy"))
    return reconcile(_ledger(), events, date(2026, 9, 8), date(2026, 9, 18))


def _records_request(**overrides) -> RecordsRequest:
    base = dict(
        request_id="req-001",
        covers_start=date(2026, 9, 8),
        covers_end=date(2026, 10, 30),
        services=["Speech-Language Therapy", "Occupational Therapy"],
        state=RequestState.DRAFT,
    )
    base.update(overrides)
    return RecordsRequest(**base)


def _overdue_request() -> RecordsRequest:
    return _records_request(
        state=RequestState.UNANSWERED_OVERDUE,
        sent_on=date(2026, 9, 21),
        response_due=date(2026, 11, 5),
    )


def _deadline(days_from_today: int, kind: DeadlineKind = DeadlineKind.ANNUAL_REVIEW) -> Deadline:
    """A deadline whose due date really is ``days_from_today`` from TODAY."""
    return Deadline(
        kind=kind,
        due=TODAY + timedelta(days=days_from_today),
        description="IEP team reconvenes for the annual review",
        source_quote=(
            "The IEP team will reconvene for the annual review no later than "
            f"{(TODAY + timedelta(days=days_from_today)).isoformat()}."
        ),
    )


def _status(state: DeadlineState, days: int, met_on: date | None = None) -> DeadlineStatus:
    """A status that is arithmetically possible.

    ``days_remaining`` and the deadline's own due date must agree, or the
    fixture certifies a self-contradictory letter as sound — which is the very
    property under test. Every case here derives the due date from ``days``.
    """
    return DeadlineStatus(
        deadline=_deadline(days),
        state=state,
        days_remaining=days,
        met_on=met_on,
    )


def _all_letters() -> list[Letter]:
    ledger = _ledger()
    return [
        compile_records_request(ledger, _records_request(), today=TODAY),
        compile_records_request(ledger, _overdue_request(), today=TODAY),
        compile_shortfall_notice(ledger, _result(), today=TODAY),
        compile_compensatory_request(ledger, _result(), [_overdue_request()], today=TODAY),
        compile_deadline_reminder(ledger, _status(DeadlineState.OVERDUE, -30), today=TODAY),
    ]


# ---------------------------------------------------------------------------
# Every compiled letter is sound
# ---------------------------------------------------------------------------


def test_records_request_validates():
    letter = compile_records_request(_ledger(), _records_request(), today=TODAY)
    assert validate_letter(letter) == []
    assert letter.kind is LetterKind.RECORDS_REQUEST
    assert letter.citations


def test_shortfall_notice_validates():
    letter = compile_shortfall_notice(_ledger(), _result(), today=TODAY)
    assert validate_letter(letter) == []
    assert letter.kind is LetterKind.SHORTFALL_NOTICE


def test_compensatory_request_validates():
    letter = compile_compensatory_request(_ledger(), _result(), [_overdue_request()], today=TODAY)
    assert validate_letter(letter) == []
    assert letter.kind is LetterKind.COMPENSATORY_REQUEST


@pytest.mark.parametrize(
    ("state", "days", "met_on"),
    [
        (DeadlineState.UPCOMING, 120, None),
        (DeadlineState.DUE_SOON, 9, None),
        (DeadlineState.OVERDUE, -14, None),
        (DeadlineState.MET, -3, date(2026, 10, 29)),
    ],
)
def test_deadline_reminder_validates(state, days, met_on):
    letter = compile_deadline_reminder(_ledger(), _status(state, days, met_on), today=TODAY)
    assert validate_letter(letter) == []
    assert letter.kind is LetterKind.DEADLINE_REMINDER


@pytest.mark.parametrize("kind", list(DeadlineKind))
def test_deadline_reminder_validates_for_every_deadline_kind(kind):
    ledger = _ledger()
    status = DeadlineStatus(
        deadline=Deadline(
            kind=kind,
            due=date(2026, 10, 29),  # four days before TODAY, so days_remaining is real
            description="Progress report on all goals provided to parents",
            source_quote="Progress reports on all goals will be provided to the parents at each grading period.",
        ),
        state=DeadlineState.OVERDUE,
        days_remaining=-4,
    )
    letter = compile_deadline_reminder(ledger, status, today=TODAY)
    assert validate_letter(letter) == []
    assert "4 days in the past" in letter.body


def test_every_letter_carries_a_disclaimer_and_never_advises():
    for letter in _all_letters():
        assert "not a substitute for the advice of an attorney" in letter.disclaimer
        assert "does not give legal advice" in letter.disclaimer
        assert "you should" not in letter.body.lower()
        assert "we recommend" not in letter.body.lower()


def test_markers_are_unique_sequential_and_all_present_in_the_body():
    for letter in _all_letters():
        markers = [c.marker for c in letter.citations]
        assert markers == [f"[{i}]" for i in range(1, len(markers) + 1)]
        for citation in letter.citations:
            assert citation.marker in letter.body
            assert citation.evidence.detail.strip()


def test_the_citation_table_matches_the_hand_transcribed_one():
    """The one citation assertion that is not the module grading its own work.

    `set(letter.legal_basis) <= ALLOWED_CITATIONS` can never fail: both sides
    come from letters.py, so a section number fabricated straight into the
    table would sail through. This pins the table to a list transcribed by
    hand, so the table cannot move without a reviewable edit on both sides.
    """
    assert ALLOWED_CITATIONS == EXPECTED_CITATION_TABLE
    assert len(EXPECTED_CITATION_TABLE) == 77


def test_each_compiler_emits_exactly_the_legal_basis_it_should():
    ledger = _ledger()

    assert (
        compile_records_request(ledger, _records_request(), today=TODAY).legal_basis
        == EXPECTED_RECORDS_REQUEST_BASIS
    )
    assert (
        compile_shortfall_notice(ledger, _result(), today=TODAY).legal_basis
        == EXPECTED_SHORTFALL_NOTICE_BASIS
    )
    assert (
        compile_compensatory_request(ledger, _result(), [_overdue_request()], today=TODAY).legal_basis
        == EXPECTED_COMPENSATORY_BASIS
    )
    assert (
        compile_deadline_reminder(ledger, _status(DeadlineState.OVERDUE, -30), today=TODAY).legal_basis
        == EXPECTED_ANNUAL_REVIEW_BASIS
    )


def test_no_compiled_letter_reaches_for_case_authority():
    """Characterizing this child's facts against a materiality standard is an
    adjudicative call. The case citations exist so validate_letter can reject
    the traps, not so a compiled letter can lean on them."""
    for letter in _all_letters():
        assert letter.legal_basis
        for entry in letter.legal_basis:
            assert entry.startswith(("34 CFR ", "20 U.S.C. ")), entry


def test_the_only_unfootnoted_numbers_are_the_parents_own_arithmetic():
    """The module's invariant, checked rather than asserted in a docstring.

    Three sentences in a reconciliation carry no footnote: the owed-minutes
    arithmetic, the undocumented count, and the per-period total. Each is the
    parent's own count over figures cited above it — true about the state of
    the parent's records whatever the district's logs say — and each is
    recorded on the draft, so the set of unmarked sentences is enumerable
    rather than incidental.
    """
    draft = _Draft()
    _reconciliation_blocks(draft, _ledger(), _result())

    assert len(draft.derived_lines) == 3
    body = draft.body()
    for line in draft.derived_lines:
        assert line in body
        assert "[" not in line  # no marker, because it cites nothing
        assert not any(phrase in line.lower() for phrase in ACCUSATORY_PHRASES)
    assert draft.derived_lines[0].startswith("On that basis our reconciliation counts 480 minutes owed")
    assert draft.derived_lines[1].startswith("We count 120 minutes as undocumented")
    assert draft.derived_lines[2].startswith("For this period: 480 minutes owed")

    # Everything else that made a claim about the district went through cite().
    for citation in draft.citations:
        assert citation.marker in body
        assert citation.claim in body


def test_letters_are_compiled_not_generated():
    """No LLM may touch a letter: there is no step where prose could be
    invented around a fact the ledger does not hold."""
    source = (Path(__file__).resolve().parents[1] / "minutes" / "letters.py").read_text(encoding="utf-8")
    for forbidden in ("strands", "BedrockModel", "boto3", "structured_output", "import openai"):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# The claim rules
# ---------------------------------------------------------------------------


def test_parent_observed_minutes_are_attributed_not_asserted():
    letter = compile_shortfall_notice(_ledger(), _result(), today=TODAY)

    assert "We recorded at home a further 60 minutes" in letter.body
    assert "that is our observation and not a district record" in letter.body
    # The parent's 60 minutes are never folded into the district's figure.
    assert "District records account for 60 minutes" not in letter.body

    parent_claims = [c.claim for c in letter.citations if c.evidence.provenance is Provenance.PARENT_OBSERVED]
    assert parent_claims
    for claim in parent_claims:
        assert any(phrase in claim.lower() for phrase in PARENT_ATTRIBUTION_PHRASES)


def test_undocumented_minutes_are_never_described_as_missed():
    letter = compile_shortfall_notice(_ledger(), _result(), today=TODAY)
    body = letter.body.lower()

    assert "120 minutes as undocumented" in letter.body
    assert "describes the state of our records rather than what took place at school" in letter.body
    for phrase in ACCUSATORY_PHRASES:
        assert phrase not in body


def test_minutes_lost_to_a_recorded_absence_are_stated_and_left_out_of_the_difference():
    """The sentence that keeps the rest of the letter believable.

    A district that can answer one line of a compensatory request with "she
    was not in school that day" has been handed a reason to doubt every other
    line. So the 45 minutes come out of the difference and the letter says so,
    with the records that put them there footnoted beside it.
    """
    letter = compile_compensatory_request(_ledger(), _excused_result(), [], today=TODAY)

    assert (
        "45 minutes are excluded from the difference below as minutes owed on a date noted as "
        "an absence, rather than asked of the district." in letter.body
    )
    assert (
        "For this period: 315 minutes owed, 180 minutes documented as delivered, 45 minutes "
        "excluded as falling on a noted absence, a difference of 90 minutes." in letter.body
    )
    assert validate_letter(letter) == []


def test_the_absence_sentences_are_footnoted_to_the_records_that_note_the_absence():
    letter = compile_compensatory_request(_ledger(), _excused_result(), [], today=TODAY)

    absence_claims = [c for c in letter.citations if "absent on 2026-10-14" in c.claim]
    assert len(absence_claims) == 2  # the district's record and the parent's, stated apart
    for citation in absence_claims:
        assert citation.marker in letter.body
        assert "absent" in citation.evidence.detail


def test_the_absence_sentence_names_the_date_instead_of_a_bare_plural():
    """One record, one date. "absent on dates in this period", said twice about
    the same day, lets a district infer more absences than the records hold —
    and the dates are in the footnotes already, so naming them costs nothing
    and is unarguable."""
    letter = compile_compensatory_request(_ledger(), _excused_result(), [], today=TODAY)

    assert "District records note Maya R. as absent on 2026-10-14." in letter.body
    assert "absent on dates in this period" not in letter.body
    assert "a date noted as an absence" in letter.body  # singular, because there is one


def test_no_sentence_in_a_letter_claims_a_session_was_scheduled():
    """The ledger holds no schedule, so no footnote can support one.

    "absent on dates when Occupational Therapy was scheduled" reads as though
    the records establish that a session was due that day. Nothing in the
    ledger or the evidence does, which makes it exactly the unsourced claim
    this module exists to make impossible — and validate_letter cannot catch
    it, because the sentence is properly footnoted to a real record that
    simply does not say that.
    """
    for compiled in (
        compile_compensatory_request(_ledger(), _excused_result(), [], today=TODAY),
        compile_shortfall_notice(_ledger(), _excused_result(), today=TODAY),
    ):
        assert "scheduled" not in compiled.body
        for citation in compiled.citations:
            if ABSENCE_NOTE in citation.evidence.detail:
                assert "scheduled" not in citation.claim


def test_an_absence_record_is_never_cited_as_proof_of_a_delivery():
    """A record saying the session did not happen cannot source the delivered
    figure. It is the record the letter has just used to give minutes up."""
    letter = compile_compensatory_request(_ledger(), _excused_result(), [], today=TODAY)

    delivered_claims = [
        c for c in letter.citations if "of those minutes as delivered" in c.claim
    ]
    assert delivered_claims
    for citation in delivered_claims:
        assert ABSENCE_NOTE not in citation.evidence.detail
        assert "recorded as not delivered" not in citation.evidence.detail
        assert "recorded as delivered" in citation.evidence.detail


def test_the_parents_own_note_of_an_absence_stays_the_parents():
    """Parent-observed evidence is attributed wherever it appears, including
    where it is being used against the family's own claim."""
    letter = compile_compensatory_request(_ledger(), _excused_result(), [], today=TODAY)

    assert "We recorded at home that Maya R. was absent" in letter.body
    parent_absence = [
        c
        for c in letter.citations
        if c.evidence.provenance is Provenance.PARENT_OBSERVED
        and "absent on 2026-10-14" in c.claim
    ]
    assert parent_absence
    for citation in parent_absence:
        assert any(phrase in citation.claim.lower() for phrase in PARENT_ATTRIBUTION_PHRASES)


def test_the_claimed_total_names_every_term_it_was_computed_from():
    """765 minus 180 is not 540 unless the 45 excused minutes are named too.

    A district that checks the summary figure against the per-service
    paragraphs above it must not find a 45-minute discrepancy: that is the
    "handed a reason to doubt every other line" failure, arriving through the
    one sentence a reader checks first.
    """
    letter = compile_compensatory_request(_ledger(), _excused_result(), [], today=TODAY)

    assert (
        "Across the services above, the IEP provides 765 minutes for this period, 180 minutes "
        "are documented as delivered, and 45 minutes fall on dates a record notes Maya R. was "
        "absent and are excluded, leaving an arithmetic difference of 540 minutes." in letter.body
    )
    assert "585 minutes" not in letter.body
    assert validate_letter(letter) == []


def test_the_shortfall_notice_summary_names_every_term_too():
    notice = compile_shortfall_notice(_ledger(), _excused_result(), today=TODAY)

    assert (
        "Across the services above, the IEP provides 765 minutes for this period, 180 minutes "
        "are documented as delivered, and 45 minutes fall on dates a record notes Maya R. was "
        "absent and are excluded, leaving a difference of 540 minutes for this period."
        in notice.body
    )
    assert validate_letter(notice) == []


def test_an_absence_record_is_never_what_makes_a_gap_escalate():
    """The lever on a compensatory demand may not be the child's absence.

    The same district record, read two ways. Unattributed it documents part of
    the gap and the ask is legitimate. Attributed to Maya's own absence,
    reconciliation takes those minutes OUT of the difference — and what is
    left is 420 minutes nobody recorded either way. Counting the absence
    record as the evidence for that remainder would ask a district to convene
    a team over minutes the family holds no record of, on the strength of a
    record it has just used to give minutes up.
    """
    def speech(attribution: Attribution) -> ReconciliationResult:
        event = ServiceEvent(
            event_date=date(2026, 9, 24),
            service="Speech-Language Therapy",
            minutes=0,
            delivered=False,
            provenance=Provenance.SCHOOL_CONFIRMED,
            source="email-2026-09-24-01",
            attribution=attribution,
        )
        return reconcile(_ledger(), [event], date(2026, 9, 8), date(2026, 10, 30))

    unattributed = speech(Attribution.SCHOOL_OR_UNRECORDED)
    line = unattributed.shortfalls[0]
    assert (line.shortfall_minutes, line.undocumented_minutes) == (450, 420)
    asked = compile_compensatory_request(_ledger(), unattributed, [], today=TODAY)
    assert "Request for an IEP team meeting to consider compensatory services" in asked.subject

    excused = speech(Attribution.STUDENT_ABSENCE)
    line = excused.shortfalls[0]
    assert (line.excused_minutes, line.shortfall_minutes, line.undocumented_minutes) == (30, 420, 420)
    not_asked = compile_compensatory_request(_ledger(), excused, [], today=TODAY)
    assert "no compensatory services requested" in not_asked.subject
    assert "I am not requesting compensatory services on the strength of it" in not_asked.body
    assert "Please offer meeting dates" not in not_asked.body
    assert validate_letter(not_asked) == []


def test_a_period_excused_to_zero_never_says_the_records_reconcile():
    """Nothing was delivered. The letters may not report that as agreement.

    "The records we hold reconcile with the minutes the IEP provides" and "I am
    not raising a concern about delivery for this period" are both false of a
    period in which none of the promise was delivered, and both are a written
    concession sitting in the district's file. The three figures are stated
    instead.
    """
    result = _fully_excused_result()
    assert sum(line.shortfall_minutes for line in result.shortfalls) == 0
    assert sum(line.excused_minutes for line in result.shortfalls) == 135

    notice = compile_shortfall_notice(_ledger(), result, today=TODAY)
    comp = compile_compensatory_request(_ledger(), result, [], today=TODAY)

    for compiled in (notice, comp):
        assert "reconcile with the minutes the IEP provides" not in compiled.body
        assert "I am not raising a concern about delivery" not in compiled.body
        assert (
            "the IEP provides 135 minutes for this period, 0 minutes are documented as "
            "delivered, and 135 minutes fall on dates a record notes Maya R. was absent and "
            "are excluded" in compiled.body
        )
        assert validate_letter(compiled) == []

    assert "I am not requesting compensatory services" in comp.body
    assert "no compensatory services requested" in comp.subject


def test_the_exclusion_claim_disappears_with_the_records_under_it():
    """The one claim in this module that is about the CHILD.

    A line can carry excused minutes whose absence records are not on it —
    ``ServiceShortfall`` is a frozen contract other code fills in. The two
    cited sentences drop out on their own, and the exclusion sentence has to
    go with them: it carries no marker, so validate_letter would never see it
    standing in the body as an unfootnoted statement that a child was away.
    """
    orphaned = _result(
        shortfalls=[
            _shortfall(
                owed_minutes=480,
                delivered_minutes=240,
                excused_minutes=120,
                shortfall_minutes=120,
                school_confirmed_minutes=240,
                parent_observed_minutes=0,
                undocumented_minutes=120,
                evidence=[_silence_ref()],  # no absence record anywhere on the line
            )
        ]
    )
    draft = _Draft()
    _reconciliation_blocks(draft, _ledger(), orphaned)

    assert "absent" not in draft.body()
    assert not [line for line in draft.derived_lines if "noted as an absence" in line]
    assert not [c for c in draft.citations if "absent" in c.claim]

    letter = compile_shortfall_notice(_ledger(), orphaned, today=TODAY)
    assert "excluded from the difference below" not in letter.body
    assert validate_letter(letter) == []


def test_the_shortfall_notice_states_the_exclusion_too():
    """Both letters that carry a reconciliation carry the same subtraction."""
    notice = compile_shortfall_notice(_ledger(), _excused_result(), today=TODAY)

    assert "45 minutes excluded as falling on a noted absence" in notice.body
    assert validate_letter(notice) == []


def test_the_exclusion_is_the_parents_own_arithmetic_and_carries_no_marker():
    """The claim about the district is the absence record, and it is cited.
    The exclusion itself is the parent's own count over that record, so it is
    recorded on the draft rather than footnoted — the same treatment the owed
    and undocumented counts get."""
    draft = _Draft()
    _reconciliation_blocks(draft, _ledger(), _excused_result())

    # Three unmarked sentences on the speech line, four on the OT line: the
    # extra one is the exclusion.
    assert len(draft.derived_lines) == 7
    exclusion = [line for line in draft.derived_lines if "noted as an absence" in line]
    assert len(exclusion) == 1
    assert "[" not in exclusion[0]
    assert exclusion[0] in draft.body()


def test_a_period_with_no_recorded_absence_says_nothing_about_absence():
    """The sentence appears where it is true and nowhere else."""
    for letter in _all_letters():
        assert "noted as an absence" not in letter.body
        assert "was absent" not in letter.body


def test_no_compiled_letter_uses_accusatory_vocabulary():
    for letter in _all_letters():
        body = letter.body.lower()
        for phrase in ACCUSATORY_PHRASES:
            assert phrase not in body, f"{letter.kind.value} body contains {phrase!r}"


def test_a_claim_with_no_evidence_of_its_grade_is_omitted_entirely():
    """School-confirmed minutes with no school-confirmed evidence are not
    asserted in softer words — the sentence never appears."""
    result = _result(
        shortfalls=[
            _shortfall(
                school_confirmed_minutes=300,
                parent_observed_minutes=0,
                undocumented_minutes=180,
                evidence=[_parent_ref()],  # nothing from the district at all
            )
        ]
    )
    letter = compile_shortfall_notice(_ledger(), result, today=TODAY)

    assert "District records account for" not in letter.body
    assert "have no delivery record from the district" not in letter.body
    # The IEP provision and the undocumented note survive, because both are
    # grounded: one in the IEP quote, one in the parent's own ledger.
    assert "The IEP dated 2026-09-01 provides Speech-Language Therapy" in letter.body
    assert "180 minutes as undocumented" in letter.body
    assert validate_letter(letter) == []


def test_a_service_absent_from_the_ledger_is_left_out_rather_than_described():
    result = _result(
        shortfalls=[
            _shortfall(),
            _shortfall(service="Physical Therapy", owed_minutes=900, shortfall_minutes=900),
        ]
    )
    letter = compile_shortfall_notice(_ledger(), result, today=TODAY)

    assert "no matching provision in the IEP on file" in letter.body
    assert "Physical Therapy" in letter.body  # named, but not quantified
    assert "900 minutes" not in letter.body
    assert validate_letter(letter) == []


# ---------------------------------------------------------------------------
# A provision may only be quoted for a period its own term covers
# ---------------------------------------------------------------------------

# The reconciliation under test runs 2026-09-08 to 2026-10-30 throughout.
PERIOD_START = date(2026, 9, 8)
PERIOD_END = date(2026, 10, 30)


def _ledger_with_term(start: date, end: date) -> IEPLedger:
    """The standard ledger with the speech provision's term moved."""
    ledger = _ledger()
    speech = ledger.obligations[0]
    ledger.obligations = [speech.model_copy(update={"start_date": start, "end_date": end})]
    return ledger


@pytest.mark.parametrize(
    ("start", "end", "covers"),
    [
        (date(2025, 9, 8), date(2027, 6, 11), True),  # straddles the period
        (date(2025, 9, 8), PERIOD_START, True),  # ends exactly on the first day
        (date(2025, 9, 8), PERIOD_START - timedelta(days=1), False),  # ends one day before
        (PERIOD_END, date(2027, 6, 11), True),  # starts exactly on the last day
        (PERIOD_END + timedelta(days=1), date(2027, 6, 11), False),  # starts one day after
        (date(2025, 9, 8), date(2026, 6, 11), False),  # expired last school year
        (date(2027, 1, 1), date(2027, 6, 11), False),  # not yet begun
    ],
    ids=[
        "straddles",
        "ends-on-period-start",
        "ends-day-before-period-start",
        "starts-on-period-end",
        "starts-day-after-period-end",
        "expired",
        "not-yet-begun",
    ],
)
def test_a_provision_is_quoted_only_for_a_period_its_term_covers(start, end, covers):
    """An out-of-term provision is not a basis for minutes owed.

    Quoting an expired or not-yet-started IEP line and then counting minutes
    against it overstates the case against the school on the strength of a
    document that did not govern the period — the exact failure this product
    exists to prevent. Out of term, the service is named and set aside.
    """
    ledger = _ledger_with_term(start, end)
    notice = compile_shortfall_notice(ledger, _result(), today=TODAY)

    if covers:
        assert "The IEP dated 2026-09-01 provides Speech-Language Therapy" in notice.body
        assert "480 minutes owed" in notice.body
        assert "is 120 minutes for this period" in notice.body
    else:
        assert "provides Speech-Language Therapy" not in notice.body
        assert "480 minutes owed" not in notice.body
        assert "120 minutes" not in notice.body
        assert "no matching provision in the IEP on file" in notice.body
        assert "Speech-Language Therapy" in notice.body  # named, never quantified
        assert "makes no findings" in notice.body
    assert validate_letter(notice) == []


def test_an_out_of_term_provision_never_becomes_a_compensatory_request():
    """The same leak, one letter downstream: unquotable minutes must not reach
    the summary total, and must not title the letter as a request."""
    ledger = _ledger_with_term(date(2025, 9, 8), date(2026, 6, 11))  # expired
    comp = compile_compensatory_request(ledger, _result(), [_overdue_request()], today=TODAY)

    assert "Request for an IEP team meeting to consider compensatory services" not in comp.subject
    assert "no compensatory services requested" in comp.subject
    assert "makes no request and states no figures" in comp.body
    assert "480" not in comp.body
    assert "120 minutes" not in comp.body
    assert validate_letter(comp) == []


def test_the_records_request_does_not_quote_an_out_of_term_provision():
    ledger = _ledger_with_term(date(2027, 1, 1), date(2027, 6, 11))  # begins after the request window
    letter = compile_records_request(ledger, _records_request(), today=TODAY)

    assert "records no provision for this service covering 2026-09-08 through 2026-10-30" in letter.body
    assert "provides Speech-Language Therapy at 30 minutes" not in letter.body
    assert "Speech-Language Therapy" in letter.body  # still requested, just not quoted
    assert validate_letter(letter) == []


def test_documented_silence_is_cited_as_dated_evidence():
    letter = compile_records_request(_ledger(), _overdue_request(), today=TODAY)

    silence = [c for c in letter.citations if c.evidence.provenance is Provenance.DOCUMENTED_SILENCE]
    assert silence
    assert "2026-09-21" in silence[0].evidence.detail
    assert "no response has been recorded as received" in letter.body
    assert validate_letter(letter) == []


def test_compensatory_request_carries_the_records_request_history():
    answered = _records_request(
        request_id="req-000",
        state=RequestState.ANSWERED,
        sent_on=date(2026, 9, 1),
        answered_on=date(2026, 9, 30),
    )
    draft_only = _records_request(request_id="req-002", state=RequestState.DRAFT)
    letter = compile_compensatory_request(
        _ledger(), _result(), [answered, _overdue_request(), draft_only], today=TODAY
    )

    assert "req-000" in letter.body
    assert "req-001" in letter.body
    # An unsent draft is not evidence of anything and never enters the letter.
    assert "req-002" not in letter.body
    assert validate_letter(letter) == []


# ---------------------------------------------------------------------------
# A period with nothing wrong produces nothing that reads like an accusation
# ---------------------------------------------------------------------------


def test_zero_shortfall_shortfall_notice_makes_no_accusation():
    letter = compile_shortfall_notice(_ledger(), _clean_result(), today=TODAY)

    assert "the records we hold reconcile with the minutes the IEP provides" in letter.body
    assert "I am not raising a concern about delivery for this period" in letter.body
    assert "undocumented" not in letter.body
    for phrase in ACCUSATORY_PHRASES:
        assert phrase not in letter.body.lower()
    assert validate_letter(letter) == []


def test_zero_shortfall_compensatory_request_asks_for_nothing():
    letter = compile_compensatory_request(_ledger(), _clean_result(), [], today=TODAY)

    assert "I am not requesting compensatory services" in letter.body
    # Nothing anywhere — subject line included — may read as a request that
    # the letter does not go on to make.
    assert "consider whether compensatory services are appropriate" not in letter.body
    assert "Please offer meeting dates" not in letter.body
    assert "no compensatory services requested" in letter.subject
    assert validate_letter(letter) == []


def test_a_gap_made_only_of_undocumented_minutes_does_not_ask_for_compensation():
    """The escalation trigger has to see the distinction the whole provenance
    system exists to draw. A gap with no evidence of any grade behind it is a
    hole in the parent's records; the answer to that is to ask the district for
    its records, not to ask the team for compensatory services."""
    result = _result(
        shortfalls=[
            _shortfall(
                owed_minutes=480,
                delivered_minutes=0,
                shortfall_minutes=480,
                school_confirmed_minutes=0,
                parent_observed_minutes=0,
                undocumented_minutes=480,
                evidence=[],
            )
        ]
    )
    comp = compile_compensatory_request(_ledger(), result, [], today=TODAY)

    assert "Request for an IEP team meeting to consider compensatory services" not in comp.subject
    assert "no compensatory services requested" in comp.subject
    assert "I am not requesting compensatory services on the strength of it" in comp.body
    assert "a minute for which I hold no record either way" in comp.body
    assert "Please offer meeting dates" not in comp.body
    # The arithmetic is still stated — it is just not made into a request.
    assert "480 minutes owed" in comp.body
    assert validate_letter(comp) == []


def test_a_documented_silence_behind_the_gap_does_ask_for_compensation():
    """The other side of the same gate: an overdue, unanswered records request
    standing where a service log should be is evidence, and it escalates."""
    result = _result(
        shortfalls=[
            _shortfall(
                owed_minutes=480,
                delivered_minutes=0,
                shortfall_minutes=480,
                school_confirmed_minutes=0,
                parent_observed_minutes=0,
                undocumented_minutes=480,
                evidence=[_silence_ref()],
            )
        ]
    )
    comp = compile_compensatory_request(_ledger(), result, [_overdue_request()], today=TODAY)

    assert "Request for an IEP team meeting to consider compensatory services" in comp.subject
    assert "whether compensatory services are appropriate" in comp.body
    assert validate_letter(comp) == []


def test_met_deadline_reminder_requests_nothing_and_says_so():
    letter = compile_deadline_reminder(
        _ledger(), _status(DeadlineState.MET, -3, date(2026, 10, 29)), today=TODAY
    )

    assert "Our file records this as met on 2026-10-29." in letter.body
    assert "No action is requested." in letter.body
    assert "It makes no request and asserts nothing about the district." in letter.body
    assert "asks what is scheduled" not in letter.body
    assert "prior written notice" not in letter.body
    assert validate_letter(letter) == []


def test_alias_ending_in_an_initial_does_not_double_its_period():
    for letter in _all_letters():
        assert "R.." not in letter.body


def test_a_vowel_initial_provider_role_reads_correctly():
    """Provider roles come out of the IEP verbatim, so the article has to be
    chosen, not hardcoded: 'by a Occupational Therapist' in a letter to a
    district is the kind of thing that costs a parent credibility."""
    ledger = _ledger()
    ledger.obligations = [
        ledger.obligations[0].model_copy(update={"provider_role": "Occupational Therapist"})
    ]
    letter = compile_shortfall_notice(ledger, _result(), today=TODAY)

    assert "delivered by an Occupational Therapist" in letter.body
    assert "by a Occupational" not in letter.body

    ledger.obligations = [
        ledger.obligations[0].model_copy(update={"provider_role": "Licensed Speech-Language Pathologist"})
    ]
    consonant = compile_shortfall_notice(ledger, _result(), today=TODAY)
    assert "delivered by a Licensed Speech-Language Pathologist" in consonant.body


def test_the_disclaimer_keeps_the_regulation_disjunctive():
    """34 CFR 300.507(b) triggers on a parent's request OR a filed complaint.
    Rendering it as a conjunction tells a parent they have no right to the
    information until they file, which is not what the section says."""
    letter = compile_records_request(_ledger(), _records_request(), today=TODAY)

    assert "if you request the information, or if a due process complaint is filed" in letter.disclaimer
    assert "on request, and when a due process complaint is filed" not in letter.disclaimer


def test_the_shortfall_notice_leads_with_the_general_implementation_authority():
    """34 CFR 300.323(c)(2) is headed 'Initial IEPs' and begins 'As soon as
    possible following development of the IEP'. It is the weakest available
    hook for a general implementation obligation, so it follows 300.17(d),
    300.101(a) and 300.323(a) rather than leading, and keeps its qualifier."""
    letter = compile_shortfall_notice(_ledger(), _result(), today=TODAY)

    assert letter.legal_basis[0] == "34 CFR 300.17(d)"
    assert letter.legal_basis.index("34 CFR 300.101(a)") < letter.legal_basis.index(
        "34 CFR 300.323(c)(2)"
    )
    assert letter.legal_basis.index("34 CFR 300.323(a)") < letter.legal_basis.index(
        "34 CFR 300.323(c)(2)"
    )
    assert "as soon as possible following development of the IEP" in letter.body


def test_deadline_reminder_asks_a_question_rather_than_making_a_finding():
    status = _status(DeadlineState.OVERDUE, -30)
    assert status.deadline.due == date(2026, 10, 3)  # TODAY minus 30, really

    letter = compile_deadline_reminder(_ledger(), status, today=TODAY)

    assert "30 days in the past" in letter.body
    assert "confirm in writing whether this has taken place" in letter.body
    assert "It does not assert that any requirement has been broken." in letter.body
    assert validate_letter(letter) == []


# ---------------------------------------------------------------------------
# The day count is a fact about the letter's own dateline
# ---------------------------------------------------------------------------


def test_a_stale_days_remaining_never_reaches_the_letter():
    """A DeadlineStatus is computed at one moment and a letter compiled at
    another. Restating the stale count as a fact about this letter's date is a
    false, school-adverse claim, so the compiler counts from the dateline."""
    stale = DeadlineStatus(
        deadline=_deadline(-4),  # due 2026-10-29, four days before the dateline
        state=DeadlineState.OVERDUE,
        days_remaining=-61,  # computed two months later, and wrong for this letter
    )
    letter = compile_deadline_reminder(_ledger(), stale, today=TODAY)

    assert "4 days in the past" in letter.body
    assert "61 days" not in letter.body
    assert validate_letter(letter) == []


def test_a_state_sign_contradiction_does_not_render_literally():
    """OVERDUE with a positive count, and UPCOMING with a negative one, are
    both arithmetically impossible. Neither may print as written."""
    overdue_but_future = DeadlineStatus(
        deadline=_deadline(30), state=DeadlineState.OVERDUE, days_remaining=30
    )
    letter = compile_deadline_reminder(_ledger(), overdue_but_future, today=TODAY)
    assert "30 days away" in letter.body
    assert "in the past" not in letter.body
    assert validate_letter(letter) == []

    upcoming_but_past = DeadlineStatus(
        deadline=_deadline(-7), state=DeadlineState.UPCOMING, days_remaining=-7
    )
    letter = compile_deadline_reminder(_ledger(), upcoming_but_past, today=TODAY)
    assert "7 days in the past" in letter.body
    assert "-7 days" not in letter.body
    assert "days away" not in letter.body
    assert validate_letter(letter) == []


def test_a_deadline_due_on_the_dateline_is_not_counted_in_either_direction():
    status = DeadlineStatus(deadline=_deadline(0), state=DeadlineState.DUE_SOON, days_remaining=0)
    letter = compile_deadline_reminder(_ledger(), status, today=TODAY)

    assert "That date is the date of this letter." in letter.body
    assert "0 days" not in letter.body
    assert validate_letter(letter) == []


def test_the_day_count_is_singular_at_one_day():
    for days, expected in ((1, "1 day away"), (-1, "1 day in the past")):
        letter = compile_deadline_reminder(_ledger(), _status(DeadlineState.DUE_SOON, days), today=TODAY)
        assert expected in letter.body
        assert "1 days" not in letter.body


# ---------------------------------------------------------------------------
# The adversarial half: each rule, broken on purpose, must be caught
# ---------------------------------------------------------------------------


def _hand_built(**overrides) -> Letter:
    base = dict(
        kind=LetterKind.SHORTFALL_NOTICE,
        subject="Hand-built letter",
        body="The IEP provides 60 minutes of speech therapy per week.[1]",
        citations=[
            LetterCitation(
                marker="[1]",
                claim="The IEP provides 60 minutes of speech therapy per week.",
                evidence=_school_ref("iep-2026-09-01"),
            )
        ],
        legal_basis=["34 CFR 300.323(c)(2)"],
    )
    base.update(overrides)
    return Letter(**base)


def test_hand_built_control_letter_validates():
    assert validate_letter(_hand_built()) == []


def test_dangling_marker_fails_validation():
    letter = _hand_built(
        body="The IEP provides 60 minutes of speech therapy per week.[1] The school cancelled 8 sessions.[2]"
    )
    violations = validate_letter(letter)
    assert any("[2]" in v and "no matching citation" in v for v in violations)


def test_citation_that_never_appears_in_the_body_fails_validation():
    letter = _hand_built(
        citations=_hand_built().citations
        + [
            LetterCitation(
                marker="[2]",
                claim="An orphan footnote.",
                evidence=_school_ref(),
            )
        ]
    )
    violations = validate_letter(letter)
    assert any("[2]" in v and "never appears in the body" in v for v in violations)


def test_duplicate_marker_fails_validation():
    citations = _hand_built().citations
    letter = _hand_built(
        citations=citations
        + [LetterCitation(marker="[1]", claim="A second, different claim.", evidence=_parent_ref())]
    )
    assert any("more than one citation" in v for v in validate_letter(letter))


def test_unapproved_legal_basis_fails_validation():
    letter = _hand_built(legal_basis=["34 CFR 300.323(c)(2)", "34 CFR 300.999(z)"])
    violations = validate_letter(letter)
    assert any("34 CFR 300.999(z)" in v for v in violations)
    assert not any("300.323" in v for v in violations)


def test_the_superseded_van_duyn_reporter_is_rejected():
    """Van Duyn was amended in full and superseded at 502 F.3d 811; secondary
    sources still hand out the vacated 481 F.3d 770. The allowlist is the only
    thing standing between that trap and a parent's letter."""
    trap = "Van Duyn v. Baker Sch. Dist. 5J, 481 F.3d 770 (9th Cir. 2007)"
    assert trap not in ALLOWED_CITATIONS
    assert "Van Duyn ex rel. Van Duyn v. Baker Sch. Dist. 5J, 502 F.3d 811, 822 (9th Cir. 2007)" in ALLOWED_CITATIONS
    assert any(trap in v for v in validate_letter(_hand_built(legal_basis=[trap])))


def test_invented_regulation_in_the_body_fails_validation():
    letter = _hand_built(
        body="The IEP provides 60 minutes of speech therapy per week.[1] 34 CFR 300.999 requires make-up sessions."
    )
    assert any("34 CFR 300.999" in v for v in validate_letter(letter))


@pytest.mark.parametrize(
    ("prose", "expected"),
    [
        ("34 C.F.R. 300.999 requires make-up sessions.", "34 CFR 300.999"),
        ("34 C.F.R. § 300.999 requires make-up sessions.", "34 CFR 300.999"),
        ("34 CFR  300.999 requires make-up sessions.", "34 CFR 300.999"),
        ("34 cfr 300.999 requires make-up sessions.", "34 CFR 300.999"),
        ("Section 300.999 of Title 34 requires make-up sessions.", "34 CFR 300.999"),
        ("20 USC 1499(z) requires make-up sessions.", "20 USC 1499"),
        ("20 U.S.C. 1499(z) requires make-up sessions.", "20 USC 1499"),
    ],
    ids=[
        "periods",
        "section-symbol",
        "double-space",
        "lowercase",
        "section-of-title",
        "usc-unpunctuated",
        "usc-punctuated",
    ],
)
def test_an_invented_section_fails_in_every_spelling(prose, expected):
    """A gate that recognizes one spelling is not a gate. Each of these once
    returned no violations at all."""
    letter = _hand_built(body=f"The IEP provides 60 minutes of speech therapy per week.[1] {prose}")
    assert any(expected in v for v in validate_letter(letter)), prose


@pytest.mark.parametrize(
    "prose",
    [
        "Doe v. Nowhere Sch. Dist., 999 F.3d 1234 (9th Cir. 2099) requires make-up sessions.",
        "Van Duyn v. Baker Sch. Dist. 5J, 481 F.3d 770 (9th Cir. 2007) controls here.",
        "Nobody v. Nowhere, 999 U.S. 1 (2099) controls here.",
        "See Letter to Nobody, 99 IDELR 999 (OSEP 2099).",
    ],
    ids=["fabricated-f3d", "superseded-van-duyn", "fabricated-us", "fabricated-idelr"],
)
def test_case_authority_in_a_body_is_checked_against_the_table_too(prose):
    """The superseded Van Duyn reporter was rejected only in legal_basis. A
    body is where a parent would actually paste one."""
    letter = _hand_built(body=f"The IEP provides 60 minutes of speech therapy per week.[1] {prose}")
    assert any("outside the verified citation table" in v for v in validate_letter(letter)), prose


def test_a_verified_opinion_may_be_pin_cited_in_a_body():
    """The check is at volume and page, so a pin cite into an opinion the table
    holds is fine — it is the fabricated volume that fails."""
    letter = _hand_built(
        body=(
            "The IEP provides 60 minutes of speech therapy per week.[1] "
            "Van Duyn ex rel. Van Duyn v. Baker Sch. Dist. 5J, 502 F.3d 811, 822 (9th Cir. 2007)."
        )
    )
    assert validate_letter(letter) == []


@pytest.mark.parametrize(
    "sentence",
    [
        "The district failed to deliver 480 minutes of speech therapy.",
        "Eight sessions were missed in October.",
        "The provider did not deliver the services in the IEP.",
        "The district refused to provide the records.",
        "Maya was shortchanged 480 minutes this period.",
    ],
)
def test_accusatory_vocabulary_fails_validation(sentence):
    """The product's central promise is enforced by the public gate, not only
    by this file: a Letter another module assembles is held to it too."""
    letter = _hand_built(body=f"The IEP provides 60 minutes of speech therapy per week.[1] {sentence}")
    assert any("accusation" in v for v in validate_letter(letter)), sentence


def test_the_blocklists_are_defense_in_depth_and_say_so():
    """A blocklist loses to paraphrase and this file does not pretend
    otherwise. What holds is that a compiled body sentence about the district
    can only come from cite(), which cannot say more than its EvidenceRef.
    These two are caught because they name the conclusion outright."""
    caught = "The district failed to deliver 480 minutes and the district owes Maya compensatory services."
    assert validate_letter(_hand_built(body=f"Cited.[1] {caught}")) != []

    conclusion = "This is a substantial failure to implement the IEP, which denies my child a free appropriate public education."
    assert validate_letter(_hand_built(body=f"Cited.[1] {conclusion}")) != []

    law = "The school broke federal law."
    assert validate_letter(_hand_built(body=f"Cited.[1] {law}")) != []


# ---------------------------------------------------------------------------
# Blanks only the parent can fill
# ---------------------------------------------------------------------------


def test_the_records_request_reports_its_one_parent_placeholder():
    """Minutes will not invent the family's circumstances, so the letter ships
    with a blank. An unedited letter carrying an instruction-to-self must not
    be able to reach a district unnoticed."""
    letter = compile_records_request(_ledger(), _records_request(), today=TODAY)

    found = placeholders(letter)
    assert len(found) == 1
    assert "circumstance that would prevent on-site inspection" in found[0]

    # Sound as a draft: it invents nothing.
    assert validate_letter(letter) == []
    # Not sendable: the blank is still a blank.
    sendable = validate_letter(letter, for_sending=True)
    assert any("unfilled parent placeholder" in v for v in sendable)


def test_a_filled_placeholder_makes_the_letter_sendable():
    letter = compile_records_request(_ledger(), _records_request(), today=TODAY)
    filled = letter.model_copy(
        update={
            "body": _PLACEHOLDER_SUB.sub(
                "I work weekdays and cannot attend an on-site appointment.", letter.body
            )
        }
    )
    assert placeholders(filled) == []
    assert validate_letter(filled, for_sending=True) == []


def test_letters_without_placeholders_are_sendable_as_compiled():
    for letter in _all_letters():
        if letter.kind is LetterKind.RECORDS_REQUEST:
            continue
        assert placeholders(letter) == []
        assert validate_letter(letter, for_sending=True) == []


def test_unattributed_parent_observation_fails_validation():
    letter = _hand_built(
        body="The school delivered only 4 sessions in October.[1]",
        citations=[
            LetterCitation(
                marker="[1]",
                claim="The school delivered only 4 sessions in October.",
                evidence=_parent_ref(),
            )
        ],
    )
    violations = validate_letter(letter)
    assert any("not attributed" in v for v in violations)


def test_attributed_parent_observation_passes_validation():
    letter = _hand_built(
        body="We recorded at home that speech did not appear to happen on four dates in October.[1]",
        citations=[
            LetterCitation(
                marker="[1]",
                claim="We recorded at home that speech did not appear to happen on four dates in October.",
                evidence=_parent_ref(),
            )
        ],
    )
    assert validate_letter(letter) == []


@pytest.mark.parametrize(
    "sentence",
    [
        "This is a material failure to implement the IEP.",
        "The shortfall is a denial of FAPE.",
        "The district violated the IDEA.",
        "You are entitled to 47 hours of compensatory speech therapy.",
    ],
)
def test_legal_conclusions_fail_validation(sentence):
    letter = _hand_built(body=f"The IEP provides 60 minutes of speech therapy per week.[1] {sentence}")
    assert any("legal conclusion" in v for v in validate_letter(letter))


def test_missing_disclaimer_fails_validation():
    assert any("disclaimer" in v for v in validate_letter(_hand_built(disclaimer="   ")))


# ---------------------------------------------------------------------------
# The invariant, across every shape of evidence a period can hand us
# ---------------------------------------------------------------------------

_REF_FOR_GRADE = {
    Provenance.SCHOOL_CONFIRMED: _school_ref,
    Provenance.PARENT_OBSERVED: _parent_ref,
    Provenance.DOCUMENTED_SILENCE: _silence_ref,
}

_GRADE_COMBINATIONS = [
    combo
    for size in range(len(Provenance) + 1)
    for combo in combinations(list(Provenance), size)
]


@pytest.mark.parametrize("grades", _GRADE_COMBINATIONS, ids=lambda g: "+".join(p.value for p in g) or "no-evidence")
def test_any_combination_of_evidence_grades_compiles_to_a_sound_letter(grades):
    """Whatever the reconciler managed to prove — all three grades, one, or
    nothing at all — the compiled letter still validates. Missing evidence
    removes sentences; it never produces an unsupported one."""
    result = _result(shortfalls=[_shortfall(evidence=[_REF_FOR_GRADE[g]() for g in grades])])
    ledger = _ledger()

    for letter in (
        compile_shortfall_notice(ledger, result, today=TODAY),
        compile_compensatory_request(ledger, result, [_overdue_request()], today=TODAY),
    ):
        assert validate_letter(letter) == []
        # The IEP's own numbers survive regardless; the delivery claims do not.
        assert "480 minutes owed" in letter.body
        if Provenance.SCHOOL_CONFIRMED not in grades:
            assert "District records account for" not in letter.body
        if Provenance.PARENT_OBSERVED not in grades:
            assert "We recorded at home" not in letter.body


@pytest.mark.parametrize(
    "result",
    [
        ReconciliationResult(
            period_start=date(2026, 9, 8), period_end=date(2026, 10, 30), shortfalls=[], events_considered=0
        ),
        ReconciliationResult(
            period_start=date(2026, 9, 8),
            period_end=date(2026, 10, 30),
            shortfalls=[_shortfall(service="Physical Therapy")],
            events_considered=4,
        ),
    ],
    ids=["nothing-reconciled", "only-unmatched-services"],
)
def test_letters_hold_up_when_nothing_can_be_quoted(result):
    ledger = _ledger()
    notice = compile_shortfall_notice(ledger, result, today=TODAY)
    comp = compile_compensatory_request(ledger, result, [], today=TODAY)

    assert validate_letter(notice) == []
    assert validate_letter(comp) == []
    assert "makes no findings" in notice.body
    assert "makes no request and states no figures" in comp.body
    assert "120 minutes" not in notice.body
    assert "120 minutes" not in comp.body


def test_a_gap_the_letter_cannot_quote_is_left_out_of_the_total():
    """The reconciler may count minutes for a service the IEP on file does not
    contain. Those minutes are real, but this letter cannot source them, so
    they must not turn up inside a summary figure."""
    result = _result(
        shortfalls=[
            _shortfall(  # quotable, and fully delivered
                owed_minutes=480,
                delivered_minutes=480,
                shortfall_minutes=0,
                school_confirmed_minutes=480,
                parent_observed_minutes=0,
                undocumented_minutes=0,
                evidence=[_school_ref()],
            ),
            _shortfall(service="Physical Therapy", owed_minutes=900, shortfall_minutes=900),
        ]
    )
    assert result.total_shortfall_minutes == 900

    notice = compile_shortfall_notice(_ledger(), result, today=TODAY)
    comp = compile_compensatory_request(_ledger(), result, [], today=TODAY)

    assert "900" not in notice.body
    assert "900" not in comp.body
    assert "the records we hold reconcile with the minutes the IEP provides" in notice.body
    assert "I am not requesting compensatory services" in comp.body
    assert "no matching provision in the IEP on file" in notice.body
    assert validate_letter(notice) == []
    assert validate_letter(comp) == []


def test_records_request_with_no_named_services_falls_back_to_the_iep():
    letter = compile_records_request(_ledger(), _records_request(services=[]), today=TODAY)
    assert "Speech-Language Therapy" in letter.body
    assert "Occupational Therapy" in letter.body
    assert validate_letter(letter) == []


def test_records_request_without_an_annual_review_on_file_still_validates():
    ledger = _ledger()
    ledger.deadlines = [d for d in ledger.deadlines if d.kind is not DeadlineKind.ANNUAL_REVIEW]
    letter = compile_records_request(ledger, _records_request(), today=TODAY)

    assert "before any meeting regarding an IEP" in letter.body  # the rule still stated
    assert "no later than 2027-09-01" not in letter.body  # the date is not invented
    assert validate_letter(letter) == []


def test_overdue_request_with_no_recorded_dates_does_not_invent_them():
    request = _records_request(state=RequestState.UNANSWERED_OVERDUE)  # sent_on/response_due unset
    letter = compile_records_request(_ledger(), request, today=TODAY)

    assert "a date not recorded" in letter.body
    assert validate_letter(letter) == []


def test_met_deadline_without_a_recorded_date_does_not_invent_one():
    letter = compile_deadline_reminder(_ledger(), _status(DeadlineState.MET, -3, met_on=None), today=TODAY)
    assert "Our file records this as met." in letter.body
    assert validate_letter(letter) == []
