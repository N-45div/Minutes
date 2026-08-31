"""Statement tests -- pure unit tests over inline synthetic fixtures.

Every fixture below is invented for testing. No child, school, or district
named here is real, and no test in this file makes a network call.

Two habits are deliberate here. First, the hostile fixtures (``_hostile_card``,
``_HOSTILE_SERVICE``) exist because a test that feeds the renderer text this
module wrote itself can only prove this module's own string literals are clean;
it proves nothing about the artifact a parent receives, which is assembled from
text extracted by a model out of a real document. Second, the figures asserted
in the rendering tests are computed by hand in the test, not read back out of
the object under test.
"""

import re
from datetime import date

import pytest

from minutes.letters import ALLOWED_CITATIONS
from minutes.models import (
    Accommodation,
    Deadline,
    DeadlineKind,
    DeadlineState,
    DeadlineStatus,
    DecisionCard,
    EvidenceRef,
    IEPLedger,
    Letter,
    LetterKind,
    Period,
    Provenance,
    ReconciliationResult,
    RecordsRequest,
    RequestState,
    ServiceObligation,
    ServiceShortfall,
    Urgency,
)
from minutes.statement import (
    DISCLAIMER,
    NO_ACTIVITY_HEADLINE,
    NO_ACTIVITY_NOTE,
    NO_SHORTFALL_HEADLINE,
    NOTHING_SCHEDULED_HEADLINE,
    NOTHING_SCHEDULED_NOTE,
    QUIET_MONTH_NOTE,
    WITHHELD_CARD_NOTE,
    WITHHELD_CELL,
    build_statement,
    render_markdown,
    render_text,
)

PERIOD_START = date(2026, 9, 1)
PERIOD_END = date(2026, 9, 30)

# The characterizations a statement must never carry, whoever wrote them.
LEGAL_CONCLUSIONS = (
    "denial of fape",
    "denied fape",
    "material failure",
    "materially failed",
    "violation",
    "violated",
    "unlawful",
    "illegal",
    "you are owed",
    "you have a claim",
    "file for due process",
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
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
                source_quote="Speech-Language Therapy: 30 minutes per session, 2 sessions per week.",
            )
        ],
        deadlines=[],
        accommodations=[Accommodation(description="Extended time", source_quote="Extended time.")],
    )


def _evidence(source: str = "log-2026-09-14") -> EvidenceRef:
    return EvidenceRef(
        provenance=Provenance.PARENT_OBSERVED,
        source=source,
        detail="Parent log 9/14: 'no speech today again'.",
    )


def _shortfall(
    service: str = "Speech-Language Therapy",
    *,
    owed: int = 240,
    delivered: int = 150,
    confirmed: int = 150,
    observed: int = 0,
    undocumented: int = 0,
    start: date = PERIOD_START,
    end: date = PERIOD_END,
) -> ServiceShortfall:
    return ServiceShortfall(
        service=service,
        period_start=start,
        period_end=end,
        owed_minutes=owed,
        delivered_minutes=delivered,
        shortfall_minutes=owed - delivered,
        school_confirmed_minutes=confirmed,
        parent_observed_minutes=observed,
        undocumented_minutes=undocumented,
        evidence=[_evidence()],
    )


def _result(*shortfalls: ServiceShortfall, events: int = 12) -> ReconciliationResult:
    return ReconciliationResult(
        period_start=PERIOD_START,
        period_end=PERIOD_END,
        shortfalls=list(shortfalls),
        events_considered=events,
    )


def _deadline_status(
    *,
    kind: DeadlineKind = DeadlineKind.PROGRESS_REPORT,
    due: date = date(2026, 11, 6),
    description: str = "Progress report on all goals provided to parents",
    state: DeadlineState = DeadlineState.UPCOMING,
    days_remaining: int = 37,
    met_on: date | None = None,
) -> DeadlineStatus:
    return DeadlineStatus(
        deadline=Deadline(
            kind=kind,
            due=due,
            description=description,
            source_quote="Progress reports will be provided at each grading period.",
        ),
        state=state,
        days_remaining=days_remaining,
        met_on=met_on,
    )


def _request(
    request_id: str = "req-001",
    *,
    state: RequestState = RequestState.UNANSWERED_OVERDUE,
    sent_on: date | None = date(2026, 9, 2),
    response_due: date | None = date(2026, 9, 17),
    answered_on: date | None = None,
    services: list[str] | None = None,
) -> RecordsRequest:
    return RecordsRequest(
        request_id=request_id,
        covers_start=date(2026, 9, 1),
        covers_end=date(2026, 9, 30),
        services=services if services is not None else ["Speech-Language Therapy"],
        state=state,
        sent_on=sent_on,
        response_due=response_due,
        answered_on=answered_on,
    )


def _card(
    card_id: str = "card-001",
    *,
    urgency: Urgency = Urgency.TIME_SENSITIVE,
    deadline: date | None = date(2026, 11, 6),
    title: str = "Speech minutes are 90 short and the school has not answered",
) -> DecisionCard:
    return DecisionCard(
        card_id=card_id,
        title=title,
        why_now="The records request passed its response date on September 17.",
        facts=["240 minutes owed in September.", "150 minutes documented as delivered."],
        recommended_action="Send the follow-up records request that is drafted for you.",
        urgency=urgency,
        deadline=deadline,
    )


def _hostile_card(
    card_id: str = "card-hostile",
    *,
    title: str = "The district's denial of FAPE is now documented",
    why_now: str = "September was a material failure and a violation of the IDEA.",
    facts: list[str] | None = None,
    recommended_action: str = "File for due process this week.",
    draft: Letter | None = None,
) -> DecisionCard:
    """A card written the way a careless generator would write one.

    No module in the repo produces cards yet, which is exactly why the gate has
    to be tested against text this module did not author.
    """
    return DecisionCard(
        card_id=card_id,
        title=title,
        why_now=why_now,
        facts=facts
        if facts is not None
        else [
            "You are owed 1,680 minutes of compensatory education.",
            "What the school did here is illegal.",
        ],
        recommended_action=recommended_action,
        urgency=Urgency.DEADLINE_IMMINENT,
        deadline=date(2026, 11, 6),
        draft=draft,
    )


# A service name as an IEP PDF might actually yield it: a stray table pipe and
# a line break from the two-column layout it was lifted out of.
_HOSTILE_SERVICE = "Speech | Language\nTherapy"


def _statement(result=None, deadlines=None, requests=None, decisions=None):
    return build_statement(
        _ledger(),
        result if result is not None else _result(_shortfall()),
        deadlines if deadlines is not None else [],
        requests if requests is not None else [],
        decisions if decisions is not None else [],
    )


def _flat(text: str) -> str:
    """The text renderer wraps prose, so match sentences against one long line."""
    return " ".join(text.split())


def _rows(text: str) -> list[str]:
    """Every rendered line with its runs of padding collapsed, so a table row
    can be compared to a hand-written one in either rendering."""
    return [" ".join(line.split()) for line in text.splitlines()]


def _cited_sections(text: str) -> set[str]:
    return set(re.findall(r"34 CFR \d+\.\d+|20 U\.S\.C\. \d+", text))


def _verified_sections() -> set[str]:
    anchor = re.compile(r"^(34 CFR \d+\.\d+|20 U\.S\.C\. \d+)")
    return {m.group(1) for c in ALLOWED_CITATIONS if (m := anchor.match(c))}


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def test_totals_equal_the_sum_of_the_lines():
    statement = _statement(
        _result(
            _shortfall("Speech-Language Therapy", owed=240, delivered=150, confirmed=150),
            _shortfall("Occupational Therapy", owed=180, delivered=90, confirmed=45, observed=45),
        )
    )

    assert statement.total_owed == 420
    assert statement.total_delivered == 240
    assert statement.total_shortfall == 180
    assert statement.total_owed == sum(line.owed_minutes for line in statement.lines)
    assert statement.total_shortfall == sum(line.shortfall_minutes for line in statement.lines)


def test_a_service_split_across_periods_becomes_one_line():
    statement = _statement(
        _result(
            _shortfall(owed=120, delivered=60, confirmed=60, start=date(2026, 9, 1), end=date(2026, 9, 15)),
            _shortfall(
                owed=120,
                delivered=30,
                confirmed=0,
                observed=30,
                undocumented=45,
                start=date(2026, 9, 16),
                end=PERIOD_END,
            ),
        )
    )

    assert len(statement.lines) == 1
    line = statement.lines[0]
    assert (line.owed_minutes, line.delivered_minutes, line.shortfall_minutes) == (240, 90, 150)
    assert line.school_confirmed_minutes == 60
    assert line.parent_observed_minutes == 30
    assert line.undocumented_minutes == 45


def test_lines_lead_with_the_largest_shortfall():
    statement = _statement(
        _result(
            _shortfall("Occupational Therapy", owed=180, delivered=180, confirmed=180),
            _shortfall("Speech-Language Therapy", owed=240, delivered=60, confirmed=60),
            _shortfall("Individual Counseling", owed=30, delivered=0, confirmed=0, undocumented=30),
        )
    )

    assert [line.service for line in statement.lines] == [
        "Speech-Language Therapy",
        "Individual Counseling",
        "Occupational Therapy",
    ]


# ---------------------------------------------------------------------------
# Filtering and ordering of what is still open
# ---------------------------------------------------------------------------


def test_met_deadlines_drop_out_and_the_rest_sort_by_urgency():
    met = _deadline_status(state=DeadlineState.MET, days_remaining=5, met_on=date(2026, 9, 20))
    overdue = _deadline_status(
        kind=DeadlineKind.ANNUAL_REVIEW,
        due=date(2026, 9, 18),
        description="Annual review meeting held",
        state=DeadlineState.OVERDUE,
        days_remaining=-12,
    )
    upcoming = _deadline_status()

    statement = _statement(deadlines=[upcoming, met, overdue])

    assert [status.state for status in statement.open_deadlines] == [
        DeadlineState.OVERDUE,
        DeadlineState.UPCOMING,
    ]


def test_only_sent_and_overdue_requests_count_as_unanswered():
    draft = _request("req-000", state=RequestState.DRAFT, sent_on=None, response_due=None)
    answered = _request("req-001", state=RequestState.ANSWERED, answered_on=date(2026, 9, 12))
    sent = _request("req-002", state=RequestState.SENT, sent_on=date(2026, 9, 20), response_due=date(2026, 10, 5))
    overdue = _request("req-003", state=RequestState.UNANSWERED_OVERDUE, sent_on=date(2026, 9, 2))

    statement = _statement(requests=[draft, answered, sent, overdue])

    # Oldest first: the request that has been waiting longest leads.
    assert [req.request_id for req in statement.unanswered_requests] == ["req-003", "req-002"]


def test_decisions_are_ordered_by_urgency():
    routine = _card("card-c", urgency=Urgency.ROUTINE, deadline=None)
    imminent = _card("card-a", urgency=Urgency.DEADLINE_IMMINENT, deadline=date(2026, 10, 2))
    timely = _card("card-b", urgency=Urgency.TIME_SENSITIVE)

    statement = _statement(decisions=[routine, timely, imminent])

    assert [card.card_id for card in statement.decisions] == ["card-a", "card-b", "card-c"]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_rendering_is_stable_and_carries_the_key_figures():
    statement = _statement(
        _result(_shortfall(owed=240, delivered=150, confirmed=150)),
        deadlines=[_deadline_status()],
        requests=[_request()],
        decisions=[_card()],
    )
    markdown = render_markdown(statement)

    assert markdown == render_markdown(statement)
    assert render_text(statement) == render_text(statement)

    assert "Maya R." in markdown
    assert "Sep 1, 2026" in markdown and "Sep 30, 2026" in markdown
    assert "90 minutes short this period." in markdown
    assert (
        "240 minutes owed. 150 minutes documented as delivered. "
        "1 of 1 service on the ledger came up short." in markdown
    )
    assert "Speech-Language Therapy" in markdown
    assert "Progress report on all goals provided to parents" in markdown
    assert "req-001" in markdown
    assert "Send the follow-up records request that is drafted for you." in markdown
    assert DISCLAIMER in markdown


def test_zero_shortfall_reads_as_reassuring():
    statement = _statement(_result(_shortfall(owed=240, delivered=240, confirmed=240)))
    markdown = render_markdown(statement)
    text = render_text(statement)

    assert NO_SHORTFALL_HEADLINE in markdown
    assert QUIET_MONTH_NOTE in markdown
    assert QUIET_MONTH_NOTE in _flat(text)
    for alarming in ("short this period", "overdue", "no evidence either way -- no school record"):
        assert alarming not in markdown
        assert alarming not in _flat(text)
    assert "Nothing is waiting on you this month." in markdown


def test_a_quiet_month_is_not_declared_while_something_is_outstanding():
    statement = _statement(
        _result(_shortfall(owed=240, delivered=240, confirmed=240)),
        requests=[_request()],
    )
    markdown = render_markdown(statement)

    assert NO_SHORTFALL_HEADLINE in markdown
    assert QUIET_MONTH_NOTE not in markdown
    assert "req-001" in markdown


def test_empty_deadlines_requests_and_decisions_render_gracefully():
    statement = _statement(_result())
    markdown = render_markdown(statement)
    text = render_text(statement)

    for rendered in (markdown, text):
        assert "No deadlines from the IEP are open." in _flat(rendered)
        assert "No records request is outstanding." in _flat(rendered)
        assert "Nothing is waiting on you this month." in _flat(rendered)
        assert "None" not in rendered
        assert "[]" not in rendered
        assert "\n\n\n" not in rendered

    # Nothing was reconciled, so no service table is fabricated.
    assert "| Service |" not in markdown
    assert "TOTAL" not in markdown


def test_a_period_with_nothing_reconciled_is_never_called_quiet():
    """An empty reconciliation makes the shortfall trivially zero. Every real
    cause of one is a pipeline failure, so it must not read as a calm month."""
    statement = _statement(_result())

    assert statement.lines == []
    assert statement.total_shortfall == 0

    for rendered in (render_markdown(statement), _flat(render_text(statement))):
        assert NO_ACTIVITY_HEADLINE in rendered
        assert NO_ACTIVITY_NOTE in rendered
        assert QUIET_MONTH_NOTE not in rendered
        assert "Nothing this month needs a decision from you" not in rendered


def test_a_large_shortfall_surfaces_prominently():
    statement = _statement(
        _result(
            _shortfall("Specialized Academic Instruction", owed=1800, delivered=480, confirmed=480),
            _shortfall("Speech-Language Therapy", owed=240, delivered=180, confirmed=180),
        )
    )
    markdown = render_markdown(statement)

    assert statement.total_shortfall == 1380
    headline = "1,380 minutes (23 hours) short this period"
    assert headline in markdown
    assert (
        "2,040 minutes (34 hours) owed. 660 minutes (11 hours) documented as "
        "delivered. 2 of 2 services on the ledger came up short." in markdown
    )
    # It leads: the figure appears before the first table, in the opening lines.
    assert markdown.index(headline) < markdown.index("| Service |")
    assert headline in "\n".join(markdown.splitlines()[:8])
    assert headline in render_text(statement)


def test_hours_are_glossed_only_once_minutes_stop_being_legible():
    small = render_markdown(_statement(_result(_shortfall(owed=90, delivered=45, confirmed=45))))
    assert "45 minutes short this period" in small
    assert "hour" not in small.replace(DISCLAIMER, "")

    large = render_markdown(_statement(_result(_shortfall(owed=1200, delivered=0, confirmed=0))))
    assert "1,200 minutes (20 hours) short this period" in large
    assert "1,200 (20h)" in large


@pytest.mark.parametrize(
    ("shortfall", "expected_prose", "expected_cell"),
    [
        (599, "599 minutes short this period.", "599"),
        (600, "600 minutes (10 hours) short this period.", "600 (10h)"),
        (601, "601 minutes (10 hours) short this period.", "601 (10h)"),
    ],
)
def test_the_hours_gloss_starts_exactly_at_the_threshold(shortfall, expected_prose, expected_cell):
    """HOURS_THRESHOLD is 600, so 599 is bare and 600 is glossed."""
    statement = _statement(_result(_shortfall(owed=shortfall, delivered=0, confirmed=0)))
    markdown = render_markdown(statement)

    assert f"**{expected_prose}**" in markdown
    assert f"| Speech-Language Therapy | {expected_cell} | 0 | {expected_cell} |" in markdown


def test_both_renderings_carry_the_same_hand_computed_rows():
    """The two renderers must agree on the tables, cell for cell.

    The figures below are computed here rather than read back from the
    statement: 1,800 - 480 = 1,320 short on one line, 240 - 150 = 90 on the
    other, 2,040 owed and 1,410 short in total, which is 23.5 hours.
    """
    statement = _statement(
        _result(
            _shortfall("Specialized Academic Instruction", owed=1800, delivered=480, confirmed=300, observed=180),
            _shortfall("Speech-Language Therapy", owed=240, delivered=150, confirmed=0, observed=150, undocumented=60),
        )
    )
    markdown_rows = _rows(render_markdown(statement))
    text_rows = _rows(render_text(statement))

    expected = [
        # service | owed | delivered | short
        ("Specialized Academic Instruction", "1,800 (30h)", "480", "1,320 (22h)"),
        ("Speech-Language Therapy", "240", "150", "90"),
        ("TOTAL", "2,040 (34h)", "630 (10.5h)", "1,410 (23.5h)"),
        # service | school-confirmed | parent-observed | undocumented
        ("Specialized Academic Instruction", "300", "180", "0"),
        ("Speech-Language Therapy", "0", "150", "60"),
        ("TOTAL", "300", "330", "60"),
    ]
    for cells in expected:
        assert "| " + " | ".join(cells) + " |" in markdown_rows
        assert " ".join(cells) in text_rows

    # Both renderings carry all four tables (services, evidence, deadlines,
    # requests) -- the plain-text one does not quietly drop a section.
    assert sum(1 for row in markdown_rows if row.startswith("| TOTAL |")) == 2
    assert sum(1 for row in text_rows if row.startswith("TOTAL ")) == 2


def test_the_lead_number_leads_in_both_renderings():
    """Markdown bolds the lead; plain text has no bold, so it rules it off.
    Rendering the lead as an ordinary paragraph would leave the headline
    figure with no emphasis at all in the email body."""
    statement = _statement(_result(_shortfall(owed=1800, delivered=120, confirmed=120)))
    headline = "1,680 minutes (28 hours) short this period."

    assert f"**{headline}**" in render_markdown(statement)

    lines = render_text(statement).splitlines()
    positions = [n for n, line in enumerate(lines) if line == headline]
    assert len(positions) == 1, "the lead figure is stated once, on a line of its own"
    above, below = lines[positions[0] - 1], lines[positions[0] + 1]
    assert above == "-" * len(headline)
    assert below == "-" * len(headline)

    # An ordinary paragraph is not ruled off the same way.
    method = next(n for n, line in enumerate(lines) if line.startswith("Every figure above"))
    assert set(lines[method - 1]) != {"-"}


def test_evidence_breakdown_is_reported_honestly():
    statement = _statement(
        _result(_shortfall(owed=240, delivered=120, confirmed=0, observed=120, undocumented=60))
    )
    markdown = render_markdown(statement)

    assert "| Service | School-confirmed | Parent-observed | Undocumented |" in markdown
    assert "Nothing this period is confirmed by the school's own records." in markdown
    assert "60 minutes of what was promised have no evidence either way" in markdown
    assert "60 minutes of what was promised have no evidence either way" in _flat(render_text(statement))


# ---------------------------------------------------------------------------
# What the lead number is allowed to imply
# ---------------------------------------------------------------------------


def test_a_month_documented_only_by_the_parent_log_is_attributed_in_the_lead():
    """Owed 240, delivered 240 -- but the school confirmed none of it. The
    reassurance rests entirely on the parent's own log and has to say so where
    the parent reads it, not two sections down."""
    statement = _statement(
        _result(_shortfall(owed=240, delivered=240, confirmed=0, observed=240))
    )
    markdown = render_markdown(statement)
    attribution = (
        "All 240 minutes promised over these dates are documented as delivered, "
        "entirely from your own log."
    )

    assert NO_SHORTFALL_HEADLINE in markdown
    assert attribution in markdown
    assert attribution in _flat(render_text(statement))
    # It travels with the number: above both tables, and above the quiet note.
    assert markdown.index(attribution) < markdown.index(QUIET_MONTH_NOTE)
    assert markdown.index(attribution) < markdown.index("| Service |")


def test_an_all_undocumented_shortfall_is_qualified_in_the_lead():
    """Owed 1,200, delivered nothing, and nobody recorded anything either way.
    The subtraction alone is not evidence of non-delivery, so the headline may
    not state a bare 20-hour gap."""
    statement = _statement(
        _result(_shortfall(owed=1200, delivered=0, confirmed=0, undocumented=1200))
    )
    headline = "1,200 minutes (20 hours) short this period, all of it with no record either way."
    markdown = render_markdown(statement)

    assert f"**{headline}**" in markdown
    assert headline in "\n".join(markdown.splitlines()[:8])
    assert headline in _flat(render_text(statement))
    # The bare figure never appears as a sentence of its own.
    assert "1,200 minutes (20 hours) short this period." not in markdown


def test_a_partly_undocumented_shortfall_splits_the_figure_in_the_lead():
    """Owed 1,200, 300 confirmed delivered, 600 of the 900-minute gap with no
    record either way: the split belongs in the headline sentence."""
    statement = _statement(
        _result(_shortfall(owed=1200, delivered=300, confirmed=300, undocumented=600))
    )
    headline = (
        "900 minutes (15 hours) short this period, "
        "600 minutes (10 hours) of it with no record either way."
    )
    markdown = render_markdown(statement)

    assert statement.total_shortfall == 900
    assert f"**{headline}**" in markdown
    assert headline in _flat(render_text(statement))
    assert (
        "Nobody has recorded 600 minutes (10 hours) of that shortfall either way, "
        "which is a gap in the evidence rather than a record of non-delivery."
        in markdown
    )


def test_a_period_with_nothing_scheduled_does_not_report_a_vacuous_all_clear():
    """An obligation on the ledger that no session fell inside gives owed = 0.
    'All 0 minutes are documented as delivered' would be an affirmative
    all-clear about a period in which nothing was promised."""
    statement = _statement(_result(_shortfall(owed=0, delivered=0, confirmed=0)))
    markdown = render_markdown(statement)

    assert statement.total_owed == 0
    assert f"**{NOTHING_SCHEDULED_HEADLINE}**" in markdown
    assert NOTHING_SCHEDULED_NOTE in markdown
    assert "All 0 minutes" not in markdown
    assert "documented as delivered." not in markdown


# ---------------------------------------------------------------------------
# Clocks
# ---------------------------------------------------------------------------


def test_outstanding_request_shows_how_long_it_has_waited():
    statement = _statement(requests=[_request(sent_on=date(2026, 9, 2), response_due=date(2026, 9, 17))])
    markdown = render_markdown(statement)

    # 28 days between the send date and the close of the statement period.
    assert "28 days, response was due Sep 17, 2026" in markdown


def test_a_request_sent_after_the_period_never_shows_negative_days():
    """September's statement is compiled in October, so a request sent on the
    8th is ordinary. It has waited no days yet -- and '-8 days' is nonsense."""
    statement = _statement(
        requests=[
            _request("req-009", state=RequestState.SENT, sent_on=date(2026, 10, 8), response_due=None)
        ]
    )
    markdown = render_markdown(statement)

    assert "req-009" in markdown
    assert "sent after this period" in markdown
    assert not re.search(r"-\d+ days", markdown)
    assert not re.search(r"-\d+ days", render_text(statement))


def test_a_deadline_due_today_says_so():
    statement = _statement(
        deadlines=[_deadline_status(state=DeadlineState.DUE_SOON, days_remaining=0, due=PERIOD_END)]
    )
    markdown = render_markdown(statement)

    assert "due today" in markdown
    assert "0 days remaining" not in markdown


# ---------------------------------------------------------------------------
# The gate: text this module did not write
# ---------------------------------------------------------------------------


def test_a_hostile_service_name_cannot_break_either_table():
    """Service names are lifted out of a PDF by a model. A stray pipe would
    open a new Markdown column and slide every figure one place left; a stray
    newline breaks the plain-text column widths."""
    statement = _statement(
        _result(_shortfall(_HOSTILE_SERVICE, owed=240, delivered=0, confirmed=0))
    )
    markdown = render_markdown(statement)
    text = render_text(statement)

    table_lines = [line for line in markdown.splitlines() if line.startswith("|")]
    assert table_lines, "the service table should have rendered"
    for line in table_lines:
        # Four columns means five delimiters; an escaped pipe is not one.
        assert len(re.findall(r"(?<!\\)\|", line)) == 5, line
    assert r"| Speech \| Language Therapy | 240 | 0 | 240 |" in markdown

    assert "Speech | Language Therapy 240 0 240" in _rows(text)
    assert "\n\n\n" not in text


def test_an_extracted_label_that_states_a_conclusion_is_withheld_but_its_clock_is_not():
    """Deadline descriptions and request scopes are extracted text too. A row
    may lose its label to the gate; it must never lose its date or its count."""
    statement = _statement(
        deadlines=[
            _deadline_status(
                description="Compensatory minutes owed for the district's denial of FAPE",
                state=DeadlineState.OVERDUE,
                days_remaining=-12,
                due=date(2026, 9, 18),
            )
        ],
        requests=[_request("req-007", services=["Speech, after the violation of the IEP"])],
    )
    markdown = render_markdown(statement)

    assert "denial of FAPE" not in markdown
    assert "violation" not in markdown
    assert WITHHELD_CELL in markdown
    # The clocks survive the withholding.
    assert "Sep 18, 2026" in markdown
    assert "12 days overdue" in markdown
    assert "req-007" in markdown
    assert "28 days, response was due Sep 17, 2026" in markdown


def test_statement_states_the_arithmetic_and_draws_no_legal_conclusion():
    """The banned phrases are fed in through a decision card, not merely absent
    from this module's own literals -- a card is the only free text that
    reaches the artifact, and no module produces one yet."""
    statement = _statement(
        _result(_shortfall(owed=1800, delivered=120, confirmed=120)),
        deadlines=[_deadline_status(state=DeadlineState.OVERDUE, days_remaining=-40)],
        requests=[_request()],
        decisions=[_hostile_card()],
    )
    rendered = (render_markdown(statement) + render_text(statement)).lower()

    for conclusion in LEGAL_CONCLUSIONS:
        assert conclusion not in rendered

    assert "not legal advice" in rendered
    assert "reconciliation, not a conclusion" in rendered


def test_a_card_that_states_a_conclusion_is_held_back_whole_and_visibly():
    statement = _statement(decisions=[_hostile_card("card-99")])
    markdown = render_markdown(statement)

    assert "card-99" in markdown
    assert WITHHELD_CARD_NOTE in markdown
    assert "compensatory education" not in markdown  # no half a card survives
    # The card's clock is not lost with its prose.
    assert "Urgency: deadline imminent." in markdown
    assert "Deadline Nov 6, 2026" in markdown
    assert WITHHELD_CARD_NOTE in _flat(render_text(statement))


@pytest.mark.parametrize(
    "card",
    [
        pytest.param(_hostile_card(title="September was a denial of FAPE"), id="title"),
        pytest.param(
            _hostile_card(why_now="The district violated 34 CFR 300.320(a)(7).", facts=["a"]),
            id="why_now",
        ),
        pytest.param(
            _hostile_card(
                title="Speech is short",
                why_now="The response date passed.",
                facts=["The shortfall is a material failure."],
                recommended_action="Send the drafted request.",
            ),
            id="fact",
        ),
        pytest.param(
            _hostile_card(
                title="Speech is short",
                why_now="The response date passed.",
                facts=["240 minutes owed."],
                recommended_action="File for due process.",
            ),
            id="recommended_action",
        ),
        pytest.param(
            _hostile_card(
                title="Speech is short",
                why_now="The response date passed.",
                facts=["240 minutes owed."],
                recommended_action="Send the drafted request.",
                draft=Letter(
                    kind=LetterKind.SHORTFALL_NOTICE,
                    subject="Notice of the district's violation of the IDEA",
                    body="Body.",
                    citations=[],
                ),
            ),
            id="draft_subject",
        ),
    ],
)
def test_every_pass_through_field_of_a_card_is_gated(card):
    markdown = render_markdown(_statement(decisions=[card])).lower()

    assert WITHHELD_CARD_NOTE.lower() in markdown
    for conclusion in LEGAL_CONCLUSIONS:
        assert conclusion not in markdown


def test_a_card_citing_an_unverified_regulation_is_held_back():
    """The statement is the document most likely to be forwarded to a district,
    so a section number it cannot vouch for never reaches the page."""
    card = _hostile_card(
        title="Speech minutes are short",
        why_now="34 CFR 300.999(c) requires the district to make this up.",
        facts=["240 minutes owed."],
        recommended_action="Send the drafted request.",
    )
    markdown = render_markdown(_statement(decisions=[card]))

    assert "300.999" not in markdown
    assert WITHHELD_CARD_NOTE in markdown


def test_a_clean_card_is_rendered_in_full():
    """The gate withholds; it does not swallow ordinary cards."""
    markdown = render_markdown(_statement(decisions=[_card()]))

    assert "Speech minutes are 90 short and the school has not answered" in markdown
    assert "Why now: The records request passed its response date on September 17." in markdown
    assert "- 240 minutes owed in September." in markdown
    assert "Suggested next step: Send the follow-up records request that is drafted for you." in markdown
    assert WITHHELD_CARD_NOTE not in markdown


def test_every_regulation_the_statement_cites_is_on_the_verified_list():
    """Mechanical, not a matter of the author having been careful once: the
    one citation in the disclaimer is looked up in letters.ALLOWED_CITATIONS."""
    statement = _statement(
        _result(_shortfall(owed=1800, delivered=120, confirmed=120)),
        deadlines=[_deadline_status(state=DeadlineState.OVERDUE, days_remaining=-40)],
        requests=[_request()],
        decisions=[_card()],
    )
    verified = _verified_sections()

    for rendered in (render_markdown(statement), render_text(statement)):
        cited = _cited_sections(rendered)
        assert cited, "the disclaimer names the free-legal-services regulation"
        assert cited <= verified, sorted(cited - verified)

    assert "34 CFR 300.507(b)" in ALLOWED_CITATIONS


def test_a_citation_is_never_broken_across_two_lines_of_the_email_body():
    """The body is wrapped at 76 columns, and the disclaimer's citation lands
    near the end of a line. A reference split as '34' / 'CFR 300.507(b)' is one
    more thing for a district reader to doubt."""
    text = render_text(_statement())

    assert "34 CFR 300.507(b)" in text
    assert "34\nCFR" not in text
    assert all(len(line) <= 76 for line in text.splitlines() if not line.startswith(" "))
