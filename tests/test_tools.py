"""Tool-layer tests — the shapes the model sees, and the numbers behind them.

Every expected figure here was computed by hand against the committed fixture
case ("Maya R.", fall 2026) and is asserted literally. A tool that quietly
starts returning different arithmetic is a tool that will put a different
number in a letter to a school district, so nothing in this file asserts
merely that a key exists.

Two properties get their own tests because the product's safety rests on them:
no tool mutates the case, and no tool raises into the event loop.

No test in this file makes a network call, an AWS call, or an LLM call. The
tools wrap the deterministic engine and read committed fixtures; there is
nothing here for a model to be asked.
"""

from datetime import date

import pytest

from minutes import tools
from minutes.discovery import RESPONSE_WINDOW_DAYS
from minutes.models import DecisionCard, IEPLedger, RecordsRequest, ServiceEvent
from minutes.tools import (
    CaseRecord,
    build_monthly_statement,
    check_deadlines,
    draft_records_request,
    draft_shortfall_letter,
    load_case,
    load_case_record,
    pending_decisions,
    reconcile_services,
    records_requests_status,
)

# The fall term of the fixture case: services start 2026-09-08, the cached
# evidence runs from 2026-09-16, and 2026-12-19 is the last day of term.
TERM_START = "2026-09-01"
TERM_END = "2026-12-19"
TODAY = "2026-12-19"

SPEECH = "Speech-Language Therapy"
OT = "Occupational Therapy"
SAI = "Specialized Academic Instruction"
COUNSELING = "Individual Counseling"

# Hand-computed from fixtures/cache/iep_maya_ledger.json against the cached
# events, per service, for 2026-09-01..2026-12-19.
EXPECTED_SHORTFALLS = {
    SPEECH: {"owed": 870, "delivered": 615, "excused": 0, "short": 255, "undocumented": 195},
    OT: {"owed": 630, "delivered": 585, "excused": 45, "short": 0, "undocumented": 0},
    SAI: {"owed": 4440, "delivered": 0, "excused": 0, "short": 4440, "undocumented": 4440},
    COUNSELING: {"owed": 90, "delivered": 60, "excused": 0, "short": 30, "undocumented": 0},
}

EVERY_TOOL = (
    load_case,
    reconcile_services,
    check_deadlines,
    records_requests_status,
    draft_records_request,
    draft_shortfall_letter,
    build_monthly_statement,
    pending_decisions,
)


def _call(tool_fn, **kwargs) -> dict:
    """Invoke a tool the way the model would, but without a model.

    ``@tool`` returns a ``DecoratedFunctionTool`` that is still directly
    callable as the plain Python function, so the deterministic layer is
    testable with no agent, no Bedrock, and no tokens.
    """
    return tool_fn(**kwargs)


# ---------------------------------------------------------------------------
# The tool contract itself
# ---------------------------------------------------------------------------


def test_every_tool_is_registered_with_a_model_readable_spec():
    for tool_fn in EVERY_TOOL:
        spec = tool_fn.tool_spec
        assert spec["name"] == tool_fn.__name__
        # The description is what the model reads to decide what to call, so a
        # tool that lost its docstring is a tool the model will misuse.
        assert len(spec["description"]) > 150
        assert spec["inputSchema"]["json"]["type"] == "object"


# The sentence in each tool's description that keeps the model honest about
# what the tool's numbers do and do not mean. The @tool decorator builds the
# description by stripping the docstring's Args: section, and it strips
# everything after it too — so guidance written below Args: is silently
# invisible to the model. These are the exact phrases that must survive.
GUIDANCE_THE_MODEL_MUST_READ = {
    "load_case": "is ever a reason a service fell short",
    "reconcile_services": "NOT proven non-delivery",
    "check_deadlines": "not a finding",
    "records_requests_status": "never establishes that a session was missed",
    "draft_records_request": "Nothing is sent",
    "draft_shortfall_letter": "Nothing is sent",
    "build_monthly_statement": "do not restate its numbers",
    "pending_decisions": "do not manufacture",
}


def test_the_honesty_guidance_survives_into_the_description_the_model_reads():
    """A regression guard with teeth. Move a line below Args: and the model
    stops seeing it, with no error anywhere — which is how a tool that says
    'undocumented is not non-delivery' quietly stops saying it."""
    for tool_fn in EVERY_TOOL:
        expected = GUIDANCE_THE_MODEL_MUST_READ[tool_fn.__name__]
        assert expected in tool_fn.tool_spec["description"], tool_fn.__name__


def test_no_tool_description_is_truncated_at_the_args_block():
    """Every tool's full docstring prose reaches the model, not just the
    summary line that precedes Args:."""
    for tool_fn in EVERY_TOOL:
        description = tool_fn.tool_spec["description"]
        summary = (tool_fn.__doc__ or "").strip().splitlines()[0]
        assert description.strip() != summary.strip(), tool_fn.__name__


def test_date_parameters_are_strings_in_the_schema():
    """ISO strings cross the boundary, never date objects: the model sees a
    serialized rendering of whatever a tool takes and returns."""
    for tool_fn, params in (
        (reconcile_services, ("start", "end")),
        (check_deadlines, ("today",)),
        (records_requests_status, ("today",)),
        (draft_records_request, ("today",)),
        (draft_shortfall_letter, ("start", "end", "today")),
        (build_monthly_statement, ("start", "end", "today")),
        (pending_decisions, ("today",)),
    ):
        properties = tool_fn.tool_spec["inputSchema"]["json"]["properties"]
        for param in params:
            assert properties[param]["type"] == "string", (tool_fn.__name__, param)


def test_returns_are_json_friendly_primitives():
    """No Pydantic object may cross the tool boundary."""
    import json

    payloads = [
        _call(load_case),
        _call(reconcile_services, start=TERM_START, end=TERM_END),
        _call(check_deadlines, today=TODAY),
        _call(records_requests_status, today=TODAY),
        _call(draft_records_request, today=TODAY),
        _call(draft_shortfall_letter, start=TERM_START, end=TERM_END, today=TODAY),
        _call(build_monthly_statement, start=TERM_START, end=TERM_END, today=TODAY),
        _call(pending_decisions, today=TODAY),
    ]
    for payload in payloads:
        json.dumps(payload)  # raises TypeError on a stray model or date


# ---------------------------------------------------------------------------
# load_case
# ---------------------------------------------------------------------------


def test_load_case_returns_the_promise_not_the_delivery():
    case = _call(load_case)

    assert case["student"] == "Maya R."
    assert case["iep_date"] == "2026-09-01"
    assert case["promised_minutes_per_week"] == 412.5

    services = {o["service"]: o for o in case["obligations"]}
    assert set(services) == {SPEECH, OT, SAI, COUNSELING}
    assert services[SPEECH]["minutes_per_session"] == 30
    assert services[SPEECH]["sessions_per_period"] == 2
    assert services[SPEECH]["period"] == "week"
    assert services[SPEECH]["minutes_per_week"] == 60.0
    assert services[SAI]["period"] == "day"
    assert services[COUNSELING]["period"] == "month"

    # Nothing about delivery appears here. Confusing the promise with the
    # record is the single mistake this product cannot make.
    assert "delivered_minutes" not in services[SPEECH]
    assert "shortfall_minutes" not in services[SPEECH]


def test_load_case_summarizes_the_evidence_without_dumping_it():
    case = _call(load_case)
    evidence = case["evidence"]

    assert evidence["events"] == 64
    assert evidence["correspondence_items"] == 68
    assert evidence["earliest_event"] == "2026-09-16"
    assert evidence["latest_event"] == "2027-01-28"
    assert evidence["by_evidence_grade"]["school_confirmed"] > 0
    # 64 events summarized to counts, not returned one by one.
    assert "events_detail" not in evidence


def test_the_absence_count_never_travels_without_the_note_that_frames_it():
    """The one figure here that is about the child rather than the district.

    reconcile_services attaches a note to every number it returns; this count
    used to travel bare, next to her accommodations, with nothing to stop a
    model writing "she was away three times, which is part of why her speech
    minutes are behind" into a permanent case record.
    """
    evidence = _call(load_case)["evidence"]

    assert evidence["dates_noting_student_absence"] == 3
    note = evidence["absence_note"]
    assert "never a reason the services fell short" in note
    assert "never a claim about the child" in note
    # And the model is told to read it, in the part of the docstring it sees.
    description = load_case.tool_spec["description"]
    assert "the promise, not the delivery" in description
    assert "absence_note" in description


def test_load_case_states_the_four_deadlines_the_iep_sets():
    case = _call(load_case)
    kinds = [d["kind"] for d in case["deadlines"]]

    assert kinds.count("progress_report") == 4
    assert kinds.count("annual_review") == 1
    assert kinds.count("reevaluation") == 1
    assert len(case["accommodations"]) == 4


# ---------------------------------------------------------------------------
# reconcile_services — the arithmetic the model must never do itself
# ---------------------------------------------------------------------------


def test_reconcile_matches_the_hand_computed_fixture_figures():
    result = _call(reconcile_services, start=TERM_START, end=TERM_END)

    rows = {row["service"]: row for row in result["services"]}
    assert set(rows) == set(EXPECTED_SHORTFALLS)

    for service, expected in EXPECTED_SHORTFALLS.items():
        row = rows[service]
        assert row["owed_minutes"] == expected["owed"], service
        assert row["delivered_minutes"] == expected["delivered"], service
        assert row["excused_minutes"] == expected["excused"], service
        assert row["shortfall_minutes"] == expected["short"], service
        assert row["undocumented_minutes"] == expected["undocumented"], service


def test_reconcile_totals_agree_with_the_per_service_rows():
    result = _call(reconcile_services, start=TERM_START, end=TERM_END)
    totals = result["totals"]

    assert totals["owed_minutes"] == 6030
    assert totals["delivered_minutes"] == 1260
    assert totals["excused_minutes"] == 45
    assert totals["shortfall_minutes"] == 4725
    assert totals["undocumented_minutes"] == 4635

    for field, key in (
        ("owed_minutes", "owed"),
        ("delivered_minutes", "delivered"),
        ("shortfall_minutes", "short"),
    ):
        assert totals[field] == sum(e[key] for e in EXPECTED_SHORTFALLS.values())

    assert result["services_short"] == 3
    assert result["events_considered"] == 54


def test_delivered_splits_into_evidence_grades_that_sum_back():
    result = _call(reconcile_services, start=TERM_START, end=TERM_END)
    for row in result["services"]:
        assert (
            row["school_confirmed_minutes"] + row["parent_observed_minutes"]
            == row["delivered_minutes"]
        ), row["service"]


def test_reconcile_carries_the_honesty_notes_the_model_must_read():
    """undocumented is missing evidence, not proven non-delivery. The model is
    told so in the payload, because that is where it is looking."""
    result = _call(reconcile_services, start=TERM_START, end=TERM_END)
    joined = " ".join(result["notes"]).lower()

    assert "not a record of non-delivery" in joined
    assert "absent" in joined


def test_a_window_before_any_service_started_is_all_zeroes_not_an_error():
    result = _call(reconcile_services, start="2026-06-01", end="2026-06-30")

    assert result["totals"]["owed_minutes"] == 0
    assert result["totals"]["shortfall_minutes"] == 0
    assert result["services_short"] == 0
    # A predictable row per service survives even an empty window.
    assert len(result["services"]) == 4


# ---------------------------------------------------------------------------
# check_deadlines
# ---------------------------------------------------------------------------


def test_check_deadlines_states_the_overdue_progress_report():
    result = _call(check_deadlines, today=TODAY)

    assert len(result["deadlines"]) == 6
    assert result["overdue"] == 1
    assert result["next_action_date"] == "2026-11-06"

    overdue = [d for d in result["deadlines"] if d["state"] == "overdue"]
    assert len(overdue) == 1
    assert overdue[0]["kind"] == "progress_report"
    assert overdue[0]["due"] == "2026-11-06"
    assert overdue[0]["days_remaining"] == -43
    # The calendar calls any overdue deadline imminent; decisions.py caps a
    # progress report below that, because a report that has not arrived is a
    # document to chase, not an opportunity lapsing. Both tools report the
    # capped value — see the test below for why that matters.
    assert overdue[0]["urgency"] == "time_sensitive"


def test_a_deadline_carries_one_urgency_no_matter_which_tool_reports_it():
    """Two scales for one fact would defeat the cap's whole purpose.

    check_deadlines and pending_decisions both describe the progress report
    missed on 2026-11-06, and a model composing prose from both is never told
    they might be measured differently. On 2026-12-01 the raw engine urgency is
    deadline_imminent and the interruption urgency is time_sensitive; the tools
    must agree on the second.
    """
    from minutes.deadlines import evaluate_deadlines, urgency_for
    from minutes.models import DeadlineState

    day = "2026-12-01"
    row = next(
        d for d in _call(check_deadlines, today=day)["deadlines"] if d["due"] == "2026-11-06"
    )
    card = next(
        c
        for c in _call(pending_decisions, today=day)["decisions"]
        if c["card_id"].startswith("deadline:progress_report:2026-11-06")
    )

    status = next(
        s
        for s in evaluate_deadlines(load_case_record().ledger, date(2026, 12, 1))
        if s.deadline.due == date(2026, 11, 6)
    )
    assert status.state is DeadlineState.OVERDUE
    assert urgency_for(status).value == "deadline_imminent"

    assert row["urgency"] == card["urgency"] == "time_sensitive"


def test_deadlines_come_back_in_due_date_order():
    result = _call(check_deadlines, today=TODAY)
    dues = [d["due"] for d in result["deadlines"]]
    assert dues == sorted(dues)


def test_days_remaining_is_positive_before_the_date_and_negative_after():
    early = _call(check_deadlines, today="2026-10-01")
    by_due = {d["due"]: d for d in early["deadlines"]}

    assert by_due["2026-11-06"]["days_remaining"] == 36
    assert by_due["2026-11-06"]["state"] == "upcoming"
    assert early["overdue"] == 0


def test_a_deadline_inside_its_window_reads_due_soon():
    """A progress report carries a 14-day window, so it is upcoming at 17 days
    out and due_soon at 12."""
    early = _call(check_deadlines, today="2026-10-20")
    late = _call(check_deadlines, today="2026-10-25")

    assert {d["due"]: d for d in early["deadlines"]}["2026-11-06"]["state"] == "upcoming"
    assert {d["due"]: d for d in late["deadlines"]}["2026-11-06"]["state"] == "due_soon"
    assert early["due_soon"] == 0
    assert late["due_soon"] == 1


# ---------------------------------------------------------------------------
# records_requests_status
# ---------------------------------------------------------------------------


def test_the_first_request_comes_due_covering_the_whole_term():
    result = _call(records_requests_status, today=TODAY)

    assert result["requests"] == []
    assert result["outstanding"] == []
    assert result["overdue_unanswered"] == []
    assert result["response_window_days"] == RESPONSE_WINDOW_DAYS == 45

    assert len(result["due_to_send"]) == 1
    due = result["due_to_send"][0]
    assert due["request_id"] == "req-001"
    assert due["covers_start"] == "2026-09-08"
    assert due["covers_end"] == TERM_END
    assert due["state"] == "draft"
    assert set(due["services"]) == {SPEECH, OT, SAI, COUNSELING}


def test_nothing_is_due_before_the_cadence_window_has_accrued():
    result = _call(records_requests_status, today="2026-09-15")
    assert result["due_to_send"] == []


def test_status_never_reports_anything_as_sent():
    """This tool reads state. It cannot create it."""
    result = _call(records_requests_status, today=TODAY)
    for row in result["due_to_send"]:
        assert row["sent_on"] is None
        assert row["response_due"] is None
        assert row["state"] == "draft"


# ---------------------------------------------------------------------------
# draft_records_request
# ---------------------------------------------------------------------------


def test_draft_records_request_compiles_a_sound_letter():
    result = _call(draft_records_request, today=TODAY)

    assert result["drafted"] is True
    assert result["request"]["request_id"] == "req-001"

    letter = result["letter"]
    assert letter["kind"] == "records_request"
    assert "2026-09-08 to 2026-12-19" in letter["subject"]
    assert letter["integrity_violations"] == []
    assert letter["citations"]["count"] == 5
    assert "34 CFR 300.613(a)" in letter["legal_basis"]
    assert letter["disclaimer"]


def test_a_records_request_is_sound_but_not_yet_sendable():
    """It is compiled correctly and invents nothing, and it still cannot go
    out until the parent fills the one fact Minutes refuses to supply."""
    letter = _call(draft_records_request, today=TODAY)["letter"]

    assert letter["integrity_violations"] == []
    assert letter["ready_to_send"] is False
    assert len(letter["blocking_before_send"]) == 1
    assert "placeholder" in letter["blocking_before_send"][0]


def test_draft_records_request_refuses_when_nothing_is_due():
    result = _call(draft_records_request, today="2026-09-15")

    assert result["drafted"] is False
    assert "reason" in result
    assert "letter" not in result


def test_the_drafting_tool_says_out_loud_that_nothing_was_sent():
    result = _call(draft_records_request, today=TODAY)
    assert "nothing was sent" in result["note"].lower()


# ---------------------------------------------------------------------------
# draft_shortfall_letter
# ---------------------------------------------------------------------------


def test_shortfall_notice_carries_the_period_arithmetic():
    result = _call(draft_shortfall_letter, start=TERM_START, end=TERM_END, today=TODAY)

    assert result["escalated"] is False
    assert result["shortfall_minutes"] == 4725
    assert result["undocumented_minutes"] == 4635

    letter = result["letter"]
    assert letter["kind"] == "shortfall_notice"
    assert letter["integrity_violations"] == []
    assert letter["ready_to_send"] is True
    # 42 before the letter stated the reasons the records give, and four more
    # since: the district's own emails explain 2026-09-24 (speech) and
    # 2026-11-18 (OT) as a school activity, and the parent's December notes
    # explain 2026-12-08 and 2026-12-10 as the post being vacant. Each of those
    # four sentences hangs from the record it was read out of.
    assert letter["citations"]["count"] == 46


def test_escalating_compiles_a_compensatory_request_instead():
    result = _call(draft_shortfall_letter, start=TERM_START, end=TERM_END, today=TODAY, escalate=True)

    assert result["escalated"] is True
    assert result["letter"]["kind"] == "compensatory_request"
    assert result["letter"]["integrity_violations"] == []


def test_the_default_letter_is_the_non_escalating_one():
    """Escalation is a decision, so it is never the default a model falls into."""
    assert draft_shortfall_letter.tool_spec["inputSchema"]["json"]["properties"]["escalate"][
        "default"
    ] is False
    assert "escalate" not in draft_shortfall_letter.tool_spec["inputSchema"]["json"]["required"]


@pytest.mark.parametrize("escalate", ["false", "no", "0", "yes", 1, None, ""])
def test_a_non_boolean_escalate_is_refused_rather_than_read_as_true(escalate):
    """Every non-empty string is truthy, and one of them spells "false".

    Nothing enforces the declared boolean at the call boundary, so a model that
    serialized False as the string 'false' would have got the COMPENSATORY
    request — the letter the docstring gates behind an evidenced gap and a
    district that has already had its chance to answer.
    """
    result = _call(
        draft_shortfall_letter,
        start=TERM_START,
        end=TERM_END,
        today=TODAY,
        escalate=escalate,
    )

    assert set(result) == {"error"}
    assert "escalate must be true or false" in result["error"]


def test_the_letter_is_dated_today_and_not_the_end_of_the_period_it_reports():
    """A parent asking in March about the autumn term gets a letter dated in March.

    The reconciliation window is a fact about the period and does not move. The
    date on the correspondence is a fact about the day it is written, and a
    document dated three months in the past is not one a district should ever
    receive.
    """
    asked_later = _call(
        draft_shortfall_letter, start=TERM_START, end=TERM_END, today="2027-03-01"
    )

    assert asked_later["period"] == {"start": TERM_START, "end": TERM_END}
    assert asked_later["today"] == "2027-03-01"
    assert asked_later["letter"]["body"].splitlines()[0] == "Date: 2027-03-01"
    # Same period, different day: the arithmetic is identical, the date is not.
    same_day = _call(draft_shortfall_letter, start=TERM_START, end=TERM_END, today=TERM_END)
    assert same_day["shortfall_minutes"] == asked_later["shortfall_minutes"]
    assert same_day["letter"]["body"].splitlines()[0] == f"Date: {TERM_END}"


def test_letter_citations_are_summarized_by_evidence_grade():
    """Forty-six evidence records would not change the model's next move; the
    split that decides how strongly the letter may speak is three numbers.

    The parent-observed count is the one worth watching. It rose from one to
    three when the letter began stating reasons, because the only records
    explaining the December weeks are the family's own notes -- so those two
    sentences are attributed to the family in the letter, and would be a claim
    about what the district said if this split were ever collapsed.
    """
    letter = _call(draft_shortfall_letter, start=TERM_START, end=TERM_END, today=TODAY)["letter"]
    grades = letter["citations"]["by_evidence_grade"]

    assert sum(grades.values()) == letter["citations"]["count"]
    assert grades["school_confirmed"] == 43
    assert grades["parent_observed"] == 3


def test_the_letter_body_is_returned_whole_because_it_is_the_artifact():
    letter = _call(draft_shortfall_letter, start=TERM_START, end=TERM_END, today=TODAY)["letter"]
    assert len(letter["body"]) > 1000
    assert "Maya R." in letter["body"]


# ---------------------------------------------------------------------------
# build_monthly_statement
# ---------------------------------------------------------------------------


def test_statement_totals_match_the_reconciliation():
    statement = _call(build_monthly_statement, start=TERM_START, end=TERM_END, today=TODAY)
    reconciliation = _call(reconcile_services, start=TERM_START, end=TERM_END)

    assert statement["student"] == "Maya R."
    assert statement["totals"]["owed_minutes"] == reconciliation["totals"]["owed_minutes"]
    assert statement["totals"]["delivered_minutes"] == reconciliation["totals"]["delivered_minutes"]
    assert statement["totals"]["shortfall_minutes"] == reconciliation["totals"]["shortfall_minutes"]
    assert statement["services"] == 4
    assert statement["open_deadlines"] == 6


def test_statement_renders_the_finished_markdown_artifact():
    statement = _call(build_monthly_statement, start=TERM_START, end=TERM_END, today=TODAY)
    markdown = statement["markdown"]

    assert markdown.startswith("# Minutes statement for Maya R.")
    assert "4,725 minutes" in markdown
    assert SPEECH in markdown


def test_statement_evaluates_clocks_against_today_not_the_period_end():
    """The same service period, issued on two dates. The arithmetic is a fact
    about the period and does not move; what is waiting on the parent is a
    fact about today and does."""
    early = _call(build_monthly_statement, start=TERM_START, end=TERM_END, today="2026-10-01")
    late = _call(build_monthly_statement, start=TERM_START, end=TERM_END, today=TERM_END)

    assert early["totals"] == late["totals"]
    assert early["today"] == "2026-10-01"

    # On 1 Oct the progress report is still 36 days out; by 19 Dec it has been
    # missed, and the records-request cadence has come due.
    assert early["decisions"] == 0
    assert late["decisions"] == 2
    assert early["markdown"] != late["markdown"]


# ---------------------------------------------------------------------------
# pending_decisions — the only thing that interrupts a parent
# ---------------------------------------------------------------------------


def test_pending_decisions_surfaces_the_two_cards_the_fixture_earns():
    result = _call(pending_decisions, today=TODAY)

    assert result["quiet"] is False
    assert result["count"] == 2
    assert result["considered_period"] == {"start": "2026-09-08", "end": TODAY}

    cards = {c["card_id"]: c for c in result["decisions"]}
    assert any(c["urgency"] == "time_sensitive" for c in cards.values())
    for card in cards.values():
        assert card["title"]
        assert card["why_now"]
        assert card["recommended_action"]
        assert isinstance(card["facts"], list)


def test_decision_cards_come_back_most_urgent_first():
    order = {"deadline_imminent": 0, "time_sensitive": 1, "routine": 2}
    decisions = _call(pending_decisions, today=TODAY)["decisions"]
    ranks = [order[c["urgency"]] for c in decisions]
    assert ranks == sorted(ranks)


def test_an_undocumented_gap_alone_does_not_raise_a_shortfall_card():
    """4,440 of the 4,725 short minutes have no record either way. Waking a
    parent to demand compensatory services on a gap in their own records is
    exactly the interruption this product exists not to make."""
    decisions = _call(pending_decisions, today=TODAY)["decisions"]
    assert not any("shortfall" in c["card_id"] for c in decisions)


def test_a_quiet_period_is_reported_as_quiet_and_empty():
    result = _call(pending_decisions, today="2026-09-01")

    assert result["quiet"] is True
    assert result["count"] == 0
    assert result["decisions"] == []


def test_decision_cards_report_whether_a_draft_is_attached():
    decisions = _call(pending_decisions, today=TODAY)["decisions"]
    assert all(isinstance(c["has_draft_letter"], bool) for c in decisions)
    assert any(c["has_draft_letter"] for c in decisions)
    # The draft letter body is not inlined into the card list; it is compiled
    # on demand by the drafting tools.
    assert all("draft" not in c or not isinstance(c.get("draft"), dict) for c in decisions)


# ---------------------------------------------------------------------------
# Read-only: no tool changes case state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool_fn", "kwargs"),
    [
        (load_case, {}),
        (reconcile_services, {"start": TERM_START, "end": TERM_END}),
        (check_deadlines, {"today": TODAY}),
        (records_requests_status, {"today": TODAY}),
        (draft_records_request, {"today": TODAY}),
        (draft_shortfall_letter, {"start": TERM_START, "end": TERM_END, "today": TODAY}),
        (build_monthly_statement, {"start": TERM_START, "end": TERM_END, "today": TODAY}),
        (pending_decisions, {"today": TODAY}),
    ],
    ids=lambda value: value.__name__ if callable(value) else "",
)
def test_calling_a_tool_twice_returns_the_same_thing(tool_fn, kwargs):
    assert _call(tool_fn, **kwargs) == _call(tool_fn, **kwargs)


def test_no_tool_mutates_the_case_record():
    before = load_case_record()

    _call(draft_records_request, today=TODAY)
    _call(draft_shortfall_letter, start=TERM_START, end=TERM_END, today=TODAY, escalate=True)
    _call(build_monthly_statement, start=TERM_START, end=TERM_END, today=TODAY)
    _call(pending_decisions, today=TODAY)

    after = load_case_record()
    assert after.ledger == before.ledger
    assert after.events == before.events
    assert after.requests == before.requests
    assert after.requests == []


def test_the_case_record_is_handed_out_as_a_copy():
    """A caller that mutates what it got cannot corrupt the next tool call."""
    first = load_case_record()
    first.ledger.obligations.clear()
    first.events.clear()

    second = load_case_record()
    assert len(second.ledger.obligations) == 4
    assert len(second.events) == 64


def test_drafting_a_letter_does_not_mark_a_request_as_sent():
    _call(draft_records_request, today=TODAY)

    after = _call(records_requests_status, today=TODAY)
    assert after["outstanding"] == []
    assert after["due_to_send"][0]["state"] == "draft"
    assert after["due_to_send"][0]["sent_on"] is None


# ---------------------------------------------------------------------------
# The request-aware tools read the case the AGENT knows, not just the file
# ---------------------------------------------------------------------------


class _FakeState:
    """The two methods of ``agent.state`` these tools use. Not a mock of a tool."""

    def __init__(self, values: dict):
        self._values = values

    def get(self, key=None):
        return self._values.get(key) if key is not None else dict(self._values)

    def set(self, key, value):
        self._values[key] = value


class _FakeContext:
    """A ToolContext stand-in carrying only the agent whose state is read."""

    def __init__(self, requests):
        self.agent = type("_Agent", (), {"state": _FakeState({tools.STATE_REQUESTS: requests})})()


def _context_with_a_sent_request() -> _FakeContext:
    """An agent that released req-001 on 2026-10-15 and has heard nothing since.

    Hand-computed: sent 2026-10-15 plus the 45-day window of 34 CFR 300.613(a)
    makes the response due 2026-11-29, so by 2026-12-05 it is six days overdue.
    """
    sent = RecordsRequest(
        request_id="req-001",
        covers_start=date(2026, 9, 8),
        covers_end=date(2026, 10, 15),
        services=[SPEECH, OT, SAI, COUNSELING],
        state="sent",
        sent_on=date(2026, 10, 15),
        response_due=date(2026, 11, 29),
    )
    return _FakeContext([sent.model_dump(mode="json")])


def test_status_reports_the_request_the_agent_actually_sent():
    """Reading only the frozen baseline would tell a parent no request was ever made.

    The case record's request history is empty in this build, so a tool that
    read it alone would answer "outstanding: none" to a parent holding their own
    posted copy of req-001 — the single worst thing a paper-trail product can
    say.
    """
    day = "2026-12-05"
    blind = _call(records_requests_status, today=day)
    assert blind["requests"] == []  # the file's view, with no agent running

    aware = _call(records_requests_status, today=day, tool_context=_context_with_a_sent_request())

    assert [r["request_id"] for r in aware["requests"]] == ["req-001"]
    assert aware["outstanding"] == ["req-001"]
    assert aware["overdue_unanswered"] == ["req-001"]
    assert aware["requests"][0]["response_due"] == "2026-11-29"


def test_drafting_stands_down_while_the_district_is_still_inside_its_window():
    """An open request suppresses the cadence — but only if the tool can see it."""
    inside_the_window = _call(
        draft_records_request, today="2026-11-01", tool_context=_context_with_a_sent_request()
    )

    assert inside_the_window["drafted"] is False
    assert inside_the_window["outstanding"] == ["req-001"]


def test_drafting_asks_about_the_next_window_rather_than_the_one_already_asked_about():
    """Blind to req-001, this tool re-drafted it: same id, whole term, twice asked.

    An overdue request deliberately does not stall the cadence, so a second one
    IS due here — but for 2026-10-16 onward, the days req-001 never covered, and
    under a fresh id. Reading the file alone produced 'req-001' covering
    2026-09-08 to 2026-12-05: a duplicate demand to the district under an id
    already in use for a different period.
    """
    aware = _call(
        draft_records_request, today="2026-12-05", tool_context=_context_with_a_sent_request()
    )

    assert aware["drafted"] is True
    assert aware["request"]["request_id"] == "req-002"
    assert aware["request"]["covers_start"] == "2026-10-16"
    assert aware["request"]["covers_end"] == "2026-12-05"


def test_pending_decisions_raises_the_silence_and_not_the_card_already_acted_on():
    """The overdue request is rule 1's whole subject, and rule 4 must stand down.

    Blind to the agent's state, this tool re-raised records-due:2026-09-08 — a
    decision the parent already dealt with — and never raised the documented
    silence at all, because from the file's point of view nothing had been sent.
    """
    day = "2026-12-05"
    aware = _call(pending_decisions, today=day, tool_context=_context_with_a_sent_request())
    ids = {c["card_id"] for c in aware["decisions"]}

    assert "records-silence:req-001:2026-11-29" in ids
    assert "records-due:2026-09-08" not in ids


def test_the_statement_shows_the_requests_the_agent_is_waiting_on():
    statement = _call(
        build_monthly_statement,
        start=TERM_START,
        end=TERM_END,
        today="2026-12-05",
        tool_context=_context_with_a_sent_request(),
    )
    assert statement["unanswered_requests"] == 1


# ---------------------------------------------------------------------------
# Graceful failure: a bad argument is a message, never an exception
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool_fn", "kwargs", "field"),
    [
        (reconcile_services, {"start": "not-a-date", "end": TERM_END}, "start"),
        (reconcile_services, {"start": TERM_START, "end": "12/19/2026"}, "end"),
        (check_deadlines, {"today": "2026-13-45"}, "today"),
        (records_requests_status, {"today": ""}, "today"),
        (draft_records_request, {"today": "yesterday"}, "today"),
        (draft_shortfall_letter, {"start": "soon", "end": TERM_END, "today": TODAY}, "start"),
        (build_monthly_statement, {"start": TERM_START, "end": TERM_END, "today": "n/a"}, "today"),
        (pending_decisions, {"today": "2026-2-30"}, "today"),
    ],
    ids=lambda value: value.__name__ if callable(value) else "",
)
def test_a_bad_date_returns_an_actionable_error_and_does_not_raise(tool_fn, kwargs, field):
    result = _call(tool_fn, **kwargs)

    assert set(result) == {"error"}
    assert field in result["error"]
    # The message tells the model what a good value looks like, so its next
    # attempt is a correction rather than a retry of the same mistake.
    assert "YYYY-MM-DD" in result["error"]


def test_an_inverted_period_is_refused_with_an_explanation():
    result = _call(reconcile_services, start=TERM_END, end=TERM_START)

    assert set(result) == {"error"}
    assert "precedes start" in result["error"]


def test_an_inverted_period_is_refused_by_every_tool_that_takes_one():
    for tool_fn, kwargs in (
        (reconcile_services, {"start": TERM_END, "end": TERM_START}),
        (draft_shortfall_letter, {"start": TERM_END, "end": TERM_START, "today": TODAY}),
        (build_monthly_statement, {"start": TERM_END, "end": TERM_START, "today": TODAY}),
    ):
        result = _call(tool_fn, **kwargs)
        assert "error" in result, tool_fn.__name__


def test_a_non_string_date_is_refused_rather_than_coerced():
    result = _call(check_deadlines, today=20261219)

    assert set(result) == {"error"}
    assert "ISO date string" in result["error"]


@pytest.mark.parametrize(
    ("value", "would_have_meant"),
    [
        # date.fromisoformat accepts ISO week dates, and this one resolves into
        # the PREVIOUS YEAR. Silently re-dating a school district's letter by
        # twelve months is the failure this rejection exists to prevent.
        ("2026-W01-1", date(2025, 12, 29)),
        # Basic format: same day, different shape, and accepting it means the
        # accepted set is not the one the docstring and the error both name.
        ("20261219", date(2026, 12, 19)),
    ],
)
def test_a_date_that_is_not_yyyy_mm_dd_is_refused_even_when_python_would_parse_it(
    value, would_have_meant
):
    assert date.fromisoformat(value) == would_have_meant  # what would have happened

    result = _call(check_deadlines, today=value)
    assert set(result) == {"error"}
    assert "YYYY-MM-DD" in result["error"]


def test_an_unknown_case_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown case"):
        load_case_record("someone-else")


def test_a_missing_ledger_fixture_names_the_script_that_rebuilds_it(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "FIXTURES", tmp_path)
    tools._read_case.cache_clear()
    try:
        result = _call(load_case)
        assert "error" in result
        assert "extract_once.py" in result["error"]
    finally:
        tools._read_case.cache_clear()


# ---------------------------------------------------------------------------
# The data seam
# ---------------------------------------------------------------------------


def test_the_case_record_is_the_one_place_a_real_data_source_plugs_in():
    case = load_case_record()

    assert isinstance(case, CaseRecord)
    assert isinstance(case.ledger, IEPLedger)
    assert all(isinstance(e, ServiceEvent) for e in case.events)
    assert all(isinstance(r, RecordsRequest) for r in case.requests)
    assert case.case_id == tools.DEFAULT_CASE


def test_swapping_the_data_source_changes_every_tool(monkeypatch):
    """The proof that fixtures are not wired through the tools: replace the
    seam and the whole surface follows."""
    real = load_case_record()
    trimmed = CaseRecord(
        case_id=real.case_id,
        ledger=real.ledger.model_copy(
            update={"obligations": real.ledger.obligations[:1], "student_alias": "Other C."}
        ),
        events=[],
        requests=[],
        correspondence_items=0,
    )
    monkeypatch.setattr(tools, "load_case_record", lambda *a, **k: trimmed)

    case = _call(load_case)
    assert case["student"] == "Other C."
    assert len(case["obligations"]) == 1
    assert case["evidence"]["events"] == 0

    result = _call(reconcile_services, start=TERM_START, end=TERM_END)
    assert len(result["services"]) == 1
    assert result["totals"]["delivered_minutes"] == 0


def test_decision_cards_come_from_the_decisions_engine(monkeypatch):
    """The tool renders cards; it never decides what is worth an interruption."""
    card = DecisionCard(
        card_id="stub-1",
        title="A stub decision",
        why_now="Because the engine said so.",
        facts=["one fact"],
        recommended_action="Do the thing.",
        urgency="routine",
        deadline=date(2027, 1, 1),
    )
    monkeypatch.setattr(tools, "decisions_for", lambda *a, **k: [card])

    result = _call(pending_decisions, today=TODAY)
    assert result["count"] == 1
    assert result["decisions"][0]["card_id"] == "stub-1"
    assert result["decisions"][0]["deadline"] == "2027-01-01"
    assert result["decisions"][0]["has_draft_letter"] is False


# ---------------------------------------------------------------------------
# No live model calls
# ---------------------------------------------------------------------------


def test_the_tool_layer_never_reaches_a_model(monkeypatch):
    """Every tool runs over the cached ledger and cached events. If any of them
    ever calls extraction or classification, this fails rather than quietly
    spending money on a background run."""
    import minutes.correspondence as correspondence
    import minutes.extraction as extraction

    def explode(*args, **kwargs):
        raise AssertionError("a tool attempted a live model call")

    monkeypatch.setattr(extraction, "extract_ledger", explode)
    monkeypatch.setattr(correspondence, "classify_to_events", explode)

    _call(load_case)
    _call(reconcile_services, start=TERM_START, end=TERM_END)
    _call(check_deadlines, today=TODAY)
    _call(records_requests_status, today=TODAY)
    _call(draft_records_request, today=TODAY)
    _call(draft_shortfall_letter, start=TERM_START, end=TERM_END, today=TODAY)
    _call(build_monthly_statement, start=TERM_START, end=TERM_END, today=TODAY)
    _call(pending_decisions, today=TODAY)
