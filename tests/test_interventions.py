"""The intervention layer, exercised through the real agent and against the real policy file.

WHAT THIS FILE HAS TO PROVE is that the framework — not the tool body — stops
a letter the parent did not approve, and that it does so without changing
anything about the path the parent does approve. So most tests here drive the
real ``Agent`` with the same :class:`ScriptedModel` as ``test_agent.py`` and
watch what the intervention layer does before, or instead of, the tool. The
handler-level tests build a real ``BeforeToolCallEvent`` on a real agent and
seed its interrupt state by hand, which is the only way to put a claimed
approval in the tool's arguments and a decline in the agent's state at the same
time and watch which one the gate believes.

NOTHING HERE MAKES A NETWORK CALL, AN AWS CALL, OR AN LLM CALL.
"""

import uuid
from dataclasses import replace
from datetime import date

import cedarpy
import pytest
from strands.hooks import BeforeToolCallEvent
from strands.interrupt import Interrupt
from strands.interventions import Deny, Proceed
from test_agent import PARENT_WORDS, TODAY, ScriptedModel, approving, worker

from minutes import agent as caseworker_module
from minutes.agent import (
    APPROVE,
    DECLINE,
    READ_TOOLS,
    RECORD_TOOLS,
    SEND_TOOL_NAMES,
    SEND_TOOLS,
    approval_interrupt_name,
    build_caseworker,
    digest_from_interrupt_name,
    load_case_record,
)
from minutes.interventions import (
    POLICY_PATH,
    CedarAllowlist,
    SendGate,
    build_interventions,
    parent_answer,
    policy_actions,
)

TERM_START = "2026-09-08"
TERM_END = "2026-12-19"
REMINDER = ("send_letter", {"kind": "deadline_reminder", "today": TODAY, "deadline_due": "2026-11-06"})


def counting_case_loads(monkeypatch) -> dict:
    """Count entries into a send tool body: ``load_case_record`` is its first line."""
    seen = {"n": 0}
    real = load_case_record

    def counted(*args, **kwargs):
        seen["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(caseworker_module, "load_case_record", counted)
    return seen


def event_for(hand, tool_name: str, arguments: dict, tool_use_id: str = "tu-1") -> BeforeToolCallEvent:
    """A real before-tool event on a real agent, as the executor would build it."""
    return BeforeToolCallEvent(
        agent=hand.agent,
        selected_tool=hand.agent.tool_registry.registry.get(tool_name),
        tool_use={"toolUseId": tool_use_id, "name": tool_name, "input": arguments},
        invocation_state={},
    )


def seed_answer(hand, tool_name: str, digest: str, response, tool_use_id: str = "tu-1") -> Interrupt:
    """Put a letter interrupt with the parent's answer on the agent, as a resume would."""
    name = approval_interrupt_name(tool_name, digest)
    interrupt = Interrupt(
        id=f"v1:tool_call:{tool_use_id}:{uuid.uuid5(uuid.NAMESPACE_OID, name)}",
        name=name,
        reason={
            "action": tool_name,
            "letter_kind": "shortfall_notice",
            "subject": "Service minutes for the autumn term",
            "body": "compiled body [1]",
            "citations": 1,
            "letter_digest": digest,
            "reference": f"shortfall_notice:{TERM_START}:{TERM_END}",
        },
        response=response,
    )
    hand.agent._interrupt_state.interrupts[interrupt.id] = interrupt
    hand.agent._interrupt_state.activate()
    return interrupt


# ---------------------------------------------------------------------------
# Wiring.
# ---------------------------------------------------------------------------


def test_the_layer_is_on_by_default_and_in_this_order(tmp_path):
    """The gate records a refusal; Cedar only denies. Deny short-circuits, so the recorder goes first."""
    hand = worker(tmp_path, ["done"])
    names = [h.name for h in hand.agent._intervention_registry.handlers]
    assert names == ["minutes:send-gate", "cedar-authorization"]


def test_human_in_the_loop_is_deliberately_absent(tmp_path):
    """Confirm would ask the parent to approve a tool call before the letter exists, then the tool would ask again."""
    hand = worker(tmp_path, ["done"])
    assert not any("human-in-the-loop" in h.name for h in hand.agent._intervention_registry.handlers)


def test_the_layer_can_be_switched_off_for_a_test(tmp_path):
    hand = worker(tmp_path, ["done"], interventions=False)
    assert hand.agent._intervention_registry.handlers == []


# ---------------------------------------------------------------------------
# (a) A send tool without approval is refused before its body runs.
# ---------------------------------------------------------------------------


def test_a_declined_letter_is_refused_before_the_tool_body_re_enters(tmp_path, monkeypatch):
    """The tool body is entered once, to ask. The gate answers the decline; the body never re-runs.

    Without the layer the body is entered twice — once to compile and ask, once
    on resume to read the answer — and it is the body that records the decline.
    With the layer the second entry never happens: the gate reads the answer
    off the agent's interrupt state, refuses the call, and records the decline
    itself, so the trail reads the same and the tool was not trusted to get
    there.
    """
    loads = counting_case_loads(monkeypatch)
    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "understood"])
    result = hand.work("ask the district for its records", approving(DECLINE))

    assert result.stop_reason == "end_turn"
    assert loads["n"] == 1  # asked once; never re-entered
    assert hand.outbox() == []

    declined = next(e for e in hand.audit() if e.action == "parent declined a records request")
    assert "minutes:send-gate intervention refused the call before the tool re-entered" in declined.detail
    assert "'decline'" in declined.detail
    assert hand.declined()[0]["reference"] == "req-001"
    assert hand.declined()[0]["letter_digest"] in declined.detail
    assert "34 CFR 300.613(a)" in hand.declined()[0]["body"]  # kept whole, from what the parent saw

    called = next(e for e in hand.audit() if e.action == "called send_records_request")
    assert "cancelled: DENIED:" in called.detail
    assert "not an approval" in called.detail


def test_without_the_layer_the_body_re_enters_and_records_the_same_decline(tmp_path, monkeypatch):
    """The control: the guarantee in the tool body is intact, and the record reads the same."""
    loads = counting_case_loads(monkeypatch)
    hand = worker(
        tmp_path, [("send_records_request", {"today": TODAY}), "understood"], interventions=False
    )
    hand.work("ask the district for its records", approving(DECLINE))

    assert loads["n"] == 2
    assert hand.outbox() == []
    declined = next(e for e in hand.audit() if e.action == "parent declined a records request")
    assert "intervention refused" not in declined.detail
    assert hand.declined()[0]["reference"] == "req-001"


def test_the_refusal_is_the_gates_and_the_model_is_told_it_is_final(tmp_path):
    hand = worker(tmp_path, [REMINDER, "understood"])
    hand.work("chase the progress report", approving("not now"))

    called = next(e for e in hand.audit() if e.action == "called send_letter")
    assert "DENIED: The parent read this letter" in called.detail
    assert "do not re-ask" in called.detail
    assert any(e.action == "parent declined a deadline_reminder" for e in hand.audit())
    assert hand.declined()[0]["kind"] == "deadline_reminder"


# ---------------------------------------------------------------------------
# (b) The interrupt -> approve -> release path is untouched.
# ---------------------------------------------------------------------------


def test_an_approved_letter_still_releases_with_the_layer_on(tmp_path):
    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "ready"])
    paused = hand.ask("ask the district for its records")
    assert paused.stop_reason == "interrupt"  # the ask pass went through both handlers

    result = hand.answer(hand.pending(paused), approving({"answer": APPROVE, "fill": [PARENT_WORDS]}))

    assert result.stop_reason == "end_turn"
    assert len(hand.outbox()) == 1
    assert PARENT_WORDS in hand.outbox()[0]["body"]
    actions = [e.action for e in hand.audit()]
    assert "released a letter for delivery" in actions
    assert actions.count("called send_records_request") == 1
    assert not any("DENIED" in e.detail for e in hand.audit())


def test_two_send_tools_in_one_turn_are_judged_separately(tmp_path):
    """One approved, one declined, in the same resumed pass: the gate reads each call's own answer."""
    hand = worker(
        tmp_path,
        [[("send_records_request", {"today": TODAY}), REMINDER], "one is ready"],
    )

    def decide(interrupt):
        if interrupt.reason["action"] == "send_records_request":
            return {"answer": APPROVE, "fill": [PARENT_WORDS]}
        return DECLINE

    hand.work("ask for the records and chase the progress report", decide)

    assert [item["reference"] for item in hand.outbox()] == ["req-001"]
    assert [item["kind"] for item in hand.declined()] == ["deadline_reminder"]


def test_a_letter_that_changed_after_approval_is_asked_again_not_refused(tmp_path, monkeypatch):
    """An approval on file for an old digest is not a decline: the tool must be allowed to recompile and re-ask."""
    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "ready"])
    paused = hand.ask("ask the district for its records")
    first = hand.pending(paused)[0]

    original = load_case_record()
    changed = replace(
        original, ledger=original.ledger.model_copy(update={"student_alias": "Someone Else"})
    )
    monkeypatch.setattr(caseworker_module, "load_case_record", lambda *a, **k: changed)

    resumed = hand.answer([first], approving({"answer": APPROVE, "fill": [PARENT_WORDS]}))

    assert resumed.stop_reason == "interrupt"
    assert hand.pending(resumed)[0].id != first.id
    assert hand.outbox() == []
    assert not any("DENIED" in e.detail for e in hand.audit())


def test_a_pending_approval_answered_in_a_new_process_passes_the_gate(tmp_path):
    """The gate reads interrupt state the session manager restored, not anything held in memory."""
    store = tmp_path / "sessions"
    monday = worker(
        tmp_path, [("send_records_request", {"today": TODAY}), "sent"], session_id="gate-pause", storage_dir=store
    )
    paused = monday.ask("ask the district for its records")
    interrupt_id = monday.pending(paused)[0].id

    thursday = build_caseworker(
        model=ScriptedModel(["sent"]), session_id="gate-pause", storage_dir=store, trail_path=tmp_path / "t.jsonl"
    )
    result = thursday.ask(
        [{"interruptResponse": {"interruptId": interrupt_id, "response": {"answer": APPROVE, "fill": [PARENT_WORDS]}}}]
    )

    assert result.stop_reason == "end_turn"
    assert len(thursday.outbox()) == 1


# ---------------------------------------------------------------------------
# (c) Read and record tools are never blocked.
# ---------------------------------------------------------------------------


def test_every_read_tool_runs_unattended(tmp_path):
    turns = [
        ("load_case", {}),
        ("reconcile_services", {"start": TERM_START, "end": TERM_END}),
        ("check_deadlines", {"today": TODAY}),
        ("records_requests_status", {"today": TODAY}),
        ("draft_records_request", {"today": TODAY}),
        ("draft_shortfall_letter", {"start": TERM_START, "end": TERM_END, "today": TODAY}),
        ("build_monthly_statement", {"start": TERM_START, "end": TERM_END, "today": TODAY}),
        ("pending_decisions", {"today": TODAY}),
        ("audit_trail", {}),
        "quiet week",
    ]
    hand = worker(tmp_path, turns)
    result = hand.ask("check everything")

    assert result.stop_reason == "end_turn"
    for name, _ in turns[:-1]:
        entry = next(e for e in hand.audit() if e.action == f"called {name}")
        assert "cancelled" not in entry.detail, name
        assert "DENIED" not in entry.detail, name


@pytest.mark.parametrize("tool", [t.tool_name for t in (*READ_TOOLS, *RECORD_TOOLS)])
def test_both_handlers_wave_a_read_or_record_tool_through(tmp_path, tool):
    hand = worker(tmp_path, ["done"])
    gate, cedar = hand.agent._intervention_registry.handlers
    event = event_for(hand, tool, {"today": TODAY})

    assert isinstance(gate.before_tool_call(event), Proceed)
    assert isinstance(cedar.before_tool_call(event), Proceed)


# ---------------------------------------------------------------------------
# (d) An unlisted tool is denied by Cedar.
# ---------------------------------------------------------------------------


def test_a_tool_the_policy_does_not_name_is_denied_by_cedar(tmp_path):
    hand = worker(tmp_path, ["done"])
    cedar = hand.agent._intervention_registry.handlers[1]
    verdict = cedar.before_tool_call(event_for(hand, "post_to_district", {"body": "anything"}))

    assert isinstance(verdict, Deny)
    assert "Access denied by Cedar policy" in verdict.reason


def test_a_model_that_invents_a_tool_is_refused_by_policy_and_it_is_on_the_record(tmp_path):
    hand = worker(tmp_path, [("email_the_district", {"text": "hello"}), "I could not do that"])
    result = hand.ask("just email them")

    assert result.stop_reason == "end_turn"
    called = next(e for e in hand.audit() if e.action == "called email_the_district")
    assert "cancelled: DENIED: Access denied by Cedar policy" in called.detail


# ---------------------------------------------------------------------------
# (e) The policy file parses and names exactly the agent's tools.
# ---------------------------------------------------------------------------


def test_the_policy_file_parses():
    text = POLICY_PATH.read_text(encoding="utf-8")
    cedarpy.format_policies(text)  # raises on a syntax error
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))
    assert "forbid" not in code  # deny-by-default does the forbidding
    assert "*" not in code  # no wildcard action anywhere


def test_the_policy_names_exactly_the_tools_on_the_agent(tmp_path):
    """The drift test. A tool registered without a policy line, or a line for a tool that is gone, fails here."""
    hand = worker(tmp_path, ["done"])
    assert policy_actions(POLICY_PATH.read_text(encoding="utf-8")) == set(hand.agent.tool_names)


def test_a_tool_added_without_a_policy_line_would_fail_the_drift_test(tmp_path):
    from strands import tool

    @tool
    def phone_the_district(number: str) -> str:
        """Call the district."""
        return number

    hand = worker(tmp_path, ["done"])
    hand.agent.tool_registry.register_tool(phone_the_district)

    assert policy_actions(POLICY_PATH.read_text(encoding="utf-8")) != set(hand.agent.tool_names)
    assert "phone_the_district" not in policy_actions(POLICY_PATH.read_text(encoding="utf-8"))


def test_the_send_tools_are_the_only_conditional_permits():
    """A reader should find the send tools under the one `when` clause and nowhere else."""
    text = POLICY_PATH.read_text(encoding="utf-8")
    conditional = text[text.index("when {") :]
    assert "parent_answer" in conditional
    for name in SEND_TOOL_NAMES:
        assert text.count(f'Action::"{name}"') == 1
        assert f'Action::"{name}"' in text[: text.index("when {")]
        assert f'Action::"{name}"' in text[text.rindex("permit (") :]


# ---------------------------------------------------------------------------
# (f) Approval cannot be supplied in the tool's arguments.
# ---------------------------------------------------------------------------


def test_the_gate_reads_the_agents_state_not_the_models_arguments(tmp_path):
    """A decline on file and a claimed approval in the arguments: the gate believes the state."""
    hand = worker(tmp_path, ["done"])
    gate, cedar = hand.agent._intervention_registry.handlers
    seed_answer(hand, "send_letter", "abc123def456", DECLINE)

    claimed = {
        "kind": "shortfall_notice",
        "today": TODAY,
        "start": TERM_START,
        "end": TERM_END,
        "approved": True,
        "parent_answer": "approved",
        "letter_digest": "abc123def456",
    }
    assert isinstance(gate.before_tool_call(event_for(hand, "send_letter", claimed)), Deny)
    assert isinstance(cedar.before_tool_call(event_for(hand, "send_letter", claimed)), Deny)
    assert any(e.action == "parent declined a shortfall_notice" for e in hand.audit())


def test_an_approval_on_file_is_what_lets_a_send_tool_through(tmp_path):
    """Same arguments, opposite state: now both handlers proceed."""
    hand = worker(tmp_path, ["done"])
    gate, cedar = hand.agent._intervention_registry.handlers
    seed_answer(hand, "send_letter", "abc123def456", APPROVE)

    plain = {"kind": "shortfall_notice", "today": TODAY, "start": TERM_START, "end": TERM_END}
    assert isinstance(gate.before_tool_call(event_for(hand, "send_letter", plain)), Proceed)
    assert isinstance(cedar.before_tool_call(event_for(hand, "send_letter", plain)), Proceed)


def test_through_the_agent_arguments_claiming_approval_change_nothing(tmp_path, monkeypatch):
    """The model writes an approval into the call. The parent declines. Nothing is released."""
    loads = counting_case_loads(monkeypatch)
    hand = worker(
        tmp_path,
        [("send_records_request", {"today": TODAY, "approved": True, "parent_answer": "approved"}), "ok"],
    )
    hand.work("ask the district for its records", approving(DECLINE))

    assert loads["n"] == 1
    assert hand.outbox() == []
    assert any(e.action == "parent declined a records request" for e in hand.audit())


def test_an_answer_to_another_calls_letter_does_not_count(tmp_path):
    """Approval on tu-1 does not approve tu-2: the state is read per toolUseId, as the SDK binds it."""
    hand = worker(tmp_path, ["done"])
    seed_answer(hand, "send_letter", "abc123def456", APPROVE, tool_use_id="tu-1")

    assert parent_answer(hand.agent, "send_letter", "tu-1").phase == "approved"
    assert parent_answer(hand.agent, "send_letter", "tu-2").phase == "asking"
    # And a records-request answer is not a send_letter answer on the same call.
    assert parent_answer(hand.agent, "send_records_request", "tu-1").phase == "asking"


def test_the_parents_latest_answer_on_a_call_is_the_one_that_counts(tmp_path):
    """The case moved after an approval; the parent declined the new letter. The decline governs."""
    hand = worker(tmp_path, ["done"])
    seed_answer(hand, "send_letter", "old000000000", APPROVE)
    seed_answer(hand, "send_letter", "new000000000", DECLINE)

    verdict = parent_answer(hand.agent, "send_letter", "tu-1")
    assert verdict.phase == "declined"
    assert verdict.digest == "new000000000"


def test_an_unanswered_latest_letter_means_the_tool_will_ask_again(tmp_path):
    hand = worker(tmp_path, ["done"])
    seed_answer(hand, "send_letter", "old000000000", DECLINE)
    seed_answer(hand, "send_letter", "new000000000", None)

    assert parent_answer(hand.agent, "send_letter", "tu-1").phase == "asking"


# ---------------------------------------------------------------------------
# The pieces.
# ---------------------------------------------------------------------------


def test_interrupt_names_round_trip():
    for tool in SEND_TOOLS:
        name = approval_interrupt_name(tool.tool_name, "0123456789ab")
        assert digest_from_interrupt_name(tool.tool_name, name) == "0123456789ab"
    assert digest_from_interrupt_name("send_letter", "send-records-request:0123456789ab") is None
    assert digest_from_interrupt_name("send_letter", "strands:human-in-the-loop") is None
    assert digest_from_interrupt_name("send_letter", "send-letter:") is None


def test_a_broken_gate_fails_closed(tmp_path):
    """A handler error is a denial, not a pass and not a crash of the weekly run."""
    assert SendGate().on_error == "deny"
    assert CedarAllowlist(principal_id="maya").on_error == "deny"


def test_build_interventions_rejects_a_policy_that_does_not_parse(tmp_path):
    bad = tmp_path / "bad.cedar"
    bad.write_text('permit(principal, action == Action::"load_case" resource);', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid Cedar policy"):
        build_interventions(principal_id="maya", policies=bad)


def test_the_gate_dates_a_refusal_by_the_case_date_it_was_given(tmp_path):
    hand = worker(tmp_path, ["done"])
    gate = hand.agent._intervention_registry.handlers[0]
    seed_answer(hand, "send_letter", "abc123def456", DECLINE)
    gate.before_tool_call(event_for(hand, "send_letter", {"kind": "shortfall_notice", "today": "2026-11-02"}))

    entry = next(e for e in hand.audit() if e.action == "parent declined a shortfall_notice")
    assert entry.entry_date == date(2026, 11, 2)
    assert hand.declined()[0]["declined_on"] == "2026-11-02"
