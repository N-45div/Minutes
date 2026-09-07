"""The caseworker's wiring, exercised for real — with a scripted model, not a mock.

WHY A FAKE MODEL AND NOT A STUB OF THE SDK. The properties this file has to
prove are properties of the Strands event loop, not of this package: that an
interrupt raised inside a tool actually halts the run before the tool acts; that
the parent's answer actually flows back into the tool body on resume; that the
after-tool hook fires exactly once for a call whose body was entered twice; that
a pending approval actually survives a process boundary through the session
store. Mocking the agent would assert that the mock behaves as expected. So
these tests drive the real ``Agent``, the real hook registry, the real interrupt
state and the real ``FileSessionManager``, and replace only the one component
that would cost money and need credentials — the model — with
:class:`ScriptedModel`, which emits the exact stream events a Bedrock model
would for a scripted sequence of tool calls.

The pieces that need no event loop at all (reading an answer, filling a
placeholder, digesting a letter) are tested directly, because wrapping them in
an agent would only make the failure harder to read.

NOTHING HERE MAKES A NETWORK CALL, AN AWS CALL, OR AN LLM CALL. Every agent in
this file is constructed with an explicit ScriptedModel, so the Bedrock branch
of ``build_caseworker`` is never taken and no boto3 client is ever created.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from strands import Agent
from strands.interrupt import Interrupt, InterruptException
from strands.models import Model

from minutes import agent as caseworker_module
from minutes.agent import (
    ACTOR,
    APPROVE,
    DECLINE,
    DEFAULT_LIMITS,
    MINUTES_SYSTEM_PROMPT,
    STATE_AUDIT,
    STATE_OUTBOX,
    AuditTrail,
    Caseworker,
    audit_entries,
    build_caseworker,
    fill_placeholders,
    is_approval,
    letter_digest,
    read_answer,
    record_action,
    send_letter,
    send_records_request,
)
from minutes.letters import compile_records_request, placeholders, validate_letter
from minutes.models import AuditEntry, Provenance, RequestState
from minutes.tools import CaseRecord, load_case_record

TODAY = "2026-10-15"
TERM_START = "2026-09-08"
TERM_END = "2026-12-19"

PARENT_WORDS = "I work day shifts at the hospital and cannot attend during school hours."


# ---------------------------------------------------------------------------
# The scripted model.
# ---------------------------------------------------------------------------


class ScriptedModel(Model):
    """A model that says exactly what the test tells it to, and nothing else.

    A turn is ``(tool_name, arguments)`` — emitted as a real tool-use content
    block, the way Bedrock would — or a LIST of those, emitted as several blocks
    in one assistant message, which is how a real model asks for two tools at
    once and is what makes Strands run them concurrently — or a string, emitted
    as text with ``end_turn``. When the script runs out the last turn repeats,
    so a test only has to script the turns it cares about.

    ``calls`` is asserted on directly by the tests that care whether the event
    loop re-called the model (it must not, for the turn an interrupt paused).
    """

    def __init__(self, turns: list[Any]):
        self._turns = list(turns)
        self.calls = 0
        self.tool_specs_seen: list[str] = []

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> dict:
        return {}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError("Minutes never asks a model for structured output")

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        turn = self._turns[min(self.calls, len(self._turns) - 1)]
        self.calls += 1
        self.tool_specs_seen = [spec["name"] for spec in tool_specs or []]

        yield {"messageStart": {"role": "assistant"}}
        if isinstance(turn, (tuple, list)):
            for name, arguments in [turn] if isinstance(turn, tuple) else turn:
                yield {
                    "contentBlockStart": {
                        "start": {"toolUse": {"name": name, "toolUseId": f"tu-{name}-{self.calls}"}}
                    }
                }
                yield {
                    "contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(arguments)}}}
                }
                yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": str(turn)}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
                "metrics": {"latencyMs": 0},
            }
        }


def worker(tmp_path: Path, turns: list[Any], **kwargs) -> Caseworker:
    """A caseworker with a scripted model and a real, temporary audit file."""
    kwargs.setdefault("trail_path", tmp_path / "trail.jsonl")
    return build_caseworker(model=ScriptedModel(turns), **kwargs)


def approving(answer: Any = APPROVE):
    return lambda interrupt: answer


# ---------------------------------------------------------------------------
# Wiring: what the agent is allowed to do at all.
# ---------------------------------------------------------------------------


def test_every_tool_is_registered(tmp_path):
    hand = worker(tmp_path, ["done"])
    assert set(hand.agent.tool_names) == {
        "load_case",
        "reconcile_services",
        "check_deadlines",
        "records_requests_status",
        "draft_records_request",
        "draft_shortfall_letter",
        "build_monthly_statement",
        "pending_decisions",
        "read_correspondence_item",
        "audit_trail",
        "record_request_delivery",
        "send_records_request",
        "send_letter",
    }


def test_the_model_never_gets_a_parameter_it_could_write_a_letter_into(tmp_path):
    """The send tools take identifiers. There is no prose parameter, anywhere.

    This is the structural reason Minutes cannot fabricate an accusation: the
    letter is recompiled inside the tool from the ledger and the evidence, so
    the model's only influence on what a district receives is which document it
    proposes and when.
    """
    hand = worker(tmp_path, ["done"])
    registry = hand.agent.tool_registry.registry

    for name in ("send_records_request", "send_letter"):
        schema = registry[name].tool_spec["inputSchema"]["json"]
        assert set(schema.get("properties", {})) <= {
            "kind",
            "today",
            "start",
            "end",
            "deadline_due",
        }, name


def test_the_send_tools_say_they_pause(tmp_path):
    """The description the model reads must not let it think it can just send."""
    hand = worker(tmp_path, ["done"])
    registry = hand.agent.tool_registry.registry

    for name in ("send_records_request", "send_letter"):
        description = registry[name].tool_spec["description"]
        assert "parent" in description
        assert "declin" in description


def test_the_system_prompt_carries_the_disciplines():
    prompt = MINUTES_SYSTEM_PROMPT.lower()
    assert "undocumented" in prompt
    assert "never characterise the district" in prompt
    assert "quiet is success" in prompt
    assert "never give legal advice" in prompt
    assert "you never write a letter" in prompt


def test_the_system_prompt_protects_the_child_as_well_as_the_district():
    """The rules used to guard everyone in the case except the person it is about.

    load_case hands the model a bare count of dates a record notes Maya was
    away, and her accommodations verbatim. Nothing in the prompt stopped it
    writing "she was absent three times, which is part of why her speech
    minutes are behind" into a permanent case record.
    """
    prompt = MINUTES_SYSTEM_PROMPT.lower()

    assert "never characterise the child" in prompt
    assert "never a reason the services fell short" in prompt
    assert "accommodations describe what the district owes her" in prompt


def test_the_system_prompt_never_lets_the_model_say_a_letter_was_sent():
    """Minutes has no mail channel, and "released" is not "sent".

    A parent told their records request was sent on 15 October believes the
    45-day clock is running from that date, and has no reason to post it
    promptly or by a method that proves delivery.
    """
    prompt = MINUTES_SYSTEM_PROMPT.lower()

    assert "minutes cannot post anything" in prompt
    assert "never say a letter was sent" in prompt
    assert "record_request_delivery" in prompt
    assert "not from an estimate" in prompt
    assert "a date you invented would be an accusation you invented" in prompt


def test_no_model_id_is_hardcoded():
    """Model ids come from config, so MINUTES_MODEL is the only lever."""
    source = (Path(__file__).resolve().parents[1] / "minutes" / "agent.py").read_text("utf-8")
    assert "anthropic.claude" not in source
    assert "EXTRACTION_MODEL" in source


def test_execution_limits_are_set_and_passed(tmp_path):
    """No constructor turn cap exists in Strands; the invocation limit is it."""
    assert DEFAULT_LIMITS["turns"] == 12
    assert DEFAULT_LIMITS["output_tokens"] > 0
    assert DEFAULT_LIMITS["total_tokens"] > 0

    hand = worker(tmp_path, ["done"])
    assert hand.limits == DEFAULT_LIMITS


def test_a_budget_trip_is_recorded_because_nothing_raises(tmp_path):
    """Hitting a limit stops the loop silently. A truncated run must not look finished."""
    hand = worker(
        tmp_path,
        [("check_deadlines", {"today": TODAY})],
        limits={"turns": 1},
    )
    result = hand.ask("check everything")

    assert result.stop_reason == "limit_turns"
    assert any(entry.action == "stopped at a budget limit" for entry in hand.audit())


def test_a_budget_trip_survives_the_process_that_hit_it(tmp_path):
    """The entry saying a run was truncated is written after the last message.

    State written after the final message of a run is not synced by the session
    manager on its own, so the one record that a scheduled run did not finish
    would be the one record lost on exit.
    """
    store = tmp_path / "sessions"
    hand = worker(
        tmp_path,
        [("check_deadlines", {"today": TODAY})],
        limits={"turns": 1},
        session_id="maya-truncated",
        storage_dir=store,
    )
    hand.ask("check everything")

    later = worker(tmp_path, ["done"], session_id="maya-truncated", storage_dir=store)
    assert any(entry.action == "stopped at a budget limit" for entry in later.audit())


# ---------------------------------------------------------------------------
# The paper trail.
# ---------------------------------------------------------------------------


def test_the_hook_records_every_tool_call(tmp_path):
    hand = worker(
        tmp_path,
        [("reconcile_services", {"start": TERM_START, "end": TERM_END}), "done"],
    )
    hand.ask("how is the term going")

    entry = next(e for e in hand.audit() if e.action == "called reconcile_services")
    assert entry.actor == ACTOR
    assert "start='2026-09-08'" in entry.detail
    assert "-> success" in entry.detail


def test_the_whole_trail_runs_on_one_clock(tmp_path):
    """Hook entries and tool entries must not be dated by two different clocks.

    In production they are the same day and the difference is invisible. In a
    replay it is the difference between a trail a parent can follow and one that
    jumps between the date of the case and the date of the machine.
    """
    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "sent"])
    hand.work(
        "ask the district for its records",
        approving({"answer": APPROVE, "fill": [PARENT_WORDS]}),
    )

    dated = [e.entry_date for e in hand.audit()]
    assert dated == [date(2026, 10, 15)] * len(dated)


def test_an_action_with_no_case_date_falls_back_to_today(tmp_path):
    """audit_trail takes no dates, so there is nothing to date it by but now."""
    hand = worker(tmp_path, [("audit_trail", {}), "done"])
    hand.ask("what have you done")

    entry = next(e for e in hand.audit() if e.action == "called audit_trail")
    assert entry.entry_date == date.today()


def test_the_trail_mirrors_to_an_append_only_file(tmp_path):
    path = tmp_path / "nested" / "trail.jsonl"
    hand = worker(tmp_path, [("check_deadlines", {"today": TODAY}), "done"], trail_path=path)
    hand.ask("check the dates")

    lines = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    assert lines
    assert lines[0]["entry"]["actor"] == ACTOR
    # AuditEntry carries a date, not a timestamp -- models.py is frozen -- so
    # clock order lives beside the typed entry, not inside it.
    assert "recorded_at" in lines[0]
    assert "T" in lines[0]["recorded_at"]


def test_an_unwritable_trail_file_does_not_take_the_run_down(tmp_path):
    """It also does not vanish quietly: the failure is itself recorded."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")

    hand = worker(
        tmp_path,
        [("check_deadlines", {"today": TODAY}), "done"],
        trail_path=blocker / "trail.jsonl",
    )
    result = hand.ask("check the dates")

    assert result.stop_reason == "end_turn"
    assert any(entry.action == "audit file unwritable" for entry in hand.audit())


def test_the_trail_tool_says_out_loud_that_it_is_capped(tmp_path):
    """The first line is what steers the model, and it used to promise everything.

    The payload has always been the last hundred entries. "Everything Minutes
    has done on this case" is the sentence the model composes from, so it would
    hand a parent a hundred rows and call them the whole record.
    """
    hand = worker(tmp_path, ["done"])
    for n in range(150):
        record_action(
            hand.agent,
            AuditEntry(entry_date=date(2026, 10, 15), actor=ACTOR, action=f"e{n}", detail="x"),
        )

    payload = caseworker_module.audit_trail(tool_context=_Ctx(hand.agent))

    assert payload["count"] == 150
    assert payload["showing"] == 100
    assert payload["truncated"] is True
    description = caseworker_module.audit_trail.tool_spec["description"]
    assert description.startswith("The most recent 100 things Minutes has done")
    assert "there is older" in description


class _Ctx:
    """The one attribute audit_trail reads off a ToolContext."""

    def __init__(self, agent):
        self.agent = agent


def test_the_trail_tool_hands_the_record_over(tmp_path):
    hand = worker(
        tmp_path,
        [
            ("check_deadlines", {"today": TODAY}),
            ("audit_trail", {}),
            "here is everything I did",
        ],
    )
    hand.ask("what have you been doing")

    trail = hand.audit()
    assert [entry.action for entry in trail][:2] == [
        "called check_deadlines",
        "called audit_trail",
    ]


def test_record_action_is_the_only_way_in(tmp_path):
    """Entries written by a tool and entries written by the hook share one path."""
    hand = worker(tmp_path, ["done"])
    record_action(
        hand.agent,
        AuditEntry(entry_date=date(2026, 10, 15), actor=ACTOR, action="test", detail="one"),
    )

    assert [entry.action for entry in hand.audit()] == ["test"]
    assert len(hand.agent.state.get(STATE_AUDIT)) == 1


def test_concurrent_writers_cannot_lose_an_entry_to_each_other(tmp_path):
    """A dropped row is precisely the failure a paper trail exists to prevent.

    ``agent.state`` has no concurrency control and ``get``/``set`` deep-copy, so
    an append is read-copy-mutate-copy-write with a window that grows with the
    trail. Strands' default executor runs a batch of tool calls concurrently —
    each sync tool body on its own worker thread — while the after-tool hook
    writes from the event-loop thread, so two writers really do overlap.
    Unlocked, this loses entries; the JSONL mirror is per-entry and stays
    complete, which is what makes the loss silent.

    The backlog is sized to about a year of weekly runs, because that is what
    widens the window enough for the race to be reproducible rather than hoped
    for. Measured on this venv with the lock removed, all six of six trials at
    these numbers lost entries (five to nine of 120); with the lock, none did.
    """
    import threading

    hand = worker(tmp_path, ["done"], trail_path=tmp_path / "race.jsonl")
    backlog = [
        {
            "recorded_at": "2026-01-01T00:00:00+00:00",
            "entry": AuditEntry(
                entry_date=date(2026, 1, 1), actor=ACTOR, action="old", detail="x" * 300
            ).model_dump(mode="json"),
        }
        for _ in range(1_500)
    ]
    hand.agent.state.set(STATE_AUDIT, backlog)

    def write(n: int) -> None:
        record_action(
            hand.agent,
            AuditEntry(
                entry_date=date(2026, 10, 15), actor=ACTOR, action=f"concurrent-{n}", detail="one"
            ),
        )

    threads = [threading.Thread(target=write, args=(n,)) for n in range(120)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    trail = hand.audit()
    written = {e.action for e in trail if e.action.startswith("concurrent-")}
    mirror = (tmp_path / "race.jsonl").read_text("utf-8").splitlines()

    assert written == {f"concurrent-{n}" for n in range(120)}
    assert len(trail) == len(backlog) + 120
    # The state copy and the append-only mirror must agree. They are the copy a
    # parent is handed and the copy that survives the process — and it was the
    # mirror staying complete while state lost rows that made the loss silent.
    assert len(mirror) == 120


def test_two_send_tools_in_one_turn_keep_both_of_their_entries(tmp_path):
    """The same race, driven through the product's own path.

    A model emitting two tool-use blocks in one assistant message is ordinary,
    and Strands runs them concurrently by default.
    """
    hand = worker(
        tmp_path,
        [
            [
                ("send_records_request", {"today": TODAY}),
                (
                    "send_letter",
                    {"kind": "deadline_reminder", "today": TODAY, "deadline_due": "2026-11-06"},
                ),
            ],
            "both are ready to send",
        ],
        trail_path=tmp_path / "batch.jsonl",
    )
    hand.work("ask for the records and chase the progress report", approving(DECLINE))

    actions = [entry.action for entry in hand.audit()]
    assert "asked the parent to approve a records request" in actions
    assert "asked the parent to approve a deadline_reminder" in actions
    assert "parent declined a records request" in actions
    assert "parent declined a deadline_reminder" in actions

    mirror = (tmp_path / "batch.jsonl").read_text("utf-8").splitlines()
    assert len(hand.audit()) == len(mirror)


def test_a_question_nobody_answered_is_still_on_the_record(tmp_path):
    """The most consequential moment in the product, recorded before the pause.

    The after-tool hook never fires for an interrupted call — the executor skips
    it, because a halted tool has no result — so without an entry written before
    the interrupt, "Minutes compiled a letter to a school district and asked to
    send it" would be recorded nowhere until the parent answered, and nowhere at
    all if they never did.
    """
    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "sent"])
    paused = hand.ask("ask the district for its records")

    assert paused.stop_reason == "interrupt"
    entry = next(
        e for e in hand.audit() if e.action == "asked the parent to approve a records request"
    )
    assert hand.pending(paused)[0].reason["letter_digest"] in entry.detail
    assert "nothing has been released" in entry.detail
    assert "If no answer follows this entry, none was given." in entry.detail
    # It is on disk too, so it survives a process that is never resumed.
    assert entry.action in (tmp_path / "trail.jsonl").read_text("utf-8")


def test_the_question_is_recorded_once_even_though_the_tool_body_re_enters(tmp_path):
    """One question, one entry. The tool runs again from the top on resume."""
    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "sent"])
    hand.work(
        "ask the district for its records",
        approving({"answer": APPROVE, "fill": [PARENT_WORDS]}),
    )

    asked = [
        e for e in hand.audit() if e.action == "asked the parent to approve a records request"
    ]
    assert len(asked) == 1


# ---------------------------------------------------------------------------
# Interrupts: the parent's answer decides what happens.
# ---------------------------------------------------------------------------


def test_a_send_tool_pauses_instead_of_acting(tmp_path):
    """The letter is compiled and shown. Nothing has left the family."""
    model = ScriptedModel([("send_records_request", {"today": TODAY}), "sent"])
    hand = build_caseworker(model=model, trail_path=tmp_path / "trail.jsonl")

    result = hand.ask("ask the district for its records")

    assert result.stop_reason == "interrupt"
    pending = hand.pending(result)
    assert len(pending) == 1

    reason = pending[0].reason
    assert reason["action"] == "send_records_request"
    assert reason["nothing_is_sent_unless_you_approve"] is True
    assert reason["letter_kind"] == "records_request"
    assert "34 CFR 300.613(a)" in reason["body"]
    assert reason["citations"] > 0

    assert hand.outbox() == []
    assert all(request.state is RequestState.DRAFT for request in hand.requests())
    # The paused turn did not cost a second model call.
    assert model.calls == 1


def test_the_interrupt_shows_the_blank_only_the_parent_can_fill(tmp_path):
    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "sent"])
    result = hand.ask("ask the district for its records")

    reason = hand.pending(result)[0].reason
    assert len(reason["blanks_only_you_can_fill"]) == 1
    assert "on-site inspection" in reason["blanks_only_you_can_fill"][0]
    assert reason["blocking_before_send"]


def test_an_approval_with_the_parents_own_words_releases_the_letter(tmp_path):
    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "sent"])
    result = hand.work(
        "ask the district for its records",
        approving({"answer": APPROVE, "fill": [PARENT_WORDS]}),
    )

    assert result.stop_reason == "end_turn"

    released = hand.outbox()
    assert len(released) == 1
    assert released[0]["reference"] == "req-001"
    assert released[0]["released_on"] == TODAY
    assert PARENT_WORDS in released[0]["body"]
    assert "[[PARENT:" not in released[0]["body"]

    actions = [entry.action for entry in hand.audit()]
    assert "released a letter for delivery" in actions


def test_approving_does_not_start_the_statutory_clock(tmp_path):
    """THE CLOCK RUNS FROM RECEIPT, and approval is not receipt.

    Minutes cannot post anything, so between the parent approving and the
    district receiving there is a postal delay only they can measure. Starting
    the 45 days of 34 CFR 300.613(a) at approval would make response_due early
    by exactly that delay — and everything downstream treats that date as fact:
    a DOCUMENTED_SILENCE event is minted on it, a card states the district
    produced nothing inside the time the regulation allows, and an otherwise
    unevidenced gap becomes escalatable because of it. That is an accusation
    manufactured out of a postal delay.
    """
    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "sent"])
    hand.work(
        "ask the district for its records",
        approving({"answer": APPROVE, "fill": [PARENT_WORDS]}),
    )

    still_a_draft = next(r for r in hand.requests() if r.request_id == "req-001")
    assert still_a_draft.state is RequestState.DRAFT
    assert still_a_draft.sent_on is None
    assert still_a_draft.response_due is None

    # And no silence can be derived from it, on any date, because there is no
    # deadline for the district to have missed.
    from minutes.discovery import refresh_states, silence_events

    much_later = refresh_states(hand.requests(), date(2027, 6, 1))
    assert silence_events(much_later, load_case_record().ledger, date(2027, 6, 1)) == []


def test_the_clock_starts_when_the_parent_reports_the_district_received_it(tmp_path):
    """The one fact only the family has, and the only thing that starts the 45 days.

    Received 2026-10-27 — twelve days after the parent approved it, which is
    twelve days of deadline that the old behaviour invented — so the response is
    due 2026-12-11, not 2026-11-29.
    """
    hand = worker(
        tmp_path,
        [
            ("send_records_request", {"today": TODAY}),
            ("record_request_delivery", {"request_id": "req-001", "received_on": "2026-10-27"}),
            "the clock is running",
        ],
    )
    hand.work(
        "ask the district for its records, then record the delivery",
        approving({"answer": APPROVE, "fill": [PARENT_WORDS]}),
    )

    sent = next(r for r in hand.requests() if r.request_id == "req-001")
    assert sent.state is RequestState.SENT
    assert sent.sent_on == date(2026, 10, 27)
    assert sent.response_due == date(2026, 12, 11)  # 45 calendar days from receipt

    entry = next(
        e for e in hand.audit() if e.action == "recorded a records request as received"
    )
    assert "2026-10-27" in entry.detail
    assert entry.evidence is not None
    assert entry.evidence.provenance is Provenance.PARENT_OBSERVED


def test_a_delivery_cannot_be_recorded_for_a_letter_that_was_never_released(tmp_path):
    """No approval, no release, no receipt — and therefore no deadline."""
    hand = worker(
        tmp_path,
        [
            ("record_request_delivery", {"request_id": "req-001", "received_on": TODAY}),
            "I could not do that",
        ],
    )
    hand.ask("the district got the request")

    assert all(r.state is RequestState.DRAFT for r in hand.requests())
    assert "no records request 'req-001' is on this case" in json.dumps(hand.agent.messages)


def test_a_delivery_dated_before_the_release_is_refused(tmp_path):
    """A district cannot receive a letter before the parent had it."""
    hand = worker(
        tmp_path,
        [
            ("send_records_request", {"today": TODAY}),
            ("record_request_delivery", {"request_id": "req-001", "received_on": "2026-10-01"}),
            "I could not do that",
        ],
    )
    hand.work(
        "ask the district for its records",
        approving({"answer": APPROVE, "fill": [PARENT_WORDS]}),
    )

    assert next(r for r in hand.requests() if r.request_id == "req-001").response_due is None
    assert "precedes the day req-001 was released" in json.dumps(hand.agent.messages)


def test_an_imminent_iep_meeting_shortens_the_deadline_the_clock_uses(tmp_path, monkeypatch):
    """34 CFR 300.613(a) stacks three deadlines and the earliest governs.

    The cards already cite the meeting; the clock has to use it too, or the
    file states a deadline it does not hold. Annual review moved to 2026-11-20,
    receipt on 2026-10-27: 45 days would be 2026-12-11, the meeting is sooner,
    so the meeting is the deadline.
    """
    original = load_case_record()
    with_meeting = CaseRecord(
        case_id=original.case_id,
        ledger=original.ledger.model_copy(
            update={
                "deadlines": [
                    d.model_copy(update={"due": date(2026, 11, 20)})
                    if d.kind.value == "annual_review"
                    else d
                    for d in original.ledger.deadlines
                ]
            }
        ),
        events=original.events,
        requests=original.requests,
        correspondence_items=original.correspondence_items,
    )
    monkeypatch.setattr(caseworker_module, "load_case_record", lambda *a, **k: with_meeting)

    hand = worker(
        tmp_path,
        [
            ("send_records_request", {"today": TODAY}),
            ("record_request_delivery", {"request_id": "req-001", "received_on": "2026-10-27"}),
            "the clock is running",
        ],
    )
    hand.work(
        "ask the district for its records, then record the delivery",
        approving({"answer": APPROVE, "fill": [PARENT_WORDS]}),
    )

    sent = next(r for r in hand.requests() if r.request_id == "req-001")
    assert sent.response_due == date(2026, 11, 20)

    entry = next(
        e for e in hand.audit() if e.action == "recorded a records request as received"
    )
    assert "shortened from the 45-day ceiling" in entry.detail


def test_a_request_we_sent_is_graded_as_our_own_record(tmp_path):
    """A letter the family wrote is not the district's admission of anything.

    Grading it SCHOOL_CONFIRMED would promote our own paperwork into the
    strongest evidence class in the ledger. The district has confirmed nothing
    by being written to; only what it produces, or fails to produce by its
    deadline, is evidence about the district.
    """
    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "sent"])
    hand.work(
        "ask the district for its records",
        approving({"answer": APPROVE, "fill": [PARENT_WORDS]}),
    )

    entry = next(e for e in hand.audit() if e.action == "released a letter for delivery")
    assert entry.evidence is not None
    assert entry.evidence.provenance is Provenance.PARENT_OBSERVED
    assert entry.evidence.source == "req-001"


def test_an_approval_without_the_parents_words_is_held_not_sent(tmp_path):
    """Approving does not override the send gate. The model may not fill the blank."""
    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "held"])
    hand.work("ask the district for its records", approving(APPROVE))

    assert hand.outbox() == []
    assert all(request.state is RequestState.DRAFT for request in hand.requests())

    entry = next(e for e in hand.audit() if e.action == "held an approved records request")
    assert "unfilled parent placeholder" in entry.detail


def test_a_denial_is_recorded_as_carefully_as_an_approval(tmp_path):
    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "understood"])
    result = hand.work("ask the district for its records", approving(DECLINE))

    assert result.stop_reason == "end_turn"
    assert hand.outbox() == []
    assert all(request.state is RequestState.DRAFT for request in hand.requests())

    entry = next(e for e in hand.audit() if e.action == "parent declined a records request")
    assert "'decline'" in entry.detail
    assert "Nothing was released" in entry.detail
    assert "cadence will offer it again" in entry.detail


def test_a_declined_letter_is_kept_whole_and_not_just_summarised(tmp_path):
    """"Recorded as carefully as an approval" has to include the document.

    An approval stores the full released body. A decline that stored only a
    one-line summary would let a hearing officer see THAT a family declined and
    never WHAT they declined — which is the more interesting half.
    """
    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "understood"])
    hand.work("ask the district for its records", approving(DECLINE))

    kept = hand.declined()
    assert len(kept) == 1
    assert kept[0]["reference"] == "req-001"
    assert kept[0]["declined_on"] == TODAY
    assert kept[0]["citations"] > 0
    assert "34 CFR 300.613(a)" in kept[0]["body"]

    # The digest ties the entry to the document beside it.
    entry = next(e for e in hand.audit() if e.action == "parent declined a records request")
    assert kept[0]["letter_digest"] in entry.detail


def test_an_unrecognised_answer_is_a_decline(tmp_path):
    """Fail closed. A letter to a district cannot be recalled."""
    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "understood"])
    hand.work("ask the district for its records", approving("I'm not sure, maybe later"))

    assert hand.outbox() == []
    assert any(e.action == "parent declined a records request" for e in hand.audit())


def test_a_decider_that_answers_nothing_is_read_as_a_decline(tmp_path):
    """``None`` means "unanswered" to the SDK and would loop forever otherwise."""
    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "understood"])
    result = hand.work("ask the district for its records", approving(None))

    assert result.stop_reason == "end_turn"
    assert hand.outbox() == []


def test_one_tool_call_is_one_audit_entry_even_though_the_body_re_enters(tmp_path):
    """The tool is entered twice for one logical call. The trail must not say so.

    Strands invokes the interrupted tool again from the top on resume, so a
    before-tool hook would fire twice. The after-tool hook fires once, because
    the executor skips it for a halted call — which is why the trail is written
    from the after hook.
    """
    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "sent"])
    hand.work(
        "ask the district for its records",
        approving({"answer": APPROVE, "fill": [PARENT_WORDS]}),
    )

    called = [e for e in hand.audit() if e.action == "called send_records_request"]
    assert len(called) == 1


def test_the_side_effect_happens_exactly_once(tmp_path):
    """Everything above the interrupt re-runs; everything below it must not."""
    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "sent"])
    hand.work(
        "ask the district for its records",
        approving({"answer": APPROVE, "fill": [PARENT_WORDS]}),
    )

    assert len(hand.outbox()) == 1
    assert len([e for e in hand.audit() if e.action == "released a letter for delivery"]) == 1


def test_approval_does_not_transfer_to_a_different_letter(tmp_path, monkeypatch):
    """A parent can never approve one letter and have another one released.

    The interrupt's name carries a digest of the compiled body, and Strands
    derives the interrupt id from the name — so a letter recompiled from
    changed evidence simply is not the thing that was approved, and a fresh
    approval is raised for it.
    """
    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "sent"])
    result = hand.ask("ask the district for its records")
    first = hand.pending(result)[0]

    original = load_case_record()
    changed = CaseRecord(
        case_id=original.case_id,
        ledger=original.ledger.model_copy(update={"student_alias": "Someone Else"}),
        events=original.events,
        requests=original.requests,
        correspondence_items=original.correspondence_items,
    )
    monkeypatch.setattr(caseworker_module, "load_case_record", lambda *a, **k: changed)

    resumed = hand.answer([first], approving({"answer": APPROVE, "fill": [PARENT_WORDS]}))

    assert resumed.stop_reason == "interrupt"
    second = hand.pending(resumed)[0]
    assert second.id != first.id
    assert second.reason["letter_digest"] != first.reason["letter_digest"]
    assert hand.outbox() == []


def test_the_guard_re_raises_an_interrupt(tmp_path):
    """The single most dangerous line to get wrong in this whole layer.

    ``InterruptException`` subclasses ``Exception``, so a bare ``except
    Exception`` between the interrupt and the event loop converts a
    human-approval pause into a caught error with no diagnostic: the tool
    reports a failure, the parent is never asked, the letter is never sent.
    """
    interrupt = Interrupt(id="v1:test", name="approve", reason=None)

    @caseworker_module._guard
    def raises_interrupt() -> dict:
        raise InterruptException(interrupt)

    @caseworker_module._guard
    def raises_value_error() -> dict:
        raise ValueError("a bad date")

    with pytest.raises(InterruptException):
        raises_interrupt()
    assert raises_value_error() == {"error": "a bad date"}


# ---------------------------------------------------------------------------
# send_letter: naming a document, never writing one.
# ---------------------------------------------------------------------------


def test_a_shortfall_notice_releases_on_a_plain_approval(tmp_path):
    hand = worker(
        tmp_path,
        [
            (
                "send_letter",
                {
                    "kind": "shortfall_notice",
                    "today": TERM_END,
                    "start": TERM_START,
                    "end": TERM_END,
                },
            ),
            "sent",
        ],
    )
    hand.work("put the arithmetic in front of the district", approving(APPROVE))

    released = hand.outbox()
    assert len(released) == 1
    assert released[0]["kind"] == "shortfall_notice"
    assert released[0]["reference"] == f"shortfall_notice:{TERM_START}:{TERM_END}"


def test_a_declined_letter_leaves_the_outbox_empty(tmp_path):
    hand = worker(
        tmp_path,
        [
            (
                "send_letter",
                {
                    "kind": "shortfall_notice",
                    "today": TERM_END,
                    "start": TERM_START,
                    "end": TERM_END,
                },
            ),
            "understood",
        ],
    )
    hand.work("put the arithmetic in front of the district", approving(DECLINE))

    assert hand.outbox() == []
    assert any(e.action == "parent declined a shortfall_notice" for e in hand.audit())


def test_a_deadline_reminder_is_named_by_its_due_date(tmp_path):
    hand = worker(
        tmp_path,
        [
            (
                "send_letter",
                {"kind": "deadline_reminder", "today": TERM_END, "deadline_due": "2026-11-06"},
            ),
            "sent",
        ],
    )
    hand.work("ask about the progress report", approving(APPROVE))

    released = hand.outbox()
    assert len(released) == 1
    assert released[0]["reference"] == "deadline_reminder:2026-11-06"


@pytest.mark.parametrize(
    "arguments, fragment",
    [
        ({"kind": "make_something_up", "today": TERM_END}, "is not one of"),
        ({"kind": "shortfall_notice", "today": TERM_END}, "needs both start and end"),
        (
            {"kind": "shortfall_notice", "today": TERM_END, "start": TERM_END, "end": TERM_START},
            "precedes start",
        ),
        ({"kind": "deadline_reminder", "today": TERM_END}, "needs deadline_due"),
        (
            {"kind": "deadline_reminder", "today": TERM_END, "deadline_due": "2026-01-01"},
            "no deadline is due",
        ),
        ({"kind": "shortfall_notice", "today": "the end of term"}, "is not an ISO date"),
    ],
)
def test_a_bad_letter_request_comes_back_as_an_error_not_an_exception(
    tmp_path, arguments, fragment
):
    """A raising tool becomes an opaque error the model retries verbatim."""
    hand = worker(tmp_path, [("send_letter", arguments), "I could not do that"])
    result = hand.ask("send something")

    assert result.stop_reason == "end_turn"
    assert hand.outbox() == []
    entry = next(e for e in hand.audit() if e.action == "called send_letter")
    assert "-> success" in entry.detail  # the guard returned a dict, it did not raise

    messages = json.dumps(hand.agent.messages)
    assert fragment in messages


def test_a_compensatory_request_is_refused_while_the_gap_rests_on_missing_records(tmp_path):
    """The escalation gate, in code rather than in a docstring.

    Nothing in decisions.py ever raises a compensatory card — rule 3 attaches a
    shortfall notice and stops — so until this gate the only restraint on a
    make-up-services demand was a sentence the model was asked to obey. Over the
    fixture term the difference is 4,725 minutes of which 4,695 are undocumented:
    99.4% of the headline figure would be minutes nobody recorded either way.
    """
    hand = worker(
        tmp_path,
        [
            (
                "send_letter",
                {
                    "kind": "compensatory_request",
                    "today": TERM_END,
                    "start": TERM_START,
                    "end": TERM_END,
                },
            ),
            "I could not do that",
        ],
    )
    result = hand.ask("demand compensatory services")

    assert result.stop_reason == "end_turn"  # no interrupt: nobody was asked
    assert hand.outbox() == []
    messages = json.dumps(hand.agent.messages)
    assert "is large enough to escalate" in messages
    assert "shortfall_notice" in messages


def test_the_same_letter_approved_twice_is_released_once(tmp_path):
    """Two consents, one document, one thing for the family to post.

    Each release was separately consented — the interrupt ids differ by
    toolUseId, so no approval was reused — but an outbox holding the identical
    letter twice has a family posting duplicates to a district, and a trail
    saying they released the same letter twice reads as a pattern that never
    happened.
    """
    reminder = (
        "send_letter",
        {"kind": "deadline_reminder", "today": TERM_END, "deadline_due": "2026-11-06"},
    )
    hand = worker(tmp_path, [[reminder, reminder], "ready to send"])
    hand.work("chase the progress report", approving(APPROVE))

    released = hand.outbox()
    assert len(released) == 1
    assert released[0]["reference"] == "deadline_reminder:2026-11-06"

    actions = [e.action for e in hand.audit()]
    assert actions.count("released a letter for delivery") == 1
    assert actions.count("collapsed a duplicate release") == 1


def test_a_release_records_both_what_was_shown_and_what_went_out(tmp_path):
    """The parent's own words change the body, so the two digests differ.

    That is exactly when the chain from "what was put in front of somebody" to
    "what was released" needs to be legible in the record rather than inferred.
    """
    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "ready to send"])
    paused = hand.ask("ask the district for its records")
    shown = hand.pending(paused)[0].reason["letter_digest"]
    hand.answer(hand.pending(paused), approving({"answer": APPROVE, "fill": [PARENT_WORDS]}))

    item = hand.outbox()[0]
    assert item["approved_digest"] == shown
    assert item["letter_digest"] != shown  # the blank was filled between the two

    entry = next(e for e in hand.audit() if e.action == "released a letter for delivery")
    assert shown in entry.detail
    assert item["letter_digest"] in entry.detail


def test_an_approval_loop_that_will_not_settle_stops_and_says_so(tmp_path):
    """``limits`` cap each round and say nothing about the number of rounds.

    A letter normally recompiles identically, so its digest and interrupt id are
    stable and the loop converges after one answer. A case that moves underneath
    it mints a fresh approval every time round — here by renaming the student on
    each recompile — and the loop would otherwise never end.
    """
    original = load_case_record()
    seen = {"n": 0}

    def moving_case(*args, **kwargs):
        seen["n"] += 1
        return CaseRecord(
            case_id=original.case_id,
            ledger=original.ledger.model_copy(update={"student_alias": f"Maya {seen['n']}."}),
            events=original.events,
            requests=original.requests,
            correspondence_items=original.correspondence_items,
        )

    hand = worker(tmp_path, [("send_records_request", {"today": TODAY}), "sent"])
    caseworker_module.load_case_record = moving_case
    try:
        result = hand.work(
            "ask the district for its records",
            approving({"answer": APPROVE, "fill": [PARENT_WORDS]}),
        )
    finally:
        caseworker_module.load_case_record = load_case_record

    assert result.stop_reason == "interrupt"  # it gave up while still being asked
    assert hand.outbox() == []
    assert any(
        e.action == "stopped an approval loop that would not settle" for e in hand.audit()
    )


def test_a_records_request_that_is_not_due_is_refused_without_asking_anyone(tmp_path):
    """No interrupt at all: there is nothing to put in front of a parent."""
    hand = worker(tmp_path, [("send_records_request", {"today": TERM_START}), "nothing due"])
    result = hand.ask("ask the district for its records")

    assert result.stop_reason == "end_turn"
    assert hand.outbox() == []
    assert "No records request is due today" in json.dumps(hand.agent.messages)


# ---------------------------------------------------------------------------
# Sessions: the weekly wake-up, across a process boundary.
# ---------------------------------------------------------------------------


def test_the_case_survives_a_process_boundary(tmp_path):
    """A second process sharing only a session id gets the whole case back."""
    store = tmp_path / "sessions"

    monday = worker(
        tmp_path,
        [
            ("send_records_request", {"today": TODAY}),
            ("record_request_delivery", {"request_id": "req-001", "received_on": "2026-10-20"}),
            "posted and delivered",
        ],
        session_id="maya-weekly",
        storage_dir=store,
    )
    monday.work(
        "ask the district for its records, then record the delivery",
        approving({"answer": APPROVE, "fill": [PARENT_WORDS]}),
    )
    monday.sync()

    # A new process: new agent, new session manager, same two ids.
    thursday = worker(tmp_path, ["hello again"], session_id="maya-weekly", storage_dir=store)

    assert len(thursday.outbox()) == 1
    assert thursday.outbox()[0]["reference"] == "req-001"

    restored = next(r for r in thursday.requests() if r.request_id == "req-001")
    assert restored.state is RequestState.SENT
    assert restored.sent_on == date(2026, 10, 20)
    assert restored.response_due == date(2026, 12, 4)  # 45 days from receipt

    assert [e.action for e in thursday.audit()] == [e.action for e in monday.audit()]
    assert thursday.agent.messages


def test_a_records_request_sent_in_one_process_goes_overdue_in_another(tmp_path):
    """The whole point of persisting the state machine, in one assertion."""
    store = tmp_path / "sessions"

    october = worker(
        tmp_path,
        [
            ("send_records_request", {"today": TODAY}),
            ("record_request_delivery", {"request_id": "req-001", "received_on": "2026-10-16"}),
            "posted and delivered",
        ],
        session_id="maya-clock",
        storage_dir=store,
    )
    october.work(
        "ask the district for its records, then record the delivery",
        approving({"answer": APPROVE, "fill": [PARENT_WORDS]}),
    )
    october.sync()

    december = worker(tmp_path, ["done"], session_id="maya-clock", storage_dir=store)
    from minutes.discovery import refresh_states

    # Received 2026-10-16, so due 2026-11-30 and overdue the day after.
    assert [r.response_due for r in december.requests()] == [date(2026, 11, 30)]
    later = refresh_states(december.requests(), date(2026, 12, 1))
    assert [r.state for r in later] == [RequestState.UNANSWERED_OVERDUE]


def test_a_pending_approval_survives_a_process_boundary(tmp_path):
    """Asked on Monday, answered on Thursday, by a process that did not exist yet."""
    store = tmp_path / "sessions"

    monday = worker(
        tmp_path,
        [("send_records_request", {"today": TODAY}), "sent"],
        session_id="maya-pause",
        storage_dir=store,
    )
    paused = monday.ask("ask the district for its records")
    assert paused.stop_reason == "interrupt"
    interrupt_id = monday.pending(paused)[0].id

    thursday_model = ScriptedModel(["sent"])
    thursday = build_caseworker(
        model=thursday_model,
        session_id="maya-pause",
        storage_dir=store,
        trail_path=tmp_path / "trail.jsonl",
    )
    assert thursday.agent._interrupt_state.activated

    result = thursday.ask(
        [
            {
                "interruptResponse": {
                    "interruptId": interrupt_id,
                    "response": {"answer": APPROVE, "fill": [PARENT_WORDS]},
                }
            }
        ]
    )

    assert result.stop_reason == "end_turn"
    assert len(thursday.outbox()) == 1
    # The paused turn was replayed from the stored message, not re-generated.
    assert thursday_model.calls == 1


def test_without_a_session_nothing_is_persisted(tmp_path):
    """The session manager is what makes a case durable; the agent alone is not."""
    hand = worker(tmp_path, ["done"])
    assert hand.session is None
    hand.sync()  # a no-op, and must not raise


# ---------------------------------------------------------------------------
# Reading a parent's answer.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("answer", ["approve", "APPROVE", " Approved ", "yes", "y", "ok", "send"])
def test_approval_words(answer):
    assert is_approval(answer)


@pytest.mark.parametrize(
    "answer", ["decline", "no", "", "not yet", "approve later", None, 1, True, {"answer": "yes"}]
)
def test_everything_else_is_a_decline(answer):
    assert not is_approval(answer)


def test_read_answer_accepts_a_bare_string():
    assert read_answer("approve") == ("approve", [])


def test_read_answer_accepts_an_answer_carrying_the_parents_words():
    assert read_answer({"answer": "approve", "fill": ["because I work"]}) == (
        "approve",
        ["because I work"],
    )


def test_read_answer_does_not_reject_an_unknown_shape():
    """An unrecognised response declines the letter rather than crashing the run."""
    answer, fill = read_answer({"unexpected": True})
    assert not is_approval(answer)
    assert fill == []


# ---------------------------------------------------------------------------
# Filling the one blank a compiler leaves for a parent.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def records_letter():
    case = load_case_record()
    from minutes.discovery import due_requests

    request = due_requests(case.ledger, [], date(2026, 10, 15))[0]
    return compile_records_request(case.ledger, request, today=date(2026, 10, 15))


def test_filling_a_blank_touches_nothing_else(records_letter):
    filled = fill_placeholders(records_letter, [PARENT_WORDS])
    prompt = placeholders(records_letter)[0]

    assert placeholders(filled) == []
    assert PARENT_WORDS in filled.body
    # Every character outside the placeholder span is identical.
    assert filled.body == records_letter.body.replace(f"[[PARENT: {prompt}]]", PARENT_WORDS, 1)
    assert filled.citations == records_letter.citations
    assert validate_letter(filled) == []
    assert validate_letter(filled, for_sending=True) == []


def test_filling_returns_a_copy(records_letter):
    before = records_letter.body
    fill_placeholders(records_letter, [PARENT_WORDS])
    assert records_letter.body == before


def test_an_empty_answer_leaves_the_blank_standing(records_letter):
    for answer in ([], [""], ["   "]):
        filled = fill_placeholders(records_letter, answer)
        assert placeholders(filled) == placeholders(records_letter)
        assert validate_letter(filled, for_sending=True)


def test_a_letter_with_no_blanks_is_unchanged(records_letter):
    filled = fill_placeholders(records_letter, [PARENT_WORDS])
    assert fill_placeholders(filled, ["ignored"]).body == filled.body


def test_a_digest_changes_when_the_letter_does(records_letter):
    other = records_letter.model_copy(update={"body": records_letter.body + " "})
    assert letter_digest(records_letter) == letter_digest(records_letter.model_copy())
    assert letter_digest(records_letter) != letter_digest(other)


# ---------------------------------------------------------------------------
# The tools, called as plain functions. No agent, no model, no event loop.
# ---------------------------------------------------------------------------


def test_the_send_tools_are_still_ordinary_functions():
    """``@tool`` leaves the function callable, which keeps the layer inspectable."""
    assert callable(send_records_request)
    assert callable(send_letter)
    assert send_records_request.tool_spec["name"] == "send_records_request"


def test_the_audit_hook_registers_only_the_after_event():
    """Registering the before event too would double-count an interrupted call."""
    registered: list[Any] = []

    class Recorder:
        def add_callback(self, event_type, callback, **kwargs):
            registered.append(event_type.__name__)

    AuditTrail().register_hooks(Recorder())
    assert registered == ["AfterToolCallEvent"]


def test_an_agent_without_the_trail_path_still_records(tmp_path):
    """The file is a mirror. Losing it must not lose the trail."""
    hand = build_caseworker(model=ScriptedModel([("check_deadlines", {"today": TODAY}), "done"]))
    hand.ask("check the dates")

    assert hand.audit()
    assert hand.agent.state.get(STATE_OUTBOX) is None


def test_audit_entries_round_trip_through_state(tmp_path):
    hand = worker(tmp_path, ["done"])
    entry = AuditEntry(
        entry_date=date(2026, 10, 15), actor=ACTOR, action="checked", detail="nothing to report"
    )
    record_action(hand.agent, entry)

    assert audit_entries(hand.agent) == [entry]


def test_a_second_agent_on_the_same_case_does_not_see_the_first_ones_state(tmp_path):
    """Without a shared session id, two caseworkers are two independent cases."""
    one = worker(tmp_path, ["done"])
    two = worker(tmp_path, ["done"])
    record_action(
        one.agent,
        AuditEntry(entry_date=date(2026, 10, 15), actor=ACTOR, action="only mine", detail="x"),
    )

    assert two.audit() == []


def test_the_caseworker_wraps_rather_than_replaces_the_agent(tmp_path):
    hand = worker(tmp_path, ["done"])
    assert isinstance(hand.agent, Agent)
    assert hand.agent.agent_id == "caseworker"
