"""The AgentCore entrypoint, exercised without a runtime, a model, or a network.

``app.invoke`` is a plain function under the decorator, so it can be called
the way the runtime would call it. Only the model-free actions are covered
here; the caseworker actions are covered by tests/test_agent.py with a stub
model, and wiring them through HTTP adds nothing a hermetic test can see.
"""

from types import SimpleNamespace

import pytest

import app

TERM_START = "2026-09-08"
TERM_END = "2027-01-29"


def _invoke(payload: dict, session_id: str | None = None) -> dict:
    context = SimpleNamespace(session_id=session_id)
    return app.invoke(payload, context)


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Every test gets its own session store and an empty caseworker cache."""
    monkeypatch.setattr(app, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(app, "STATE_BUCKET", None)
    monkeypatch.setattr(app, "_caseworkers", {})
    # No test here may reach a model. A test that needs the caseworker's answer
    # replaces ask on its own worker instance (see _recording_ask below).
    monkeypatch.setattr(
        app.Caseworker, "ask", lambda self, prompt: pytest.fail("a test reached the model")
    )


def test_the_entrypoint_is_registered():
    assert "main" in app.app.handlers


def test_a_wake_reports_what_it_checked_and_what_needs_the_parent():
    out = _invoke({"action": "wake", "today": "2026-12-01", "ask_parent": False})

    assert out["status"] == "done"
    assert out["today"] == "2026-12-01"
    assert out["quiet"] is False
    assert out["headline"].endswith("need you.")
    assert len(out["checked"]) == 3
    assert {card["title"] for card in out["new_cards"]} == {
        "Progress report on all goals provided to parents — that date has passed",
        "Time to ask the district for its service records",
    }
    assert out["period"] == [TERM_START, "2026-12-01"]


def test_a_wake_before_any_service_starts_is_quiet():
    out = _invoke({"action": "wake", "today": "2026-09-01"})

    assert out["status"] == "done"
    assert out["quiet"] is True
    assert out["new_cards"] == []


def test_a_wake_defaults_to_the_wake_action():
    out = _invoke({"today": "2026-09-01"})
    assert out["status"] == "done" and "headline" in out


def test_a_statement_carries_the_totals_and_the_rendering():
    out = _invoke({"action": "statement", "start": TERM_START, "end": TERM_END, "today": TERM_END})

    assert out["status"] == "done"
    statement = out["statement"]
    assert statement["totals"] == {
        "owed_minutes": 8520,
        "delivered_minutes": 1365,
        "shortfall_minutes": 7050,
    }
    assert statement["markdown"].startswith("# Minutes statement for Maya R.")


def test_a_statement_needs_a_start_date():
    out = _invoke({"action": "statement", "today": TERM_END})
    assert out["status"] == "error"
    assert "'start' is required" in out["error"]


@pytest.mark.parametrize("bad", ["2026-13-01", "yesterday", "12-01-2026"])
def test_a_malformed_date_comes_back_as_an_error_not_a_crash(bad):
    out = _invoke({"action": "wake", "today": bad})
    assert out["status"] == "error"
    assert "'today' must be YYYY-MM-DD" in out["error"]


def test_an_unknown_action_lists_the_real_ones():
    out = _invoke({"action": "explode"}, session_id="s" * 40)
    assert out["status"] == "error"
    assert "unknown action 'explode'" in out["error"]
    for action in ("wake", "statement", "ask", "answer", "outbox", "audit"):
        assert action in out["error"]
    assert out["session_id"] == "s" * 40


def test_an_answer_without_answers_is_refused_before_any_agent_is_built(monkeypatch):
    built = []
    monkeypatch.setattr(app, "_caseworker", lambda session_id: built.append(session_id) or None)

    out = _invoke({"action": "answer"}, session_id="t" * 40)

    assert out["status"] == "error"
    assert "'answers' must map" in out["error"]
    assert built == ["t" * 40], "the session is resolved first, but no agent call happens"


def test_a_session_id_is_minted_when_the_runtime_gives_none():
    out = _invoke({"action": "explode"})
    assert len(out["session_id"]) >= 33, "AgentCore rejects session ids shorter than 33 characters"


def test_a_second_wake_in_the_same_session_suppresses_what_the_first_raised():
    session = "w" * 40
    first = _invoke({"action": "wake", "today": "2026-12-01", "ask_parent": False}, session_id=session)
    assert first["status"] == "done" and len(first["new_cards"]) == 2 and first["suppressed"] == 0

    again = _invoke({"action": "wake", "today": "2026-12-02", "ask_parent": False}, session_id=session)
    assert again["status"] == "done"
    assert again["new_cards"] == [], "nothing changed overnight, so nothing is re-raised"
    assert again["suppressed"] == 2


def test_wakes_in_different_sessions_do_not_share_what_was_raised():
    _invoke({"action": "wake", "today": "2026-12-01", "ask_parent": False}, session_id="x" * 40)
    other = _invoke({"action": "wake", "today": "2026-12-01", "ask_parent": False}, session_id="y" * 40)
    assert len(other["new_cards"]) == 2 and other["suppressed"] == 0


def test_what_a_wake_raised_survives_a_fresh_process(monkeypatch):
    """The caseworker cache is emptied, so the next wake must rebuild from the session store."""
    session = "z" * 40
    _invoke({"action": "wake", "today": "2026-12-01", "ask_parent": False}, session_id=session)
    monkeypatch.setattr(app, "_caseworkers", {})
    again = _invoke({"action": "wake", "today": "2026-12-02", "ask_parent": False}, session_id=session)
    assert again["new_cards"] == [] and again["suppressed"] == 2


def test_without_a_bucket_the_case_is_keyed_by_the_runtime_session():
    assert app._case_key("s" * 40, {"case_id": "ignored"}) == "s" * 40


def test_with_a_bucket_the_case_is_keyed_by_the_case_not_the_microvm(monkeypatch):
    monkeypatch.setattr(app, "STATE_BUCKET", "a-bucket")
    assert app._case_key("s" * 40, {"case_id": "case-42"}) == "case-42"
    assert app._case_key("t" * 40, {"case_id": "case-42"}) == "case-42", "a different microVM, the same case"
    assert app._case_key("u" * 40, {}) == app.DEFAULT_CASE_ID


def test_with_a_bucket_the_caseworker_keeps_its_case_in_s3(monkeypatch):
    """The S3 store is selected and keyed by the case. Constructing the real
    S3SessionManager initializes the session in the bucket, so a stand-in
    records what it was asked for instead of reaching the network."""
    import minutes.agent as agent_module

    built = {}

    class RecordingS3SessionManager:
        def __init__(self, *, session_id, bucket, prefix, region_name):
            built.update(session_id=session_id, bucket=bucket, prefix=prefix, region_name=region_name)

        def __getattr__(self, name):  # any hook registration the Agent asks for
            return lambda *args, **kwargs: None

    monkeypatch.setattr(agent_module, "S3SessionManager", RecordingS3SessionManager)
    monkeypatch.setattr(app, "STATE_BUCKET", "a-bucket")
    monkeypatch.setattr(app, "STATE_PREFIX", "cases/")

    worker = app._caseworker("case-42")

    assert isinstance(worker.session, RecordingS3SessionManager)
    assert built == {
        "session_id": "case-42",
        "bucket": "a-bucket",
        "prefix": "cases/",
        "region_name": app.BEDROCK_REGION if hasattr(app, "BEDROCK_REGION") else built["region_name"],
    }
    assert built["region_name"] == "us-east-1"


# ---------------------------------------------------------------------------
# The loop closes: a wake that finds a letter asks the parent.
# ---------------------------------------------------------------------------


def _recording_ask(worker, result):
    """Replace the caseworker's model round-trip with a recorder."""
    calls = []

    def ask(prompt):
        calls.append(prompt)
        return result

    worker.ask = ask
    return calls


def test_a_quiet_wake_never_runs_the_model():
    worker = app._caseworker("q" * 40)
    calls = _recording_ask(worker, SimpleNamespace(stop_reason="end_turn", interrupts=None))

    out = _invoke({"action": "wake", "today": "2026-09-01"}, session_id="q" * 40)

    assert out["quiet"] is True and out["approval"] is None and out["status"] == "done"
    assert calls == [], "nothing to decide, so nothing to say to the caseworker"


def test_a_wake_that_finds_letters_names_each_tool_call_and_reports_the_interrupt():
    worker = app._caseworker("l" * 40)
    pending = SimpleNamespace(id="v1:tool_call:abc:def", name="send-records-request:28c3", reason={"question": "?"})
    pending.to_dict = lambda: {"id": pending.id, "name": pending.name, "reason": pending.reason}
    calls = _recording_ask(worker, SimpleNamespace(stop_reason="interrupt", interrupts=[pending]))

    out = _invoke({"action": "wake", "today": "2026-12-01"}, session_id="l" * 40)

    assert len(out["new_cards"]) == 2 and all(card["draft"] for card in out["new_cards"])
    assert len(calls) == 1, "one instruction for the whole wake"
    instruction = calls[0]
    assert "Today is 2026-12-01" in instruction
    assert 'send_letter(kind="deadline_reminder", today="2026-12-01", deadline_due="2026-11-06")' in instruction
    assert 'send_records_request(today="2026-12-01")' in instruction
    assert "do not draft anything yourself" in instruction
    assert out["status"] == "awaiting_approval"
    assert out["approval"]["status"] == "awaiting_approval"
    assert out["approval"]["interrupts"][0]["id"] == "v1:tool_call:abc:def"


def test_a_wake_can_be_told_not_to_ask():
    worker = app._caseworker("n" * 40)
    calls = _recording_ask(worker, SimpleNamespace(stop_reason="end_turn", interrupts=None))

    out = _invoke({"action": "wake", "today": "2026-12-01", "ask_parent": False}, session_id="n" * 40)

    assert len(out["new_cards"]) == 2 and calls == [] and out["approval"] is None


def test_the_instruction_never_asks_the_model_to_decide_or_restate():
    from datetime import date

    from minutes.cycle import run_cycle

    outcome = run_cycle(date(2026, 12, 1))
    text = app._approval_instruction(outcome)
    assert text is not None
    for forbidden in ("decide", "weigh", "judgement", "summarize"):
        assert forbidden not in text.lower().replace("summarise", "summarize") or "do not summarize" in text.lower().replace("summarise", "summarize")
    assert "nothing else" in text
