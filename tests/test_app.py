"""The AgentCore entrypoint, exercised without a runtime, a model, or a network.

``app.invoke`` is a plain function under the decorator, so it can be called
the way the runtime would call it. Only the model-free actions are covered
here; the caseworker actions are covered by tests/test_agent.py with a stub
model, and wiring them through HTTP adds nothing a hermetic test can see.
"""

import logging
import threading
from types import SimpleNamespace

import pytest

import app
from minutes import cases
from minutes.correspondence import load_cached_events, load_correspondence
from minutes.tools import CORRESPONDENCE_FIXTURE, load_case_record

TERM_START = "2026-09-08"
TERM_END = "2027-01-29"


def _invoke(payload: dict, session_id: str | None = None) -> dict:
    context = SimpleNamespace(session_id=session_id)
    return app.invoke(payload, context)


def _stored_copy_of_the_sample(case_id: str) -> str:
    """Write the sample case into the store under another id.

    The caseworker is keyed by case, so two tests that need two caseworkers in
    one process need two cases. A copy of the sample is one that behaves
    exactly like it.
    """
    store = cases.case_store()
    store.write_ledger(case_id, load_case_record().ledger)
    store.append_correspondence(case_id, load_correspondence(CORRESPONDENCE_FIXTURE))
    store.append_events(case_id, load_cached_events(CORRESPONDENCE_FIXTURE))
    return case_id


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    """Every test gets its own session store, case store, caseworker cache and run ledger."""
    monkeypatch.setattr(app, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(app, "STATE_BUCKET", None)
    monkeypatch.setattr(cases, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(cases, "STATE_BUCKET", None)
    monkeypatch.setattr(cases, "_s3_stores", {})
    monkeypatch.setattr(app, "_caseworkers", {})
    monkeypatch.setattr(app, "_runs", {})
    monkeypatch.setattr(app, "_run_threads", {})
    monkeypatch.setattr(app, "_inflight", {})
    # No test here may reach a model. A test that needs the caseworker's answer
    # replaces ask on its own worker instance (see _recording_ask below).
    monkeypatch.setattr(
        app.Caseworker, "ask", lambda self, prompt: pytest.fail("a test reached the model")
    )
    yield
    # No background thread outlives the test that started it: the ledger the
    # next test gets is empty because nothing is still writing to the old one.
    app._join_background(timeout=15)


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
    monkeypatch.setattr(app, "_caseworker", lambda case_id: built.append(case_id) or None)

    out = _invoke({"action": "answer"}, session_id="t" * 40)

    assert out["status"] == "error"
    assert "'answers' must map" in out["error"]
    assert built == [app.DEFAULT_CASE_ID], "the case is resolved first, but no agent call happens"


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


def test_wakes_on_different_cases_do_not_share_what_was_raised():
    first = _stored_copy_of_the_sample("family-one")
    second = _stored_copy_of_the_sample("family-two")
    _invoke({"action": "wake", "today": "2026-12-01", "ask_parent": False, "case_id": first})
    other = _invoke({"action": "wake", "today": "2026-12-01", "ask_parent": False, "case_id": second})
    assert len(other["new_cards"]) == 2 and other["suppressed"] == 0


def test_wakes_on_the_same_case_from_different_runtime_sessions_share_one_caseworker():
    """The case is the unit: the microVM that serves a wake is incidental."""
    _invoke({"action": "wake", "today": "2026-12-01", "ask_parent": False}, session_id="x" * 40)
    again = _invoke({"action": "wake", "today": "2026-12-02", "ask_parent": False}, session_id="y" * 40)
    assert again["new_cards"] == [] and again["suppressed"] == 2


def test_what_a_wake_raised_survives_a_fresh_process(monkeypatch):
    """The caseworker cache is emptied, so the next wake must rebuild from the session store."""
    session = "z" * 40
    _invoke({"action": "wake", "today": "2026-12-01", "ask_parent": False}, session_id=session)
    monkeypatch.setattr(app, "_caseworkers", {})
    again = _invoke({"action": "wake", "today": "2026-12-02", "ask_parent": False}, session_id=session)
    assert again["new_cards"] == [] and again["suppressed"] == 2


def test_the_caseworker_is_keyed_by_the_case_with_or_without_a_bucket(monkeypatch):
    assert app._case_key({"case_id": "case-4242"}) == "case-4242"
    assert app._case_key({}) == app.DEFAULT_CASE_ID
    monkeypatch.setattr(app, "STATE_BUCKET", "a-bucket")
    assert app._case_key({"case_id": "case-4242"}) == "case-4242", "a different microVM, the same case"
    assert app._case_key({}) == app.DEFAULT_CASE_ID


@pytest.mark.parametrize("bad", ["short", "has space-in-it", "-leading-dash", "x" * 65, 42])
def test_a_malformed_case_id_is_refused_before_anything_is_built(bad, monkeypatch):
    monkeypatch.setattr(app, "_caseworker", lambda case_id: pytest.fail("an agent was built"))
    out = _invoke({"action": "case", "case_id": bad})
    assert out["status"] == "error" and "case_id" in out["error"]


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

    worker = app._caseworker("case-4242")

    assert isinstance(worker.session, RecordingS3SessionManager)
    assert built == {
        "session_id": "case-4242",
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
    worker = app._caseworker(app.DEFAULT_CASE_ID)
    calls = _recording_ask(worker, SimpleNamespace(stop_reason="end_turn", interrupts=None))

    out = _invoke({"action": "wake", "today": "2026-09-01"}, session_id="q" * 40)

    assert out["quiet"] is True and out["approval"] is None and out["status"] == "done"
    assert calls == [], "nothing to decide, so nothing to say to the caseworker"


def test_a_wake_that_finds_letters_names_each_tool_call_and_reports_the_interrupt():
    worker = app._caseworker(app.DEFAULT_CASE_ID)
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
    worker = app._caseworker(app.DEFAULT_CASE_ID)
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


# ---------------------------------------------------------------------------
# A scheduled wake: acknowledged in milliseconds, done on a thread.
#
# The scheduled caller waits synchronously and gives up in about thirty
# seconds, so what these tests hold on to is the moment between the
# acknowledgement and the work — every one of them proves something while the
# wake is provably still running, rather than sleeping and hoping.
# ---------------------------------------------------------------------------


class _HeldWake:
    """A stand-in for ``_wake``, optionally held open so a test can look inside.

    ``entered`` says the background thread reached the work; ``release`` lets it
    finish, and is already set unless the test asked to hold it. The wait is
    bounded so a test that forgets to release fails on its assertions rather
    than hanging the suite.
    """

    def __init__(
        self,
        *,
        result: dict | None = None,
        error: BaseException | None = None,
        hold: bool = False,
    ):
        self.entered = threading.Event()
        self.release = threading.Event()
        if not hold:
            self.release.set()
        self.calls: list[dict] = []
        self.result = result if result is not None else {"status": "done", "quiet": True}
        self.error = error

    def __call__(self, worker, payload: dict) -> dict:
        self.calls.append(payload)
        self.entered.set()
        self.release.wait(timeout=5)
        if self.error is not None:
            raise self.error
        return self.result


def _ping() -> str:
    return app.app.get_current_ping_status().value


def test_a_background_wake_is_acknowledged_while_the_work_is_still_running(monkeypatch):
    held = _HeldWake(hold=True)
    monkeypatch.setattr(app, "_wake", held)

    out = _invoke(
        {"action": "wake", "today": "2026-12-01", "background": True}, session_id="b" * 40
    )

    assert out["status"] == "accepted"
    assert out["run_id"] and out["case"] == app.DEFAULT_CASE_ID
    assert held.entered.wait(5), "the work started"
    assert not held.release.is_set(), "and the caller already has its answer"

    # The session must not look idle for even one ping while this runs.
    assert _ping() == "HealthyBusy"
    assert app.app.get_async_task_info()["active_count"] == 1
    assert [job["name"] for job in app.app.get_async_task_info()["running_jobs"]] == ["wake"]

    held.release.set()
    app._join_background()

    assert app.app.get_async_task_info()["active_count"] == 0
    assert _ping() == "Healthy", "the task was completed, so the microVM can be reaped again"
    assert app._runs[out["run_id"]]["status"] == "done"
    assert app._runs[out["run_id"]]["result"] == held.result


def test_the_callers_run_id_is_used_so_a_retry_is_identifiable(monkeypatch):
    monkeypatch.setattr(app, "_wake", _HeldWake())

    out = _invoke(
        {"action": "wake", "background": True, "run_id": "d32c5kddcf5bb8c3"}, session_id="r" * 40
    )

    assert out["run_id"] == "d32c5kddcf5bb8c3", "Scheduler's execution id, not one of ours"
    app._join_background()


def test_a_run_id_is_minted_when_the_caller_gives_none(monkeypatch):
    monkeypatch.setattr(app, "_wake", _HeldWake())

    first = _invoke({"action": "wake", "background": True}, session_id="m" * 40)
    app._join_background()
    second = _invoke({"action": "wake", "background": True}, session_id="n" * 40)
    app._join_background()

    assert first["run_id"] and second["run_id"]
    assert first["run_id"] != second["run_id"]


def test_a_retry_of_the_same_execution_does_not_wake_the_case_twice(monkeypatch):
    held = _HeldWake()
    monkeypatch.setattr(app, "_wake", held)
    payload = {"action": "wake", "today": "2026-12-01", "background": True, "run_id": "exec-1"}

    _invoke(dict(payload), session_id="t" * 40)
    app._join_background()

    again = _invoke(dict(payload), session_id="t" * 40)

    assert again["status"] == "duplicate_run"
    assert again["run_id"] == "exec-1"
    assert again["run"]["status"] == "done"
    assert len(held.calls) == 1, "the retry of a run this microVM already did is not a second wake"


def test_a_second_wake_for_a_case_already_working_is_refused(monkeypatch):
    held = _HeldWake(hold=True)
    monkeypatch.setattr(app, "_wake", held)
    session = "c" * 40

    first = _invoke({"action": "wake", "background": True, "run_id": "exec-1"}, session_id=session)
    assert held.entered.wait(5)

    overlapping = _invoke(
        {"action": "wake", "background": True, "run_id": "exec-2"}, session_id=session
    )

    assert overlapping["status"] == "already_running"
    assert overlapping["run_id"] == first["run_id"], "the run that holds the case, not the new one"
    assert "exec-2" not in app._runs

    held.release.set()
    app._join_background()
    assert len(held.calls) == 1


def test_the_background_path_runs_exactly_the_wake_the_synchronous_path_runs():
    """No second implementation: the same ``_wake``, only nobody waits for it."""
    payload = {"action": "wake", "today": "2026-12-01", "ask_parent": False}
    _stored_copy_of_the_sample("family-direct")
    _stored_copy_of_the_sample("family-background")

    direct = _invoke({**payload, "case_id": "family-direct"})
    ack = _invoke({**payload, "case_id": "family-background", "background": True})
    app._join_background()

    assert direct["status"] == "done" and len(direct["new_cards"]) == 2
    assert ack["status"] == "accepted"
    assert app._runs[ack["run_id"]]["status"] == "done"
    assert app._runs[ack["run_id"]]["result"] == direct


def test_a_wake_without_the_flag_is_the_synchronous_wake_it_always_was():
    out = _invoke({"action": "wake", "today": "2026-12-01", "ask_parent": False}, session_id="s" * 40)

    assert out["status"] == "done" and "run_id" not in out
    assert app._runs == {} and app._run_threads == {}, "nothing was backgrounded"
    assert app.app.get_async_task_info()["active_count"] == 0, "and no task was registered"


def test_a_failing_background_wake_is_logged_and_completes_its_task(monkeypatch, caplog):
    monkeypatch.setattr(app, "_wake", _HeldWake(error=RuntimeError("the ledger would not load")))
    caplog.set_level(logging.ERROR, logger="bedrock_agentcore.app")

    out = _invoke({"action": "wake", "background": True, "run_id": "exec-9"}, session_id="f" * 40)
    app._join_background()

    assert out["status"] == "accepted", "the caller was told the wake started, and it did"
    assert app._runs["exec-9"]["status"] == "error"
    assert app._runs["exec-9"]["error"] == "RuntimeError: the ledger would not load"

    logged = [record for record in caplog.records if "exec-9" in record.getMessage()]
    assert logged and logged[0].exc_info, "the traceback is in CloudWatch, keyed by the run id"

    assert app.app.get_async_task_info()["active_count"] == 0
    assert _ping() == "Healthy", "a failure must not pin the microVM to HealthyBusy for 8 hours"


def test_a_failing_background_wake_leaves_the_case_trail_saying_so(monkeypatch):
    """A week nobody was told anything must not read like a quiet week."""
    monkeypatch.setattr(app, "_wake", _HeldWake(error=RuntimeError("the ledger would not load")))
    session = "h" * 40

    _invoke({"action": "wake", "background": True, "run_id": "exec-9"}, session_id=session)
    app._join_background()

    trail = _invoke({"action": "audit"}, session_id=session)["audit"]
    failures = [entry for entry in trail if entry["action"] == "scheduled wake failed"]
    assert len(failures) == 1
    assert failures[0]["actor"] == app.RUNTIME_ACTOR
    assert "exec-9" in failures[0]["detail"]
    assert "the ledger would not load" in failures[0]["detail"]


def test_a_failing_background_wake_does_not_take_the_runtime_down():
    # A scoped patch, not monkeypatch.undo(): undo() would also strip the
    # autouse fixture's isolation, and the follow-up wake would run against
    # the real state directory and whatever an earlier run left there.
    with pytest.MonkeyPatch.context() as held:
        held.setattr(app, "_wake", _HeldWake(error=RuntimeError("boom")))
        _invoke({"action": "wake", "background": True, "run_id": "exec-9"}, session_id="k" * 40)
        app._join_background()

    after = _invoke({"action": "wake", "today": "2026-09-01"}, session_id="k" * 40)
    assert after.get("error") is None, after
    assert after["status"] == "done" and after["quiet"] is True


def test_a_trail_that_cannot_be_written_does_not_turn_one_failure_into_two(monkeypatch):
    """The session store is a plausible cause of the failure being recorded."""
    monkeypatch.setattr(app, "_wake", _HeldWake(error=RuntimeError("boom")))
    monkeypatch.setattr(
        app, "record_action", lambda agent, entry: (_ for _ in ()).throw(OSError("no state"))
    )

    _invoke({"action": "wake", "background": True, "run_id": "exec-9"}, session_id="j" * 40)
    app._join_background()

    assert app._runs["exec-9"]["status"] == "error"
    assert app.app.get_async_task_info()["active_count"] == 0


def test_the_status_action_reports_the_run_and_the_task_behind_it(monkeypatch):
    held = _HeldWake(hold=True)
    monkeypatch.setattr(app, "_wake", held)
    session = "p" * 40

    _invoke({"action": "wake", "background": True, "run_id": "exec-1"}, session_id=session)
    assert held.entered.wait(5)

    running = _invoke({"action": "status", "run_id": "exec-1"})

    assert running["status"] == "done"
    assert running["run"]["status"] == "running" and running["run"]["case"] == app.DEFAULT_CASE_ID
    assert running["runs"] == {"exec-1": "running"}
    assert running["tasks"]["active_count"] == 1

    held.release.set()
    app._join_background()

    finished = _invoke({"action": "status", "run_id": "exec-1"})
    assert finished["run"]["status"] == "done"
    assert finished["tasks"]["active_count"] == 0


def test_the_status_action_answers_an_unknown_run_without_a_caseworker(monkeypatch):
    monkeypatch.setattr(app, "_caseworker", lambda key: pytest.fail("status built an agent"))

    out = _invoke({"action": "status", "run_id": "never-heard-of-it"})

    assert out == {"status": "done", "run": None, "runs": {}, "tasks": out["tasks"]}
    assert out["tasks"]["active_count"] == 0


def test_the_status_action_survives_a_task_table_that_moves_under_it(monkeypatch):
    def racing():
        raise RuntimeError("dictionary changed size during iteration")

    monkeypatch.setattr(app.app, "get_async_task_info", racing)

    out = _invoke({"action": "status"})

    assert out["status"] == "done"
    assert out["tasks"]["active_count"] is None and out["tasks"]["running_jobs"] == []


def test_a_run_id_from_the_caller_is_bounded(monkeypatch):
    monkeypatch.setattr(app, "_wake", _HeldWake())

    out = _invoke({"action": "wake", "background": True, "run_id": "x" * 500}, session_id="v" * 40)
    app._join_background()

    assert out["run_id"] == "x" * app.MAX_RUN_ID


def test_the_background_flag_is_explicit_and_off_by_default(monkeypatch):
    monkeypatch.setattr(app, "_wake", lambda worker, payload: {"status": "done", "ran": True})

    for payload in ({"action": "wake"}, {"action": "wake", "background": False}):
        out = _invoke(payload, session_id="e" * 40)
        assert out == {"status": "done", "ran": True}
    assert app._runs == {}
