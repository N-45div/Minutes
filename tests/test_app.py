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


def test_the_entrypoint_is_registered():
    assert "main" in app.app.handlers


def test_a_wake_reports_what_it_checked_and_what_needs_the_parent():
    out = _invoke({"action": "wake", "today": "2026-12-01"})

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
