"""Amazon Bedrock AgentCore Runtime entrypoint.

This is the deployed shape of Minutes. AgentCore gives every session its own
isolated microVM and speaks a small HTTP contract — ``POST /invocations`` with a
JSON payload, ``GET /ping`` for health — and the SDK's ``BedrockAgentCoreApp``
provides both, so this module only has to say what a payload means.

A payload names an ``action``. Two of them never touch a model and cost
nothing to run:

    {"action": "wake", "today": "2026-10-13"}          one scheduled wake-up
    {"action": "statement", "start": ..., "end": ..., "today": ...}

The rest drive the caseworker agent, and this is where the interrupt contract
crosses the wire. ``ask`` runs the agent until it either finishes or pauses on
an approval; a pause comes back as ``status: "awaiting_approval"`` with the
interrupt ids and the letter the parent is being asked about. ``answer`` sends
the parent's decisions back by id and resumes. Nothing is released without an
answer that positively reads as approval, and that rule lives in
:mod:`minutes.agent`, not here — this file cannot weaken it.

One caseworker is kept per session id. AgentCore's session *is* the microVM,
so an in-process map is the right lifetime; the file session manager beneath it
means a pending approval also survives the process, which is what lets a parent
be asked on Monday and answer on Thursday.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import date
from pathlib import Path
from typing import Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from minutes.agent import Caseworker, build_caseworker
from minutes.cycle import run_cycle
from minutes.tools import build_monthly_statement

app = BedrockAgentCoreApp()

STATE_DIR = Path(os.environ.get("MINUTES_STATE_DIR", Path(tempfile.gettempdir()) / "minutes-state"))

_caseworkers: dict[str, Caseworker] = {}


def _caseworker(session_id: str) -> Caseworker:
    worker = _caseworkers.get(session_id)
    if worker is None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        worker = build_caseworker(
            session_id=session_id,
            storage_dir=STATE_DIR,
            trail_path=STATE_DIR / f"{session_id}.trail.jsonl",
        )
        _caseworkers[session_id] = worker
    return worker


def _day(payload: dict, key: str, default: date | None = None) -> date:
    raw = payload.get(key)
    if raw is None:
        if default is None:
            raise ValueError(f"'{key}' is required, as YYYY-MM-DD")
        return default
    try:
        return date.fromisoformat(str(raw))
    except ValueError as exc:
        raise ValueError(f"'{key}' must be YYYY-MM-DD, got {raw!r}") from exc


def _result(worker: Caseworker, result: Any) -> dict:
    pending = worker.pending(result)
    if pending:
        return {
            "status": "awaiting_approval",
            "interrupts": [interrupt.to_dict() for interrupt in pending],
            "how_to_answer": (
                "POST {action: 'answer', answers: {<interrupt id>: 'approve' | 'decline'}}. "
                "Anything other than an explicit approval is a decline."
            ),
        }
    return {
        "status": "done",
        "stop_reason": result.stop_reason,
        "message": str(result).strip(),
    }


def _wake(payload: dict) -> dict:
    outcome = run_cycle(_day(payload, "today", date.today()))
    return {
        "status": "done",
        "today": outcome.today.isoformat(),
        "headline": outcome.headline,
        "quiet": outcome.quiet,
        "checked": outcome.checked,
        "period": [outcome.period_start.isoformat(), outcome.period_end.isoformat()],
        "new_cards": [card.model_dump(mode="json") for card in outcome.new_cards],
        "suppressed": len(outcome.suppressed_cards),
        "due_requests": [request.model_dump(mode="json") for request in outcome.due_requests],
    }


def _statement(payload: dict) -> dict:
    today = _day(payload, "today", date.today())
    rendered = build_monthly_statement(
        start=_day(payload, "start").isoformat(),
        end=_day(payload, "end", today).isoformat(),
        today=today.isoformat(),
    )
    return {"status": "done", "statement": rendered}


def _answer(worker: Caseworker, payload: dict) -> dict:
    answers = payload.get("answers")
    if not isinstance(answers, dict) or not answers:
        raise ValueError("'answers' must map each interrupt id to the parent's answer")
    responses = [
        {"interruptResponse": {"interruptId": interrupt_id, "response": answer}}
        for interrupt_id, answer in answers.items()
    ]
    return _result(worker, worker.ask(responses))


@app.entrypoint
def invoke(payload: dict, context) -> dict:
    """Route one invocation. Errors come back as data, never as a 500."""
    session_id = (
        getattr(context, "session_id", None)
        or (payload or {}).get("session_id")
        or str(uuid.uuid4())
    )
    action = str((payload or {}).get("action", "wake")).lower()

    try:
        if action == "wake":
            return _wake(payload)
        if action == "statement":
            return _statement(payload)

        worker = _caseworker(session_id)
        if action == "ask":
            prompt = payload.get("prompt")
            if not prompt:
                raise ValueError("'prompt' is required for ask")
            return _result(worker, worker.ask(str(prompt)))
        if action == "answer":
            return _answer(worker, payload)
        if action == "outbox":
            return {"status": "done", "outbox": worker.outbox()}
        if action == "declined":
            return {"status": "done", "declined": worker.declined()}
        if action == "audit":
            return {
                "status": "done",
                "audit": [entry.model_dump(mode="json") for entry in worker.audit()],
            }
        if action == "requests":
            return {
                "status": "done",
                "requests": [request.model_dump(mode="json") for request in worker.requests()],
            }
        raise ValueError(
            f"unknown action {action!r}; expected one of wake, statement, ask, answer, "
            "outbox, declined, audit, requests"
        )
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "session_id": session_id}
    except Exception as exc:  # the runtime boundary: report, do not crash the microVM
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "session_id": session_id}


if __name__ == "__main__":
    app.run()
