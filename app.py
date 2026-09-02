"""Amazon Bedrock AgentCore Runtime entrypoint.

This is the deployed shape of Minutes. AgentCore gives every session its own
isolated microVM and speaks a small HTTP contract — ``POST /invocations`` with a
JSON payload, ``GET /ping`` for health — and the SDK's ``BedrockAgentCoreApp``
provides both, so this module only has to say what a payload means.

A payload names an ``action``. Two of them never touch a model and cost
nothing to run:

    {"action": "wake", "today": "2026-10-13"}          one scheduled wake-up
    {"action": "statement", "start": ..., "end": ..., "today": ...}

A wake is model-free right up until it finds a decision that carries a
letter. Then, and only then, it hands the caseworker one instruction naming
the exact tool call for each letter, the send tool compiles the letter and
pauses on its interrupt, and the wake comes back carrying
``approval.status == "awaiting_approval"`` with the interrupt ids. Most weeks
there is nothing to hand over and no model runs at all. That is the whole
product in one request: background work, and a human only for a decision.

The rest drive the caseworker agent, and this is where the interrupt contract
crosses the wire. ``ask`` runs the agent until it either finishes or pauses on
an approval; a pause comes back as ``status: "awaiting_approval"`` with the
interrupt ids and the letter the parent is being asked about. ``answer`` sends
the parent's decisions back by id and resumes. Nothing is released without an
answer that positively reads as approval, and that rule lives in
:mod:`minutes.agent`, not here — this file cannot weaken it.

One caseworker is kept per case. When ``MINUTES_SESSION_BUCKET`` is set the
case lives in S3 and the caseworker is keyed by ``case_id`` — the microVM that
happens to serve an invocation is incidental, and two invocations weeks apart
on machines that never met open the same case. That is what lets a parent be
asked on Monday and answer on Thursday. Without a bucket (a laptop, a test) the
case is keyed by the runtime session id and kept in a directory.
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
from minutes.cycle import CycleOutcome, RaisedCard, run_cycle
from minutes.models import DecisionCard, LetterKind
from minutes.tools import build_monthly_statement, case_requests, load_case_record, store_requests

app = BedrockAgentCoreApp()

STATE_DIR = Path(os.environ.get("MINUTES_STATE_DIR", Path(tempfile.gettempdir()) / "minutes-state"))

# Durable case state. Set on the AgentCore runtime (agentcore/agentcore.json);
# unset on a laptop, where the session is a directory under STATE_DIR.
STATE_BUCKET = os.environ.get("MINUTES_SESSION_BUCKET") or None
STATE_PREFIX = os.environ.get("MINUTES_SESSION_PREFIX", "cases/")

# The sample case. A real deployment names the case in the payload.
DEFAULT_CASE_ID = "maya-demo"

# The cards already put in front of this parent, at the worst they reached.
# Kept in agent state beside the records-request machine so a wake can
# suppress what an earlier wake raised. Without it every wake re-raises the
# same cards, which trains a parent to close them unread.
STATE_RAISED = "minutes.raised"

_caseworkers: dict[str, Caseworker] = {}


def _case_key(session_id: str, payload: dict) -> str:
    """What a caseworker is keyed by: the case when state is durable, else the session."""
    if STATE_BUCKET:
        return str((payload or {}).get("case_id") or DEFAULT_CASE_ID)
    return session_id


def _caseworker(key: str) -> Caseworker:
    worker = _caseworkers.get(key)
    if worker is None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        worker = build_caseworker(
            session_id=key,
            storage_dir=STATE_DIR,
            trail_path=STATE_DIR / f"{key}.trail.jsonl",
            state_bucket=STATE_BUCKET,
            state_prefix=STATE_PREFIX,
        )
        _caseworkers[key] = worker
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


def _approval_instruction(outcome: CycleOutcome) -> str | None:
    """The one thing a wake says to the caseworker, or None when nothing needs saying.

    Each card that carries a draft becomes a single named tool call. The
    caseworker is not asked to decide anything, weigh anything or write
    anything: the decision rules already decided, the letter is recompiled
    inside the tool from the ledger, and the parent is the one who chooses.
    """
    today = outcome.today.isoformat()
    calls: list[str] = []
    for card in outcome.new_cards:
        if card.draft is None:
            continue
        kind = card.draft.kind
        if kind is LetterKind.RECORDS_REQUEST:
            calls.append(f'send_records_request(today="{today}")')
        elif kind is LetterKind.DEADLINE_REMINDER:
            due = card.deadline.isoformat() if card.deadline else today
            calls.append(
                f'send_letter(kind="deadline_reminder", today="{today}", deadline_due="{due}")'
            )
        else:
            calls.append(
                f'send_letter(kind="{kind.value}", today="{today}", '
                f'start="{outcome.period_start.isoformat()}", end="{outcome.period_end.isoformat()}")'
            )
    if not calls:
        return None
    listed = "\n".join(f"  {index}. {call}" for index, call in enumerate(calls, 1))
    return (
        f"Today is {today}. The weekly check found {len(calls)} decision(s) that carry a "
        "letter for the parent. Put each one to the parent by making exactly these tool "
        f"calls, in order, and nothing else:\n{listed}\n"
        "Do not summarise the case, do not restate any figure, and do not draft anything "
        "yourself. When a call pauses for the parent's answer, stop."
    )


def _wake(worker: Caseworker, payload: dict) -> dict:
    """One scheduled wake, carrying the case forward from the last one.

    State threads through in exactly the two places the weekly cycle threads
    it: the records-request state machine (shared with the send tools, so a
    request the parent released is visible to the next wake) and the ids
    already raised. Both live in the caseworker's session, so they come back
    in a later invocation of the same session.
    """
    agent = worker.agent
    case = load_case_record()
    raised = {
        key: RaisedCard.model_validate(value)
        for key, value in (agent.state.get(STATE_RAISED) or {}).items()
    }
    outcome = run_cycle(
        _day(payload, "today", date.today()),
        case=case,
        requests=case_requests(agent, case),
        raised=raised,
    )
    store_requests(agent, outcome.requests)
    agent.state.set(
        STATE_RAISED, {key: card.model_dump(mode="json") for key, card in outcome.raised.items()}
    )
    worker.sync()

    # The loop closes here. A wake that found a letter does not leave it in a
    # JSON list nobody reads: it puts the release decision to the parent
    # through the same interrupt the interactive path uses.
    approval: dict | None = None
    instruction = _approval_instruction(outcome)
    if instruction and payload.get("ask_parent", True):
        approval = _result(worker, worker.ask(instruction))

    return {
        "status": approval["status"] if approval else "done",
        "approval": approval,
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
        if action == "statement":
            return _statement(payload)

        worker = _caseworker(_case_key(session_id, payload))
        if action == "wake":
            return _wake(worker, payload)
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
