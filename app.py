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

A scheduled caller cannot wait for that. Amazon EventBridge Scheduler invokes
this runtime synchronously and gives up on the response after a short window —
around thirty seconds, observed rather than promised — so a wake that reaches a
model would be recorded as a failed invocation and retried, waking the same
case twice. A scheduled wake therefore carries a flag:

    {"action": "wake", "today": ..., "background": true, "run_id": ...}
    {"action": "status", "run_id": ...}      what this microVM is doing now

The work is registered as an AgentCore async task and run on a thread, and the
invocation returns ``{"status": "accepted", "run_id": ...}`` in milliseconds.
While a task is open ``/ping`` answers ``HealthyBusy``, which is how AgentCore
is told this session is still working and must not be reaped. Nothing else
changes: a wake without the flag is the synchronous wake it has always been,
and that is still what a person, a test and ``scripts/invoke_runtime.py`` get.

The rest drive the caseworker agent, and this is where the interrupt contract
crosses the wire. ``ask`` runs the agent until it either finishes or pauses on
an approval; a pause comes back as ``status: "awaiting_approval"`` with the
interrupt ids and the letter the parent is being asked about. ``answer`` sends
the parent's decisions back by id and resumes. Nothing is released without an
answer that positively reads as approval, and that rule lives in
:mod:`minutes.agent`, not here — this file cannot weaken it.

One caseworker is kept per case, and the case is the unit of everything.
Every payload may carry a ``case_id``; one that carries none is about the
read-only sample case. The id is set on :data:`minutes.cases.current_case_id`
for the whole invocation, which is how every tool in the engine reads the
right family's file without being told, and the caseworker is keyed by it —
the microVM that happens to serve an invocation is incidental, and two
invocations weeks apart on machines that never met open the same case. That is
what lets a parent be asked on Monday and answer on Thursday. With
``MINUTES_SESSION_BUCKET`` set the case lives in S3; without it (a laptop, a
test) it lives in a directory. The runtime session id appears only in error
responses, so a caller can find the invocation in the logs.

The remaining actions are how a parent runs THEIR case rather than the sample:

    {"action": "case", "case_id": ...}                     the ledger and the counts
    {"action": "ingest_iep", "case_id": ..., "text": ...}  one model call, creates the case
    {"action": "add_note", ...}                            a dated parent observation, no model
    {"action": "add_correspondence", ...}                  a pasted school item, classified
    {"action": "list_evidence"} / {"action": "deadlines"}  the record, newest first
    {"action": "mark_received", "request_id": ..., "received_on": ...}
    {"action": "set_notify_email", "email": ...}         where a decision notice goes
    {"action": "notify_status"}                          whether that address is ready

``add_note`` and ``mark_received`` are deterministic. ``ingest_iep`` and
``add_correspondence`` are the only two actions here that run a model, and
neither lets it decide a figure: extraction is structured output over the IEP,
and classification is the same voted, grounded reading the fixture evidence
went through.

Both of those two also take a FILE, because neither an IEP nor a service log
arrives as text anybody typed. ``ingest_iep`` takes the PDF a district emailed,
scanned or born-digital alike -- Bedrock rasterises the pages, so there is one
path and no OCR dependency. ``add_correspondence`` takes a photograph of a
page, which a tool-less transcriber reads into text; that text becomes the
item's body and goes through the identical gates a pasted email does, and the
photograph is kept, because the body is Minutes' reading of it.
"""

from __future__ import annotations

import base64
import binascii
import contextvars
import hashlib
import os
import re
import tempfile
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from minutes.agent import Caseworker, build_caseworker, record_action, record_delivery
from minutes.cases import (
    DEFAULT_CASE_ID,
    case_store,
    is_sample,
    reset_current_case,
    set_current_case,
    validate_case_id,
)
from minutes.correspondence import attributed, load_correspondence
from minutes.cycle import CycleOutcome, RaisedCard, run_cycle
from minutes.deadlines import evaluate_deadlines
from minutes.extraction import extract_ledger
from minutes.notify import SesMailer, notification_for, recipient_is_ready
from minutes.models import (
    Attachment,
    AttachmentKind,
    AuditEntry,
    Correspondence,
    CorrespondenceKind,
    DecisionCard,
    IEPLedger,
    LetterKind,
    Provenance,
    ServiceEvent,
    ServiceObligation,
)
from minutes.quarantine import scan
from minutes.transcribe import transcribe
from minutes.reader import documents_for, read_items
from minutes.reconcile import canonical_service
from minutes.tools import (
    CORRESPONDENCE_FIXTURE,
    build_monthly_statement,
    case_requests,
    load_case_record,
    store_requests,
)

app = BedrockAgentCoreApp()

STATE_DIR = Path(os.environ.get("MINUTES_STATE_DIR", Path(tempfile.gettempdir()) / "minutes-state"))

# Durable case state. Set on the AgentCore runtime (agentcore/agentcore.json);
# unset on a laptop, where the session is a directory under STATE_DIR.
STATE_BUCKET = os.environ.get("MINUTES_SESSION_BUCKET") or None
STATE_PREFIX = os.environ.get("MINUTES_SESSION_PREFIX", "cases/")

# The decision notice. Both are set on the runtime (agentcore/agentcore.json)
# and unset on a laptop, where a wake that finds a decision simply returns it.
# The sender is an SES identity the account has verified; the app URL is where
# the email points -- the This Week screen, which is the only place a letter
# can be read and released. Neither is ever a secret: the demo key is not in
# the email, and the link decides nothing.
NOTIFY_FROM = os.environ.get("MINUTES_NOTIFY_FROM") or None
APP_URL = os.environ.get("MINUTES_APP_URL") or None

_EMAIL_SHAPE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# DEFAULT_CASE_ID, the read-only sample, is minutes.cases.DEFAULT_CASE_ID and is
# imported above so the runtime and the store cannot name it differently.

# The cards already put in front of this parent, at the worst they reached.
# Kept in agent state beside the records-request machine so a wake can
# suppress what an earlier wake raised. Without it every wake re-raises the
# same cards, which trains a parent to close them unread.
STATE_RAISED = "minutes.raised"

_caseworkers: dict[str, Caseworker] = {}


def _case_key(payload: dict) -> str:
    """What a caseworker is keyed by: the case, always.

    A parent's case is the unit. Two runtime sessions on the same case must
    open the same caseworker (or the parent asked on Monday cannot answer on
    Thursday), and one runtime session serving two cases must never share
    one. The session id is not part of the key on a laptop either, so a test
    and a deployment exercise the same rule.
    """
    return validate_case_id((payload or {}).get("case_id") or DEFAULT_CASE_ID)


_VISITOR = re.compile(r"^[a-z0-9]{8,32}$")


def _worker_key(payload: dict) -> str:
    """Which caseworker serves this request. The case, except on the sample.

    A parent's case is the unit, and :func:`_case_key` says so. The sample is
    not a parent's case: it is a demo, and a demo whose interrupts, outbox and
    raised-card set are shared by everyone holding the key is a demo only the
    first visitor sees. The first person to release or decline the letter left
    every later person a quiet page saying the decisions had already been put
    to them.

    So a browser sends a ``visitor`` token with the sample only, and gets a
    session of its own -- the same fixture, replayed fresh. A malformed or
    missing token falls back to the shared session rather than failing: it is
    the sample, and a script that never sent one still works as before. Real
    cases ignore the token entirely.
    """
    case = _case_key(payload)
    visitor = str((payload or {}).get("visitor") or "").strip().lower()
    if is_sample(case) and _VISITOR.fullmatch(visitor):
        return f"{case}--{visitor}"
    return case


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

    # The parent is told before the interrupt is raised, not after: the ask
    # below calls a model and can fail or run long, and a notice that depended
    # on it would be a notice that sometimes never went. Sending is gated on
    # the cards being NEW, so the interactive wake and the scheduled one share
    # this line and a parent hears about each decision exactly once.
    notification = _notify_if_needed(worker, case, outcome, payload)

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
        "notification": notification,
    }


# ---------------------------------------------------------------------------
# Scheduled wakes: acknowledge now, work in the background.
# ---------------------------------------------------------------------------

RUNTIME_ACTOR = "minutes-runtime"
"""The ``actor`` on the trail entries this module writes.

:mod:`minutes.agent` and :mod:`minutes.cycle` name themselves on theirs. A wake
that never got as far as either of them was this file's.
"""

MAX_RUN_ID = 128
"""How much of a caller-supplied run id is kept. An id is a dictionary key here."""

# What the background runs of this process left behind, keyed by run id. This
# ledger dies with the microVM, deliberately: everything a later invocation
# needs is already in the case's session in S3, and this only answers "how did
# the run I just started go" for the caller that started it.
_runs: dict[str, dict[str, Any]] = {}
_run_threads: dict[str, threading.Thread] = {}
_inflight: dict[str, str] = {}  # case key -> the run id currently working it
_runs_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record(run_id: str, **fields: Any) -> None:
    with _runs_lock:
        _runs.setdefault(run_id, {}).update(fields)


def _run_id(payload: dict) -> str:
    """The caller's own id for this run, or a fresh one.

    EventBridge Scheduler substitutes ``<aws.scheduler.execution-id>`` into the
    payload it sends, and it keeps that id across its own retries. Taking it is
    what makes a retry recognisable as one: the id of a run this microVM is
    already doing arrives a second time, and a second arrival is not a second
    wake.
    """
    supplied = str((payload or {}).get("run_id") or "").strip()
    return supplied[:MAX_RUN_ID] or uuid.uuid4().hex


def _record_failure(worker: Caseworker, run_id: str, exc: BaseException) -> None:
    """Put a failed scheduled wake in the case's own trail, not only in the log.

    The ledger above dies with the microVM, and CloudWatch is not part of
    anybody's case. "The weekly check ran and got nowhere" is a fact about this
    family's case in exactly the way a budget trip is, and :mod:`minutes.agent`
    records those for the same reason: a week nobody was told anything must not
    read like a quiet week. A parent who later asks why November was silent
    should find the answer in the trail they already have.

    Best effort, deliberately. Whatever broke the wake may be the session store
    itself, and a trail that cannot be written must not turn one failure into
    two — so this reports and returns.
    """
    try:
        record_action(
            worker.agent,
            AuditEntry(
                entry_date=date.today(),
                actor=RUNTIME_ACTOR,
                action="scheduled wake failed",
                detail=(
                    f"Run {run_id} did not finish: {type(exc).__name__}: {exc}. Nothing from "
                    "this wake was put to the parent and no letter was released. The case is "
                    "as the last wake left it, and the next wake starts from there."
                ),
            ),
        )
        worker.sync()
    except Exception:
        app.logger.exception("could not record the failure of background wake %s", run_id)


_mailer_cache: SesMailer | None = None


def _mailer() -> SesMailer | None:
    """The one SES client, or None where notifications are not configured.

    None is the laptop and the test suite. A wake there still finds the
    decision and still returns it; it just has no one to tell. Tests replace
    this function to hand the app a fake, which is why it is a function and not
    a module constant.
    """
    global _mailer_cache
    if not NOTIFY_FROM or not APP_URL:
        return None
    if _mailer_cache is None:
        import boto3

        from minutes.config import BEDROCK_REGION

        _mailer_cache = SesMailer(boto3.client("sesv2", region_name=BEDROCK_REGION), NOTIFY_FROM)
    return _mailer_cache


def _audit(worker: Caseworker, day: date, action: str, detail: str) -> None:
    """A line in the case's own trail about the notice. Best effort, like _record_failure."""
    try:
        record_action(
            worker.agent,
            AuditEntry(entry_date=day, actor="notify", action=action, detail=detail),
        )
    except Exception:  # noqa: BLE001 -- a trail that cannot be written must not fail the wake
        app.logger.exception("could not record a notification entry in the trail")


def _notify_if_needed(worker: Caseworker, case, outcome: CycleOutcome, payload: dict) -> dict | None:
    """Tell the parent a decision is waiting -- only when one newly is.

    This is the step that makes "a background agent that surfaces only for
    real decisions" true for a parent who is not watching. Everything about it
    is gated on ``outcome.new_cards``: the cards the engine raised this wake
    and had not raised before. A quiet week sends nothing. A card the parent
    was already shown sends nothing. Nothing here is ever a reason to skip the
    interrupt below it -- the email is a notice, and the decision still waits
    in the app for a person to read the letter and answer.

    The email itself cannot act. It links to the case; it carries no approve or
    decline, because mail providers fetch every link in a message the instant
    it arrives, and an approve-by-link would release a letter to a district no
    human read. That rule lives in :mod:`minutes.notify`; this function only
    decides whether to send.

    A failure to send is recorded in the case's trail and returned, never
    raised. A week nobody was told about must not read like a quiet week --
    but it must not turn into a wake that did not happen either.
    """
    if not outcome.new_cards or payload.get("notify") is False:
        return None
    # The sample never has an address -- it is read-only, and set_notify_email
    # refuses it -- so it is left out before the store is asked anything. That
    # order matters: the sample's record carries the fixture's own legacy id,
    # which the case store does not accept, and reading its meta from here
    # broke every first wake on the live sample the moment a sender was
    # configured, while every local run passed because no sender was.
    if is_sample(case.case_id):
        return None
    mailer = _mailer()
    if mailer is None:
        return None
    email = (case_store().read_meta(case.case_id) or {}).get("notify_email")
    if not email:
        return None

    count = len(outcome.new_cards)
    noun = "decision" if count == 1 else "decisions"
    try:
        status = mailer.status(email)
        if not recipient_is_ready(status):
            _audit(
                worker,
                outcome.today,
                "decision notice held",
                f"{count} new {noun} to tell {email} about, but that address has not confirmed "
                f"itself with SES yet ({status or 'not started'}). The decision still waits in the app.",
            )
            return {"held": True, "to": email, "verification": status, "decisions": count}

        notice = notification_for(
            case_id=case.case_id,
            student_alias=case.ledger.student_alias,
            cards=list(outcome.new_cards),
            app_url=APP_URL,
            today=outcome.today,
        )
        message_id = mailer.send(email, notice)
        _audit(
            worker,
            outcome.today,
            "emailed a decision notice",
            f"Told {email} that {count} new {noun} need them (SES message {message_id or 'sent'}). "
            "The email links to the case and cannot approve or decline anything.",
        )
        return {"sent": True, "to": email, "message_id": message_id, "decisions": count}
    except Exception as exc:  # noqa: BLE001 -- recorded, returned, never raised
        app.logger.exception("decision notice to %s failed", email)
        _audit(
            worker,
            outcome.today,
            "decision notice failed",
            f"Could not email {email} about {count} new {noun}: {type(exc).__name__}: {exc}. "
            "The decision still waits in the app.",
        )
        return {"error": f"{type(exc).__name__}: {exc}", "to": email, "decisions": count}


def _set_notify_email(case_id: str, payload: dict) -> dict:
    """Where a decision notice for this case goes. Starts SES's own confirmation.

    In sandbox SES delivers only to addresses it has verified, so setting an
    address here also asks SES to email that address a confirmation link. Until
    the parent clicks it, a wake that finds a decision holds the notice and
    says so in the trail. An empty address clears it.
    """
    _writable(case_id)
    if not case_store().exists(case_id):
        raise ValueError(f"unknown case {case_id!r}; ingest_iep creates a case")

    raw = payload.get("email")
    if raw in (None, ""):
        case_store().touch(case_id, notify_email=None)
        return {"status": "done", "email": None, "verification": None, "ready": False}

    email = str(raw).strip().lower()
    if not _EMAIL_SHAPE.fullmatch(email):
        raise ValueError("'email' does not look like an address")

    mailer = _mailer()
    if mailer is None:
        raise ValueError(
            "notifications are not configured on this deployment; the runtime needs "
            "MINUTES_NOTIFY_FROM (a verified SES sender) and MINUTES_APP_URL"
        )
    mailer.verify(email)
    case_store().touch(case_id, notify_email=email)
    status = mailer.status(email)
    return {
        "status": "done",
        "email": email,
        "verification": status,
        "ready": recipient_is_ready(status),
        "note": (
            "Amazon Web Services has sent that address a confirmation email. Until the link in it "
            "is clicked, Minutes holds the notice and the decision waits in the app."
            if not recipient_is_ready(status)
            else "That address is confirmed. A wake that finds a new decision will email it."
        ),
    }


def _notify_status(case_id: str) -> dict:
    """Whether this case can be told about a decision, and by what address."""
    mailer = _mailer()
    email = (case_store().read_meta(case_id) or {}).get("notify_email") if not is_sample(case_id) else None
    status = mailer.status(email) if (mailer is not None and email) else None
    return {
        "status": "done",
        "configured": mailer is not None,
        "email": email,
        "verification": status,
        "ready": recipient_is_ready(status),
    }


def _background_wake(worker: Caseworker, payload: dict, case: str) -> dict:
    """Acknowledge a scheduled wake immediately and run it on a thread.

    The work is :func:`_wake`, unchanged and unduplicated — the only difference
    between the two paths is who waits for it.

    The task is registered before the acknowledgement is returned, not as the
    first line of the thread: ``add_async_task`` mutates the SDK's table under
    its own lock, so the very next ``/ping`` already answers ``HealthyBusy`` and
    there is no window in which a working session looks idle. It is completed in
    a ``finally``, because a task left open pins the microVM to ``HealthyBusy``
    for the rest of its life — up to the eight-hour compute ceiling — doing
    nothing.

    TWO WAKES AT ONCE. Both guards below are per microVM, which is the level
    that matters: :data:`_caseworkers` is keyed by case, so two overlapping
    wakes on one box would share one Strands agent, the second ``ask`` would
    raise a concurrency error, and the model-free halves would interleave their
    writes to one session. So a run id already seen here is a retry and is
    refused, and a case already being worked is refused. Across microVMs
    nothing here can help — two boxes share only S3, and the last writer of the
    raised-card set wins. That is left alone because it is survivable: the worst
    outcome is a card put to the parent twice, and no letter reaches a district
    either way, because every release still waits on a parent's answer to an
    interrupt.
    """
    run_id = _run_id(payload)
    with _runs_lock:
        seen = _runs.get(run_id)
        if seen is not None:
            return {"status": "duplicate_run", "run_id": run_id, "case": case, "run": dict(seen)}
        working = _inflight.get(case)
        if working is not None:
            return {"status": "already_running", "run_id": working, "case": case}
        _inflight[case] = run_id
        _runs[run_id] = {"action": "wake", "case": case, "status": "accepted", "started_at": _now()}

    task_id = app.add_async_task("wake", {"run_id": run_id, "case": case})

    def _run() -> None:
        _record(run_id, status="running")
        try:
            result = _wake(worker, payload)
            _record(run_id, status="done", result=result, finished_at=_now())
            app.logger.info("background wake %s finished (case %s)", run_id, case)
        except BaseException as exc:  # a thread that dies quietly is a wake nobody can find
            app.logger.exception("background wake %s failed (case %s)", run_id, case)
            _record(
                run_id,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                finished_at=_now(),
            )
            _record_failure(worker, run_id, exc)
        finally:
            app.complete_async_task(task_id)
            with _runs_lock:
                if _inflight.get(case) == run_id:
                    del _inflight[case]

    # A new thread starts with an empty context, and the SDK's log formatter
    # reads the request and session ids off contextvars. Without this copy every
    # line a failing wake writes is unattributable in CloudWatch.
    thread = threading.Thread(
        target=contextvars.copy_context().run, args=(_run,), name=f"wake-{run_id}", daemon=True
    )
    with _runs_lock:
        _run_threads[run_id] = thread
    thread.start()

    return {
        "status": "accepted",
        "run_id": run_id,
        "case": case,
        "how_to_check": "POST {action: 'status', run_id: <run id>} on the same runtime session",
    }


def _task_info() -> dict:
    """``app.get_async_task_info()``, minus its one sharp edge.

    The SDK iterates its task table with the ``for`` statement outside the
    ``try`` that guards the loop body, so a task completing on another thread
    mid-read raises ``RuntimeError: dictionary changed size during iteration``
    out of the call. A read that lost that race is worth retrying; a status
    query is not worth failing over.
    """
    for _ in range(3):
        try:
            return app.get_async_task_info()
        except RuntimeError:
            continue
    return {"active_count": None, "running_jobs": [], "note": "the task table would not hold still"}


def _status(payload: dict) -> dict:
    """What this microVM is doing right now.

    Per microVM, necessarily: both the ledger and the SDK's task table live in
    this process. A caller has to reuse the ``runtimeSessionId`` of the
    invocation that started the run, or it lands on another machine and is
    told, truthfully, that nothing is running there.
    """
    run_id = (payload or {}).get("run_id")
    with _runs_lock:
        run = dict(_runs[run_id]) if run_id in _runs else None
        runs = {known: entry.get("status") for known, entry in _runs.items()}
    return {"status": "done", "run": run, "runs": runs, "tasks": _task_info()}


def _join_background(timeout: float = 30.0) -> None:
    """Wait for the background runs this process started.

    Nothing in the runtime calls this. AgentCore installs no shutdown hook and a
    daemon thread is killed without unwinding at interpreter exit — which is why
    a wake persists the deterministic half of its work before it ever reaches a
    model, rather than at the end. This exists so a test can drive the real
    thread to completion instead of sleeping and hoping.
    """
    with _runs_lock:
        threads = list(_run_threads.values())
    for thread in threads:
        thread.join(timeout)


def _pending(worker: Caseworker) -> dict:
    """The approvals still waiting on the parent, read off the agent's own interrupt state.

    A wake that already raised a card will not raise it again, so a parent who
    opens the case in a fresh tab — or a fresh microVM — days later has no way
    to see a letter that is still waiting unless something reads the SDK's
    record of it. That record is the same one the send gate reads; nothing here
    consults the model or the tools, and an answered interrupt is not pending.
    """
    state = getattr(worker.agent, "_interrupt_state", None)
    waiting = [
        interrupt.to_dict()
        for interrupt in (getattr(state, "interrupts", None) or {}).values()
        if getattr(interrupt, "response", None) is None
    ]
    if waiting:
        return {
            "status": "awaiting_approval",
            "interrupts": waiting,
            "how_to_answer": (
                "POST {action: 'answer', answers: {<interrupt id>: 'approve' | 'decline'}}. "
                "Anything other than an explicit approval is a decline."
            ),
        }
    return {"status": "done", "interrupts": []}


def _statement(payload: dict) -> dict:
    today = _day(payload, "today", date.today())
    rendered = build_monthly_statement(
        start=_day(payload, "start").isoformat(),
        end=_day(payload, "end", today).isoformat(),
        today=today.isoformat(),
    )
    return {"status": "done", "statement": rendered}


# ---------------------------------------------------------------------------
# A parent's own case.
#
# Everything below reads and writes the case store directly. None of it drives
# the caseworker, and only two functions (ingest and classification) reach a
# model. Every response is data; a refusal is an error string a parent can read.
# ---------------------------------------------------------------------------

MIN_IEP_CHARS = 200
"""The least text that could be an IEP. Shorter is a title, a heading, or a
pasted URL, and extracting a ledger from it would invent every field."""


# What the runtime will accept as one file. The browser refuses earlier and
# with a friendlier sentence; the proxy refuses at 5 MB of request body, which
# is roughly this once base64 has added its third. This is the backstop, so it
# is stated in the units the parent's file is actually in.
MAX_UPLOAD_BYTES = 3_500_000

# Declared type -> (what it is, the format Bedrock wants, the bytes it must
# actually start with). The magic number is the point: a declared media type is
# a claim by whoever is uploading, and a PDF announced as a JPEG would otherwise
# be handed to an image block, or worse to a path that trusted the label.
UPLOAD_TYPES: dict[str, tuple[AttachmentKind, str, bytes]] = {
    "application/pdf": (AttachmentKind.PDF, "pdf", b"%PDF-"),
    "image/jpeg": (AttachmentKind.PHOTO, "jpeg", b"\xff\xd8\xff"),
    "image/png": (AttachmentKind.PHOTO, "png", b"\x89PNG\r\n\x1a\n"),
}

MIN_TRANSCRIPT_CHARS = 20
"""Less than this came off a photograph of nothing readable. Filing it would
put an item on the record whose body says nothing and whose original nobody
will look at again."""


def _writable(case_id: str) -> None:
    if is_sample(case_id):
        raise ValueError("the sample case is read-only")


def _upload(payload: dict) -> tuple[bytes, AttachmentKind, str, str]:
    """Decode and vet one uploaded file. Returns (bytes, kind, format, media_type).

    Every refusal here happens before a model is called, and each one names the
    field and the rule, because the parent reading it is holding the file.
    """
    file = payload.get("file")
    if not isinstance(file, dict):
        raise ValueError("'file' must be an object with 'media_type' and 'data'")

    media_type = str(file.get("media_type") or "").strip().lower()
    if media_type not in UPLOAD_TYPES:
        raise ValueError(
            f"'media_type' {media_type or '(missing)'!r} is not one Minutes reads; it reads "
            + ", ".join(sorted(UPLOAD_TYPES))
        )
    kind, fmt, magic = UPLOAD_TYPES[media_type]

    data = file.get("data")
    if not isinstance(data, str) or not data:
        raise ValueError("'data' must be the file, base64-encoded")
    try:
        blob = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as bad:
        raise ValueError(f"'data' is not valid base64: {bad}") from None

    if not blob:
        raise ValueError("'data' decoded to an empty file")
    if len(blob) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"that file is {len(blob) / 1_000_000:.1f} MB and Minutes accepts up to "
            f"{MAX_UPLOAD_BYTES / 1_000_000:.1f} MB. If you scanned it, scan again in black "
            "and white or at a lower quality - the words are what matter, not the picture."
        )
    if not blob.startswith(magic):
        raise ValueError(
            f"that file says it is {media_type} and its contents are not. Send the file as it "
            "came off the device, without renaming it."
        )
    return blob, kind, fmt, media_type


def _obligation_row(o: ServiceObligation) -> dict:
    return {
        "service": o.service,
        "minutes_per_session": o.minutes_per_session,
        "sessions_per_period": o.sessions_per_period,
        "period": o.period.value,
        "provider_role": o.provider_role,
        "setting": o.setting,
        "start_date": o.start_date.isoformat(),
        "end_date": o.end_date.isoformat(),
        "source_quote": o.source_quote,
    }


def _empty_case(case_id: str) -> dict:
    return {
        "status": "done",
        "case_id": case_id,
        "exists": False,
        "sample": False,
        "student": None,
        "school_year": None,
        "iep_date": None,
        "obligations": [],
        "deadlines": [],
        "accommodations": [],
        "counts": {"events": 0, "correspondence": 0, "requests": 0},
    }


def _case(case_id: str, worker_key: str | None = None) -> dict:
    """The ledger as the parent should see it, and how much evidence stands behind it.

    An unknown id is not an error: a front end asking "is there a case here
    yet" gets ``exists: false`` and empty lists, and knows to offer the paste
    box. The request count comes from the caseworker's own session, because the
    requests this agent has released live there and not in the case store.
    """
    if not is_sample(case_id) and not case_store().exists(case_id):
        return _empty_case(case_id)
    case = load_case_record(case_id)
    ledger = case.ledger
    return {
        "status": "done",
        "case_id": case_id,
        "exists": True,
        "sample": is_sample(case_id),
        "student": ledger.student_alias,
        "school_year": ledger.school_year,
        "iep_date": ledger.iep_date.isoformat(),
        "obligations": [_obligation_row(o) for o in ledger.obligations],
        "deadlines": [
            {"kind": d.kind.value, "due": d.due.isoformat(), "description": d.description}
            for d in ledger.deadlines
        ],
        "accommodations": [a.description for a in ledger.accommodations],
        "counts": {
            "events": len(case.events),
            "correspondence": case.correspondence_items,
            "requests": len(case_requests(_caseworker(worker_key or case_id).agent, case)),
        },
    }


def _ingest_iep(case_id: str, payload: dict) -> dict:
    """Turn the IEP a parent pasted into this case's ledger. ONE model call.

    This is the only place the model ever reads the IEP. What it returns is a
    typed ledger in which every obligation carries the sentence it came from,
    and from here on every figure about this family is arithmetic over it.
    """
    _writable(case_id)
    has_file = payload.get("file") is not None

    if has_file:
        if payload.get("text"):
            raise ValueError("send the IEP as text or as a file, not both")
        blob, kind, fmt, _ = _upload(payload)
        if kind is not AttachmentKind.PDF:
            raise ValueError(
                "an IEP goes in as the PDF the school sent you, or as pasted text. A "
                "photograph of one page is not the whole promise, and a ledger built from "
                "it would be missing most of it."
            )
        ledger = _extract_document(blob, fmt)
        source = "pdf"
    else:
        text = payload.get("text")
        if not isinstance(text, str) or len(text.strip()) < MIN_IEP_CHARS:
            raise ValueError(
                f"'text' must be the IEP itself, at least {MIN_IEP_CHARS} characters; got "
                f"{len(text.strip()) if isinstance(text, str) else 0}"
            )
        ledger = extract_ledger(text)
        source = "pasted"

    store = case_store()
    store.write_ledger(case_id, ledger)
    store.touch(case_id, student_alias=ledger.student_alias, source=source)
    out = _case(case_id)

    # The fence cannot wrap a PDF -- it is a string transform and the pages
    # reach Bedrock as images. What it can do is read the sentences that came
    # back out. A quote is the one part of a ledger taken verbatim from the
    # document, so an instruction printed on the page arrives here or nowhere.
    findings = scan(*[o.source_quote for o in ledger.obligations])
    if findings:
        out["instruction_findings"] = [finding.as_dict() for finding in findings]
    return out


def _extract_document(blob: bytes, fmt: str) -> IEPLedger:
    """Read a PDF into a ledger, turning the two known failures into sentences.

    Both of these otherwise reach the parent as a wall of library text through
    the generic handler, at the moment they are least able to act on it.
    """
    try:
        return extract_ledger(document=blob, document_format=fmt)
    except Exception as failure:
        message = str(failure)
        if "100 PDF pages" in message or "maximum of 100" in message.lower():
            raise ValueError(
                "that PDF has more than 100 pages, which is more than Minutes reads in one "
                "go. Send the services pages on their own, or paste them as text."
            ) from None
        if "validation error" in message.lower() and "IEPLedger" in message:
            raise ValueError(
                "Minutes read that PDF but could not build a ledger it can stand behind - "
                "some services came back without the dates or figures the IEP has to state. "
                "Paste the services pages as text instead."
            ) from None
        raise


def _obligation_for(ledger: IEPLedger, service: str) -> ServiceObligation:
    """The ledger line a parent's service name means, by the reconciler's own rule.

    The same folding :func:`minutes.reconcile.canonical_service` applies to
    events, so a note that matches here is a note reconciliation will count.
    A name that matches nothing is refused with the names that would have.
    """
    wanted = canonical_service(service)
    for obligation in ledger.obligations:
        if wanted and canonical_service(obligation.service) == wanted:
            return obligation
    raise ValueError(
        f"'service' {service!r} matches none of this IEP's services; use one of: "
        + ", ".join(o.service for o in ledger.obligations)
    )


def _next_id(prefix: str, day: date, taken: set[str]) -> str:
    """``<prefix>-<date>-<n>``, the first ``n`` this case has not used.

    The same shape the fixture correspondence uses (``plog-2026-09-14-01``),
    because an event's ``source`` and its item's ``item_id`` are one string and
    every citation in a letter points back through it.
    """
    for n in range(1, 1000):
        candidate = f"{prefix}-{day.isoformat()}-{n:02d}"
        if candidate not in taken:
            return candidate
    raise ValueError(f"more than 999 {prefix} items on {day.isoformat()}")


def _taken_ids(case_id: str) -> set[str]:
    store = case_store()
    return {item.item_id for item in store.read_correspondence(case_id)} | {
        event.source for event in store.read_events(case_id)
    }


def _add_note(case_id: str, payload: dict) -> dict:
    """A parent's dated observation, recorded without a model.

    A note is the one kind of evidence a parent can produce alone, and it is
    graded accordingly — ``PARENT_OBSERVED``, always attributed as the family's
    observation in anything compiled from it. It becomes two records: the
    event reconciliation counts, and the correspondence item that keeps what
    the parent actually wrote, under one id, so the note's own words are what
    :func:`~minutes.correspondence.attributed` reads to decide whether a missed
    session was the child's absence.
    """
    _writable(case_id)
    case = load_case_record(case_id)
    day = _day(payload, "date")
    delivered = payload.get("delivered")
    if not isinstance(delivered, bool):
        raise ValueError("'delivered' must be true or false")
    service = str(payload.get("service") or "").strip()
    if not service:
        raise ValueError("'service' is required; name the IEP service the note is about")
    obligation = _obligation_for(case.ledger, service)

    minutes = payload.get("minutes")
    if minutes is not None and (isinstance(minutes, bool) or not isinstance(minutes, int) or minutes < 0):
        raise ValueError(f"'minutes' must be a whole number of minutes, got {minutes!r}")
    if delivered:
        minutes = obligation.minutes_per_session if minutes is None else minutes
    else:
        minutes = 0

    text = payload.get("text")
    text = text.strip() if isinstance(text, str) and text.strip() else None
    body = text or (
        f"{obligation.service} on {day.isoformat()}: "
        + (f"delivered, about {minutes} minutes." if delivered else "not delivered.")
    )

    source = _next_id("note", day, _taken_ids(case_id))
    item = Correspondence(
        item_id=source,
        received=day,
        kind=CorrespondenceKind.PARENT_LOG,
        sender="parent",
        subject=f"quick log: {obligation.service}",
        body=body,
    )
    event = ServiceEvent(
        event_date=day,
        service=obligation.service,
        minutes=minutes,
        delivered=delivered,
        provenance=Provenance.PARENT_OBSERVED,
        source=source,
    )
    (event,) = attributed([event], [item])

    store = case_store()
    store.append_correspondence(case_id, [item])
    store.append_events(case_id, [event])
    store.touch(case_id)
    return {
        "status": "done",
        "event": event.model_dump(mode="json"),
        "item": item.model_dump(mode="json"),
    }


def _add_correspondence(case_id: str, payload: dict) -> dict:
    """A pasted school item, read for the dated service facts it states.

    The classifier is the fixture's classifier, unchanged: provenance comes
    from ``kind``, every date must be grounded in the item's own text, every
    duration must be written in it, and facts are voted across passes. So an
    empty ``events`` is a normal result — most school email contains no dated
    session fact — and it is returned as such, not as a failure.
    """
    _writable(case_id)
    case = load_case_record(case_id)
    received = _day(payload, "received")
    raw_kind = payload.get("kind")
    try:
        kind = CorrespondenceKind(str(raw_kind))
    except ValueError:
        raise ValueError(
            f"'kind' {raw_kind!r} is not one of: " + ", ".join(k.value for k in CorrespondenceKind)
        ) from None
    sender = str(payload.get("sender") or "").strip()
    if not sender:
        raise ValueError("'sender' is required; who the item came from")
    subject = str(payload.get("subject") or "").strip()
    store = case_store()

    if payload.get("file") is not None:
        if payload.get("body"):
            raise ValueError("send the item as text or as a photograph, not both")
        blob, upload_kind, _, media_type = _upload(payload)
        if upload_kind is not AttachmentKind.PHOTO:
            raise ValueError(
                "a school item goes in as text or as a photograph. Send a PDF as an IEP."
            )
        body = transcribe(blob, media_type)
        if len(body) < MIN_TRANSCRIPT_CHARS:
            raise ValueError(
                "nothing readable came off that photograph. Retake it square-on in better "
                "light, or type what the page says."
            )
        item_id = _next_id("photo", received, _taken_ids(case_id))
        # The file is stored BEFORE the item that cites it. A stored file no
        # item points at is litter; an item pointing at a file that was never
        # written is a citation to nothing, and this product is the citation.
        stored_as = store.write_attachment(case_id, f"{item_id}.{media_type.split('/')[1]}", blob, media_type)
        attachment = Attachment(
            kind=upload_kind,
            media_type=media_type,
            stored_as=stored_as,
            bytes=len(blob),
            sha256=hashlib.sha256(blob).hexdigest(),
        )
    else:
        body = str(payload.get("body") or "").strip()
        if not body:
            raise ValueError("'body' is required; the text of the item, or send 'file'")
        item_id = _next_id("paste", received, _taken_ids(case_id))
        attachment = None

    item = Correspondence(
        item_id=item_id,
        received=received,
        kind=kind,
        sender=sender,
        subject=subject,
        body=body,
        attachment=attachment,
    )
    reading = read_items([item], case.ledger)[0]

    store.append_correspondence(case_id, [item])
    store.append_events(case_id, reading.events)
    store.touch(case_id)
    return {
        "status": "done",
        "item": item.model_dump(mode="json"),
        "events": [event.model_dump(mode="json") for event in reading.events],
        "instruction_findings": [finding.as_dict() for finding in reading.findings],
    }


def _correspondence(case_id: str) -> list[Correspondence]:
    """Documents on file. Goes through the reader module, which owns them."""
    return documents_for(case_id)


def _list_evidence(case_id: str) -> dict:
    """The record, newest first: every delivery fact and every item it was read from.

    Each item carries the quarantine scan's findings, derived here on the way
    out rather than stored when it was filed. That is the same rule provenance
    and attribution follow, and it earns the same thing: the notice a parent
    sees reflects today's scan, so tightening a pattern does not require
    re-reading a semester, and nothing about the flag can drift away from the
    text it describes. It stays a note either way — the item is listed,
    reconciled and cited exactly as it would be unflagged.
    """
    case = load_case_record(case_id)
    events = sorted(case.events, key=lambda e: (e.event_date, e.source), reverse=True)
    items = sorted(_correspondence(case_id), key=lambda i: (i.received, i.item_id), reverse=True)
    return {
        "status": "done",
        "events": [event.model_dump(mode="json") for event in events],
        "correspondence": [
            {
                **item.model_dump(mode="json"),
                "instruction_findings": [
                    finding.as_dict() for finding in scan(item.subject, item.body, item.sender)
                ],
            }
            for item in items
        ],
    }


def _deadlines(case_id: str, payload: dict) -> dict:
    today = _day(payload, "today", date.today())
    statuses = evaluate_deadlines(load_case_record(case_id).ledger, today)
    return {
        "status": "done",
        "deadlines": [
            {
                "kind": status.deadline.kind.value,
                "due": status.deadline.due.isoformat(),
                "description": status.deadline.description,
                "state": status.state.value,
                "days_remaining": status.days_remaining,
            }
            for status in statuses
        ],
    }


def _mark_received(worker: Caseworker, payload: dict) -> dict:
    """The parent says the district received a request: start its clock.

    The same transition the caseworker's ``record_request_delivery`` tool
    performs, through the same function, so the rules cannot differ between a
    parent who typed the date and a parent who told the caseworker.
    """
    request_id = str(payload.get("request_id") or "").strip()
    if not request_id:
        raise ValueError("'request_id' is required, e.g. 'req-001'")
    received = _day(payload, "received_on")
    receipt = record_delivery(worker.agent, request_id, received)
    worker.sync()
    if not receipt.get("recorded"):
        raise ValueError(receipt["note"])
    request = next(r for r in worker.requests() if r.request_id == request_id)
    return {"status": "done", "request": request.model_dump(mode="json")}


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
    """Route one invocation. Errors come back as data, never as a 500.

    A wake carrying ``"background": true`` is acknowledged and handed to
    :func:`_background_wake`; every other action, and a wake without the flag,
    is answered synchronously on this thread as it always was.
    """
    session_id = (
        getattr(context, "session_id", None)
        or (payload or {}).get("session_id")
        or str(uuid.uuid4())
    )
    action = str((payload or {}).get("action", "wake")).lower()
    payload = payload or {}

    token = None
    try:
        # The case is resolved before anything else and held for the whole
        # invocation, including the thread a background wake copies its
        # context onto. The finally below is what keeps one request's case
        # from leaking into the next one this process serves.
        case = _case_key(payload)
        token = set_current_case(case)

        if action == "statement":
            return _statement(payload)
        if action == "status":
            return _status(payload)
        if action == "case":
            return _case(case, _worker_key(payload))
        if action == "ingest_iep":
            return _ingest_iep(case, payload)
        if action == "add_note":
            return _add_note(case, payload)
        if action == "add_correspondence":
            return _add_correspondence(case, payload)
        if action == "list_evidence":
            return _list_evidence(case)
        if action == "deadlines":
            return _deadlines(case, payload)

        worker = _caseworker(_worker_key(payload))
        if action == "wake":
            if payload.get("background"):
                return _background_wake(worker, payload, case)
            return _wake(worker, payload)
        if action == "ask":
            prompt = payload.get("prompt")
            if not prompt:
                raise ValueError("'prompt' is required for ask")
            return _result(worker, worker.ask(str(prompt)))
        if action == "answer":
            return _answer(worker, payload)
        if action == "pending":
            return _pending(worker)
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
        if action == "mark_received":
            return _mark_received(worker, payload)
        if action == "set_notify_email":
            return _set_notify_email(case, payload)
        if action == "notify_status":
            return _notify_status(case)
        raise ValueError(
            f"unknown action {action!r}; expected one of wake, statement, status, ask, "
            "answer, outbox, declined, audit, requests, case, ingest_iep, add_note, "
            "add_correspondence, list_evidence, deadlines, mark_received, set_notify_email, "
            "notify_status"
        )
    except ValueError as exc:
        return {"status": "error", "error": str(exc), "session_id": session_id}
    except Exception as exc:  # the runtime boundary: report, do not crash the microVM
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "session_id": session_id}
    finally:
        if token is not None:
            reset_current_case(token)


if __name__ == "__main__":
    app.run()
