"""The caseworker — Minutes as something that works while nobody is watching.

Everything else in this package is a library: you call it and it answers. This
module is the part that turns it into a worker with a job. Woken, it reads the
case and goes back to sleep without saying anything at all most weeks, because
most weeks nothing needs the parent. When something does, it stops and asks.

WHAT MINUTES DOES NOT DO, stated first because the rest of this docstring
describes a background worker and it would be easy to read more into that than
the code contains. It does not schedule itself: something external calls
:meth:`Caseworker.work` — a cron job, a task runner, a person — and this module
ships no scheduler. It does not read anybody's mailbox; the evidence comes from
committed fixtures through :func:`minutes.tools.load_case_record`, which is the
one seam a real data source plugs into. And **it cannot post anything**. When a
parent approves a letter, the letter goes into an outbox for the family to send
by a channel that proves delivery, and the statutory response clock does not
start until they come back and say it was received (see
:func:`record_request_delivery`). Every deadline this product enforces against a
district rests on that receipt date, so inventing it would mean inventing the
breach.

THREE PROPERTIES DEFINE THIS LAYER, and each one is enforced by a different
Strands primitive rather than by the model's good behaviour.

**Nothing reaches the district without a human answer.** ``send_records_request``
and ``send_letter`` are the only two tools in Minutes that touch the outside
world, and both PAUSE mid-execution on a Strands interrupt. The parent's answer
comes back into the tool and decides what happens next. A decline is not an
error path — it is recorded as carefully as an approval, because "the parent
read this and chose not to send it" is a fact about the case that a hearing
officer may one day want to see.

**The model cannot author a letter.** The send tools take identifiers — a date,
a period, a letter kind — never prose. The letter is recompiled from the ledger
and the evidence by :mod:`minutes.letters` inside the tool, so the model's only
influence on what a school district receives is *which* compiled document it
proposes and *when*. It cannot put a sentence in one. The one exception is a
blank the compiler itself reserved for a fact only the parent has (see
:func:`fill_placeholders`), and that text comes from the parent's answer, not
from the model.

**Approval binds to the exact letter that was shown.** The interrupt's name
carries a digest of the compiled body, and Strands derives the interrupt id from
the name. If the case changes between the pause and the answer, the recompiled
letter has a different digest, therefore a different id, and the stored approval
simply does not apply to it — a fresh approval is raised instead. A parent can
never approve one letter and have another one released.

**And the framework checks the tools' own discipline.** Every one of the
properties above lives inside a tool body, which is where it has to live — only
the tool has the compiled letter. :mod:`minutes.interventions` adds a second,
independent enforcement of the first property through Strands'
``interventions=`` layer: a Cedar policy file that permits tools by name and
denies anything unlisted, and a gate that reads the parent's answer off the
agent's interrupt state before a send tool is entered. Neither replaces the
checks in the bodies; each is a reason the guarantee holds if the other is
broken.

**The paper trail is the product.** An ``AfterToolCallEvent`` hook records every
tool call the agent makes as an :class:`~minutes.models.AuditEntry`, and the
send tools add the narrative entries beside them: approval requested, approval
given or refused, letter released or held. The trail lives in ``agent.state`` so
the session manager persists it, and mirrors to an append-only JSONL file so it
survives the session store too — that store is a mutable conversation snapshot,
not an audit log.

Two things that trail has to get right, and neither is automatic. **A question
nobody has answered yet is still an event.** The after-tool hook never fires for
an interrupted call — the executor skips it, because a halted tool has no
result — so a run that pauses for approval and is never answered would otherwise
leave no record at all that Minutes compiled a letter to a school district and
asked. Each send tool therefore writes that entry itself, immediately before it
pauses, and skips it on the resumed pass by recognising the digest it already
recorded. **And entries must not be lost to each other.** The default tool
executor runs a batch of tool calls concurrently, so two tools appending to
``agent.state`` are a read-modify-write race, and the losing entry vanishes with
no error anywhere. Every state append in this module goes through
:func:`record_action` and its siblings under one lock (:data:`_STATE_LOCK`).

Cost discipline is explicit: every invocation carries :data:`DEFAULT_LIMITS`, so
a model that gets stuck in a tool loop stops at a known ceiling instead of
spending the project's budget. Nothing in this module calls a model on import,
and the deterministic weekly pass in :mod:`minutes.cycle` calls one never.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from strands import Agent, ToolContext, tool
from strands.agent.agent_result import AgentResult
from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry
from strands.interrupt import Interrupt, InterruptException
from strands.models import Model
from strands.session import FileSessionManager, S3SessionManager
from strands.session.session_manager import SessionManager
from strands.types.agent import Limits

from .config import BEDROCK_REGION, EXTRACTION_MODEL
from .deadlines import evaluate_deadlines
from .decisions import REQUEST_CADENCE_DAYS, crosses_material_bar, meeting_pressure
from .discovery import due_requests, mark_sent, refresh_states
from .letters import (
    compile_compensatory_request,
    compile_deadline_reminder,
    compile_records_request,
    compile_shortfall_notice,
    placeholders,
    validate_letter,
)
from .models import (
    AuditEntry,
    DeadlineStatus,
    EvidenceRef,
    Letter,
    Provenance,
    RecordsRequest,
    RequestState,
)
from .reconcile import reconcile
from .tools import (
    STATE_REQUESTS,
    CaseRecord,
    build_monthly_statement,
    case_requests,
    check_deadlines,
    draft_records_request,
    draft_shortfall_letter,
    load_case,
    load_case_record,
    pending_decisions,
    reconcile_services,
    records_requests_status,
    store_requests,
)

__all__ = [
    "ACTOR",
    "APPROVAL_INTERRUPTS",
    "APPROVE",
    "DECLINE",
    "DEFAULT_LIMITS",
    "MAX_APPROVAL_ROUNDS",
    "MINUTES_SYSTEM_PROMPT",
    "READ_TOOLS",
    "RECORD_TOOLS",
    "SEND_TOOLS",
    "SEND_TOOL_NAMES",
    "STATE_AUDIT",
    "STATE_DECLINED",
    "STATE_OUTBOX",
    "STATE_REQUESTS",
    "AuditTrail",
    "Caseworker",
    "approval_interrupt_name",
    "audit_entries",
    "audit_trail",
    "build_caseworker",
    "case_requests",
    "declined",
    "digest_from_interrupt_name",
    "fill_placeholders",
    "is_approval",
    "letter_digest",
    "outbox",
    "read_answer",
    "record_action",
    "record_parent_decline",
    "record_request_delivery",
    "send_letter",
    "send_records_request",
]

ACTOR = "minutes-caseworker"
"""The ``actor`` on every :class:`~minutes.models.AuditEntry` this module writes.

:mod:`minutes.cycle` uses its own, so a trail that mixes the scheduled
deterministic pass with the agent's own actions still says which one acted.
"""

APPROVE = "approve"
DECLINE = "decline"

_APPROVAL_WORDS = frozenset({"approve", "approved", "yes", "y", "send", "ok", "okay"})
"""Answers read as approval. Everything else is a decline.

Fail-closed, deliberately. The cost of misreading a decline as an approval is a
letter to a school district the parent did not agree to send, which cannot be
recalled and which puts every other figure in the file in doubt. The cost of
misreading an approval as a decline is that the parent is asked again.
"""

# Keys under agent.state. The session manager persists agent.state verbatim, so
# these survive a process exit and come back on the next scheduled wake.
# STATE_REQUESTS lives in minutes.tools, because the read tools there and the
# send tools here have to be looking at one records-request state machine.
STATE_AUDIT = "audit_trail"
STATE_OUTBOX = "outbox"
STATE_DECLINED = "declined_letters"
STATE_TRAIL_PATH = "audit_trail_path"

_LIMIT_STOPS = frozenset({"limit_turns", "limit_output_tokens", "limit_total_tokens"})

MAX_APPROVAL_ROUNDS = 8
"""How many times :meth:`Caseworker.work` will resume before giving up.

Strands' ``limits`` are per invocation and their counters reset on every resume,
so :data:`DEFAULT_LIMITS` bounds each round of an approval loop and says nothing
about how many rounds there are. In practice a letter recompiles identically
from ``(case, today)``, so its digest and therefore its interrupt id are stable
and the loop converges — but if the case moves underneath it, every resume mints
a fresh approval and the loop is unbounded. Eight is far more approvals than a
real decision needs and few enough that a runaway stops in seconds; hitting it
is written to the trail exactly as a budget trip is, because a loop that was cut
short must not read as one that finished.
"""

_MAX_TRAIL_ENTRIES_RETURNED = 100
"""How much of the trail the ``audit_trail`` tool hands back to the model.

The trail grows for the life of a case and the model reads a serialized copy of
whatever a tool returns. A year of weekly runs is thousands of entries; the last
hundred answer "what have you been doing" without paying for the rest twice, in
context and in tokens. :meth:`Caseworker.audit` returns all of it to Python
callers, which is where a complete trail actually needs to go.
"""

DEFAULT_LIMITS: Limits = {"turns": 12, "output_tokens": 8_000, "total_tokens": 200_000}
"""The per-invocation budget cap. Passed on every call, including resumes.

Strands has no constructor-level turn cap; ``limits`` on the invocation is the
only agent-level bound, and hitting one raises nothing — the loop stops and
``stop_reason`` becomes ``limit_turns`` / ``limit_output_tokens`` /
``limit_total_tokens``. :meth:`Caseworker.ask` therefore checks for that and
writes it to the audit trail, because a run that stopped early looks exactly
like a run that finished.

The numbers: a full weekly pass is load_case, reconcile, deadlines, requests,
decisions and a statement — six tool calls, so twelve turns leaves room for one
recovery from a bad argument and still stops a loop dead. Output is capped low
because this agent writes connective prose and never documents; the documents
come out of the engine already written. Total tokens is the loose one, since a
statement and a letter body are large inputs to carry.

Token caps are checked at turn boundaries, so one oversized response can
overshoot by a turn. ``turns`` is the reliable bound.
"""

MINUTES_SYSTEM_PROMPT = """\
You are Minutes, working on one child's IEP case for their parent.

Schools send report cards about the child. Nobody sends a statement about the
school. You are that statement: you keep the district's promises on the record,
in the background, and you surface only when there is a real decision for the
parent to make.

HOW YOU KNOW ANYTHING
Every number, date and finding comes from a tool. The tools run a deterministic
engine over this child's IEP and this family's records. You do not compute
minutes, you do not estimate, and you do not carry figures over from earlier in
the conversation -- you call the tool and use what it returns. If no tool
returned a fact, you do not have that fact, and the honest sentence is that you
do not know it.

WHAT YOU MAY NEVER SAY
- Never characterise the district. Not "failed", not "neglected", not "in
  violation". You report what a record shows and what a date says.
- Never characterise the child. Not her attendance, not her needs, not her
  progress. An absence is a fact about a date that a record notes, and it
  exists for one reason only: so those minutes can be taken out of what is
  asked of the district. It is never a reason the services fell short and never
  a claim about her. Her accommodations describe what the district owes her,
  not anything about her.
- Never call undocumented minutes missed minutes. Undocumented means nobody
  wrote anything down, in either direction. It is a gap in the evidence, and
  the answer to a gap in the evidence is to ask for records, never to accuse.
- Never restate or paraphrase the figures inside a compiled letter or a
  statement. Those documents carry a footnote on every claim. Quote them or
  hand them over; a paraphrase drops the footnote and turns evidence into an
  assertion.
- Never give legal advice and never predict an outcome. You compile
  documentation. Say so plainly when it matters.

QUIET IS SUCCESS
Most weeks nothing needs the parent, and saying so is the job -- not a failure
to find something. When the decisions come back quiet, tell the parent what you
checked, say that nothing needs them, and stop. Never manufacture a concern to
justify having spoken.

ACTIONS THAT LEAVE THE FAMILY
send_records_request and send_letter are the only tools that reach the
district, and neither can act on its own. Each one pauses and puts the exact
letter in front of the parent; only their answer releases it. If they decline,
that is a complete and correct outcome: say so, and move on.

MINUTES CANNOT POST ANYTHING, AND THE WORD "SENT" IS NOT YOURS TO USE.
When a send tool reports a letter released, it means the parent approved it and
it is waiting in the outbox for them to send by a channel that proves delivery.
Say "ready to send" or "waiting for you to post". Never say a letter was sent,
delivered, or received, and never say a deadline is running because of one.

A records request's 45-day clock starts when the DISTRICT RECEIVES it, not when
the parent approves it and not when they post it. So the request stays a draft
until the parent tells you the date it was received and you call
record_request_delivery with that date. That date comes from them and from
nothing else -- not from you, not from an estimate, not from the date they
approved. Every later statement that the district was silent past its deadline
rests on it, so a date you invented would be an accusation you invented.

You never write a letter. Letters are compiled from the ledger and the evidence
with a footnote on every claim, which is what makes it structurally impossible
for this product to invent an accusation against a school. Your prose is the
connective tissue around them -- what you looked at, what it means for this
family, and what the choice in front of them is.
"""


# ---------------------------------------------------------------------------
# Failure handling.
# ---------------------------------------------------------------------------


def _guard(fn: Callable[..., dict]) -> Callable[..., dict]:
    """Turn any failure into an ``error`` key instead of an exception.

    ``InterruptException`` is re-raised, and that line is the most important one
    in this module. It subclasses ``Exception``, not ``BaseException``, so a
    bare ``except Exception`` anywhere between the interrupt and the event loop
    silently converts a human-approval pause into a caught error with no
    diagnostic — the tool returns "something went wrong", the parent is never
    asked, and the letter is never sent. The SDK's own middleware notes call
    this out as an inherent hazard of the design. Every send tool in this module
    sits under this decorator, so it is re-raised explicitly and tested for.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            return fn(*args, **kwargs)
        except InterruptException:
            raise
        except (ValueError, FileNotFoundError, KeyError, TypeError) as failure:
            return {"error": str(failure)}
        except Exception as failure:  # pragma: no cover -- last resort
            return {"error": f"{type(failure).__name__}: {failure}"}

    return wrapper


_ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def _parse_date(value: str, field: str) -> date:
    """An ISO date, or a message naming the parameter that was wrong.

    Shape-checked before parsing for the same reason as
    :func:`minutes.tools._parse_date`: ``date.fromisoformat`` also accepts ISO
    week dates, so ``'2026-W01-1'`` would silently become a date in 2025 and
    date a letter to a school district in the wrong year.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date string like '2026-10-15', got {value!r}")
    if not _ISO_DAY.fullmatch(value):
        raise ValueError(
            f"{field}={value!r} is not an ISO date; use YYYY-MM-DD, for example '2026-10-15'"
        )
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError(
            f"{field}={value!r} is not an ISO date; use YYYY-MM-DD, for example '2026-10-15'"
        ) from None


# ---------------------------------------------------------------------------
# The paper trail.
#
# One way in (record_action), two places out: agent.state, which the session
# manager persists and restores, and an append-only JSONL file, which survives
# the session store. Both matter. The session store is a conversation snapshot
# that gets rewritten -- redaction edits message files in place and restore
# repairs orphaned tool calls -- so it is not an audit log and a trail must
# never be reconstructed from it.
# ---------------------------------------------------------------------------


_STATE_LOCK = threading.RLock()
"""Serializes every read-modify-write of ``agent.state`` in this module.

``agent.state`` is a plain JSON dict with no concurrency control of its own, and
``get`` deep-copies out while ``set`` deep-copies in — so an append is
``read, copy, mutate, copy, write``, and the window between the read and the
write grows with the length of the trail. Strands' default ``ToolExecutor`` runs
a batch of tool calls concurrently, each sync tool body on its own worker
thread, while the after-tool hook writes from the event-loop thread. Two entries
landing in that window means one of them is silently gone: no exception, no log
line, just a trail that is missing the row somebody will one day come looking
for. Reentrant because :func:`record_action` records its own file failures.

Module-level rather than per-agent because the cost is nil — these writes take
microseconds and happen a handful of times per run — and a per-agent lock is one
more thing to get wrong for no gain.
"""


def _append_line(path: str, payload: dict) -> bool:
    """Append one JSON line, fsynced. ``False`` if the file could not be written."""
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return True
    except OSError:
        return False


def _append_state(agent: Agent, key: str, item: Any) -> None:
    """Append one item to a list under ``agent.state``, atomically.

    Every list in this module's state — the trail, the outbox, the declined
    letters — is appended to from tool bodies that Strands may be running
    concurrently. See :data:`_STATE_LOCK`.
    """
    with _STATE_LOCK:
        agent.state.set(key, [*(agent.state.get(key) or []), item])


def record_action(agent: Agent, entry: AuditEntry) -> None:
    """Write one entry to the trail: agent state first, then the file mirror.

    ``AuditEntry`` carries a ``date`` and not a timestamp — models.py is the
    frozen contract and this module does not get to widen it — so the on-disk
    line wraps the entry with a ``recorded_at`` instant. The typed entry stays
    exactly what the contract says it is; the clock order lives beside it.

    A file that cannot be written does not take the run down and does not vanish
    quietly either: the path is dropped from state (so the next entry does not
    retry into the same failure, and so this cannot recurse) and the failure
    itself is recorded as an entry in the trail that is still working.

    The whole body is under :data:`_STATE_LOCK`, so the state append and its
    file mirror stay in step and two concurrent tool calls cannot lose an entry
    to each other.
    """
    with _STATE_LOCK:
        payload = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "entry": entry.model_dump(mode="json"),
        }
        _append_state(agent, STATE_AUDIT, payload)

        path = agent.state.get(STATE_TRAIL_PATH)
        if not path or _append_line(path, payload):
            return

        agent.state.set(STATE_TRAIL_PATH, None)
        record_action(
            agent,
            AuditEntry(
                entry_date=entry.entry_date,
                actor=ACTOR,
                action="audit file unwritable",
                detail=(
                    f"Could not append to the audit file at {path}; the trail continues in "
                    "the case state only. Everything recorded before this point is intact."
                ),
            ),
        )


def audit_entries(agent: Agent) -> list[AuditEntry]:
    """The whole trail, oldest first, as typed entries."""
    return [
        AuditEntry.model_validate(item["entry"]) for item in (agent.state.get(STATE_AUDIT) or [])
    ]


def outbox(agent: Agent) -> list[dict]:
    """Every letter this case has released, in order. Empty is the normal state.

    "Released" means the parent approved it and it is waiting for them to post.
    Minutes has no delivery channel, so nothing here has been sent.
    """
    return list(agent.state.get(STATE_OUTBOX) or [])


def declined(agent: Agent) -> list[dict]:
    """Every letter the parent read and chose not to send, kept whole.

    A decline is recorded as carefully as an approval, and that has to mean the
    document too. "The parent declined a shortfall notice on 14 October" is a
    thinner record than the notice they were actually looking at, and a hearing
    officer asking what a family chose not to send deserves the second one.
    """
    return list(agent.state.get(STATE_DECLINED) or [])


class AuditTrail(HookProvider):
    """Records every tool call the agent makes, as it happens.

    ``AfterToolCallEvent`` alone is enough: it carries the arguments
    (``tool_use['input']``), the result, the status, the duration and any
    exception. ``BeforeToolCallEvent`` is deliberately NOT used, and the reason
    is specific to this agent — when a tool interrupts for approval, the tool
    body is entered once per resume round, so the before-hook fires twice for
    one logical call. The after-hook fires exactly once, because the executor
    skips it for an interrupted call ("a halted tool has no result"). Auditing
    on the after-hook is what makes one call read as one line.

    This hook writes the mechanical record: what was called, with what, and how
    it came back. The narrative entries a parent actually reads — approval
    asked, approval given, letter released or held — are written by the send
    tools themselves, because only they know what the answer was.

    Observe-only. Nothing here assigns to the event, so the SDK's immutability
    guard protects the trail from an accidental rewrite of a tool result.
    """

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(AfterToolCallEvent, self.on_tool_finished)

    def on_tool_finished(self, event: AfterToolCallEvent) -> None:
        name = event.tool_use["name"]
        status = event.result.get("status", "unknown")
        detail = f"{name}({_summarize_args(event.tool_use.get('input'))}) -> {status}"

        if event.duration is not None:
            detail += f", {event.duration:.2f}s"
        if event.cancel_message:
            detail += f"; cancelled: {event.cancel_message}"
        if event.exception is not None:
            detail += f"; raised {type(event.exception).__name__}"

        record_action(
            event.agent,
            AuditEntry(
                entry_date=_entry_date(event.tool_use.get("input")),
                actor=ACTOR,
                action=f"called {name}",
                detail=detail,
            ),
        )


def _entry_date(arguments: Any) -> date:
    """The date an action concerns: the case date it was given, else today's.

    The trail is a case record, not a server log, and the send tools date their
    own entries by the case date they were handed. A hook stamping wall-clock
    time beside them would make one trail carry two different clocks — which is
    invisible in production, where they are the same day, and incoherent in a
    replay, which is exactly where a reader is trying to follow the sequence.

    Nothing is lost by preferring the case date: the true instant is written
    beside every entry as ``recorded_at`` on the JSONL line, and list order
    always holds.
    """
    if isinstance(arguments, dict):
        for key in ("today", "end"):
            value = arguments.get(key)
            if isinstance(value, str):
                try:
                    return date.fromisoformat(value)
                except ValueError:
                    continue
    return date.today()


def _summarize_args(value: Any, limit: int = 80) -> str:
    """Tool arguments as one short line. Long values are cut, never dropped."""
    if not isinstance(value, dict):
        return ""
    parts = []
    for key, item in value.items():
        text = item if isinstance(item, str) else json.dumps(item, default=str)
        if len(text) > limit:
            text = text[:limit] + "..."
        parts.append(f"{key}={text!r}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Approval: reading a parent's answer, and binding it to a letter.
# ---------------------------------------------------------------------------


def letter_digest(letter: Letter) -> str:
    """A short, stable fingerprint of a compiled letter's body.

    It goes into the interrupt's *name*, and Strands builds the interrupt id as
    ``v1:tool_call:<toolUseId>:<uuid5(NAMESPACE_OID, name)>``. Two independent
    things in that id keep an approval where it belongs. The digest means a
    letter recompiled from changed evidence has a different id, so the approval
    stored against the old one is not found and the parent is asked again about
    the letter that would actually go out. The ``toolUseId`` means an approval
    cannot transfer between two calls either, even to an identical document.
    Approving one letter and releasing another is not something this design has
    to remember to prevent.
    """
    return hashlib.sha256(letter.body.encode("utf-8")).hexdigest()[:12]


APPROVAL_INTERRUPTS: Mapping[str, str] = {
    "send_records_request": "send-records-request",
    "send_letter": "send-letter",
}
"""The interrupt name each send tool raises, keyed by tool name.

The full name is ``<prefix>:<letter digest>`` (see :func:`approval_interrupt_name`).
It is a contract shared with :mod:`minutes.interventions`, which reads the
parent's answers back off the agent's interrupt state by these names: the gate
has to recognise *which* interrupts on a tool call are letter approvals, and
which letter each one was for, without re-running the tool.
"""


def approval_interrupt_name(tool_name: str, digest: str) -> str:
    """The name a send tool gives the interrupt that shows a letter to the parent."""
    return f"{APPROVAL_INTERRUPTS[tool_name]}:{digest}"


def digest_from_interrupt_name(tool_name: str, name: str) -> str | None:
    """The letter digest an interrupt name carries, or ``None`` if it is not one of ours.

    ``None`` for anything that is not the approval interrupt of *this* tool —
    another handler's confirm prompt on the same call, say — so that a reader
    never mistakes some other question's answer for the parent's decision on a
    letter.
    """
    prefix = f"{APPROVAL_INTERRUPTS[tool_name]}:"
    if not name.startswith(prefix):
        return None
    digest = name[len(prefix) :]
    return digest or None


def is_approval(answer: Any) -> bool:
    """True only for an answer positively recognised as approval.

    Note that ``None`` never reaches here as a real answer: Strands treats a
    ``None`` interrupt response as *unanswered* and re-raises the interrupt, so
    a caller that replies ``None`` gets asked again forever rather than being
    read as a decline. That is the SDK's behaviour, not a choice this module
    makes, and it fails in the safe direction.
    """
    return isinstance(answer, str) and answer.strip().lower() in _APPROVAL_WORDS


def read_answer(response: Any) -> tuple[str, list[str]]:
    """Normalise a parent's interrupt response into ``(answer, fill)``.

    Two shapes are accepted. A bare string — ``"approve"`` or ``"decline"`` — is
    the whole answer. A dict ``{"answer": "approve", "fill": ["..."]}`` also
    carries the parent's own words for the blanks a compiled letter reserved for
    them (see :func:`fill_placeholders`).

    Anything else is returned verbatim as a non-approving answer rather than
    rejected, so an unrecognised response from some future front end declines
    the letter instead of releasing it.
    """
    if isinstance(response, Mapping):
        answer = response.get("answer", "")
        raw = response.get("fill") or []
        fill = [str(item) for item in raw] if isinstance(raw, (list, tuple)) else [str(raw)]
        return (answer if isinstance(answer, str) else str(answer)), fill
    return (response if isinstance(response, str) else str(response)), []


def fill_placeholders(letter: Letter, answers: Sequence[str]) -> Letter:
    """Fill the blanks a compiler left for the parent, with the parent's words.

    :func:`minutes.letters.compile_records_request` deliberately refuses to
    invent a fact only the family has — why on-site inspection is impractical
    for them — and leaves a ``[[PARENT: ...]]`` blank instead, which
    ``validate_letter(for_sending=True)`` then treats as blocking. This is the
    one and only route by which text that was not compiled enters a letter, and
    it is narrow on purpose:

    * the text comes from the parent's approval answer, never from the model;
    * it replaces a ``[[PARENT: ...]]`` span and touches no other character, so
      footnote markers and the citation table are untouched;
    * fewer answers than blanks leaves the rest of the blanks standing, and the
      letter stays unsendable rather than going out half-filled;
    * the result is re-validated before anything is released, so a filled blank
      that broke the letter is caught by the same gate as any other defect.

    Returns a copy. The compiled original is never mutated.
    """
    if not answers:
        return letter

    body = letter.body
    for prompt, answer in zip(placeholders(letter), answers):
        text = str(answer).strip()
        if not text:
            continue
        body = body.replace(f"[[PARENT: {prompt}]]", text, 1)
    return letter.model_copy(update={"body": body})


def _approval_reason(letter: Letter, *, action: str, question: str, context: dict) -> dict:
    """What the parent is shown when the agent stops and asks.

    Every value is a plain string, number or list. Interrupt reasons are stored
    through ``asdict()`` into JSON when a session is in play, so a rich object
    here would raise at the end of the very invocation that paused.
    """
    blanks = placeholders(letter)
    return {
        "question": question,
        "action": action,
        "letter_kind": letter.kind.value,
        "subject": letter.subject,
        "body": letter.body,
        "citations": len(letter.citations),
        "legal_basis": list(letter.legal_basis),
        "disclaimer": letter.disclaimer,
        "blanks_only_you_can_fill": blanks,
        "blocking_before_send": validate_letter(letter, for_sending=True),
        "letter_digest": letter_digest(letter),
        "how_to_answer": (
            f"Reply {APPROVE!r} to release this letter, or {DECLINE!r} to stop it. "
            "To fill the blanks first, reply "
            '{"answer": "approve", "fill": ["your words for each blank, in order"]}. '
            "Anything Minutes does not read as approval is treated as a decline."
        ),
        "nothing_is_sent_unless_you_approve": True,
        **context,
    }


def _ask_approval(
    agent: Agent,
    letter: Letter,
    *,
    on: date,
    what: str,
    reference: str,
) -> None:
    """Record that a letter was compiled and put in front of the parent.

    WHY THIS EXISTS. The after-tool hook does not fire for an interrupted call —
    Strands' executor skips it deliberately, because a halted tool has no result
    — so the single most consequential moment in this product, *Minutes compiled
    a letter to a school district and asked to send it*, would be recorded
    nowhere until the parent answered, and nowhere at all if they never did. The
    pending interrupt does survive in the session store, but that store is a
    mutable conversation snapshot and this module does not reconstruct a trail
    from it.

    Written before the pause, and idempotent, because the tool body re-runs from
    the top on resume: a second entry for the same digest would make one
    question read as two. The digest is the right key for that — an unchanged
    letter has an unchanged digest, and a letter that changed is genuinely a
    second question.
    """
    digest = letter_digest(letter)
    with _STATE_LOCK:
        recorded = agent.state.get(STATE_AUDIT) or []
        if any(
            item["entry"].get("action") == what and digest in item["entry"].get("detail", "")
            for item in recorded
        ):
            return
        record_action(
            agent,
            AuditEntry(
                entry_date=on,
                actor=ACTOR,
                action=what,
                detail=(
                    f'Compiled {reference} ({letter.kind.value}): "{letter.subject}", '
                    f"digest {digest}, {len(letter.citations)} footnoted citations. It is "
                    "in front of the parent and nothing has been released. If no answer "
                    "follows this entry, none was given."
                ),
            ),
        )


def _release(
    agent: Agent,
    letter: Letter,
    *,
    on: date,
    reference: str,
    approved_digest: str,
    evidence: EvidenceRef | None = None,
) -> dict:
    """Record a letter as released by the parent, and return the receipt.

    Minutes does not operate a mail channel, and pretending otherwise would be
    the dishonest thing to build: delivery has to be provable, and proof comes
    from the channel the family actually uses — certified post, the district's
    portal, a delivery-receipted email. What this records is the fact that
    matters and that only this system can hold: the parent read this exact
    letter on this date and released it. The outbox is where the released
    document waits for that channel.

    TWO DIGESTS, NOT ONE. ``approved_digest`` fingerprints the body the parent
    was shown; ``letter_digest`` fingerprints the body that was released. They
    differ whenever the parent filled a blank the compiler left them, which is
    exactly when a reader most wants to see both — the chain from what was put
    in front of somebody to what came out of it should be legible in the trail
    itself, not inferred.

    Releasing the same letter twice on the same day is collapsed. Each release
    was separately consented (the interrupt ids differ, so no approval was
    reused), but one document that reads as two in the trail would have a family
    post duplicates to a district and a hearing officer read a repetition as a
    pattern.
    """
    digest = letter_digest(letter)
    item = {
        "reference": reference,
        "kind": letter.kind.value,
        "subject": letter.subject,
        "body": letter.body,
        "released_on": on.isoformat(),
        "letter_digest": digest,
        "approved_digest": approved_digest,
        "citations": len(letter.citations),
    }

    with _STATE_LOCK:
        duplicate = next(
            (
                held
                for held in outbox(agent)
                if held["letter_digest"] == digest and held["released_on"] == item["released_on"]
            ),
            None,
        )
        if duplicate is not None:
            record_action(
                agent,
                AuditEntry(
                    entry_date=on,
                    actor=ACTOR,
                    action="collapsed a duplicate release",
                    detail=(
                        f'"{letter.subject}" (digest {digest}) was already released on '
                        f"{item['released_on']} and is in the outbox once. The second "
                        "approval was for the identical document, so it was not added "
                        "again; the parent has one letter to post, not two."
                    ),
                ),
            )
            return duplicate

        _append_state(agent, STATE_OUTBOX, item)
        record_action(
            agent,
            AuditEntry(
                entry_date=on,
                actor=ACTOR,
                action="released a letter for delivery",
                detail=(
                    f"The parent approved {reference} ({letter.kind.value}): "
                    f'"{letter.subject}". They were shown digest {approved_digest} and '
                    f"digest {digest} was released; it carries {len(letter.citations)} "
                    "footnoted citations and passed the send gate. Minutes does not "
                    "deliver post; the letter is in the outbox for the family's own "
                    "delivery-proved channel."
                ),
                evidence=evidence,
            ),
        )
    return item


def record_parent_decline(
    agent: Agent,
    shown: Mapping[str, Any],
    answer: Any,
    *,
    on: date,
    refused_by: str | None = None,
) -> None:
    """Record a decline — the entry, and the document that was declined.

    A decline is not an error path. "The parent read this and chose not to send
    it" is a fact about the case, and a record of it that keeps only a one-line
    summary answers *that* they declined while losing *what*. So the compiled
    body is kept beside the entry, the same way an approved one is.

    ``shown`` is the approval reason the parent was actually looking at — the
    dict :func:`_approval_reason` built for the interrupt — and the record is
    made from it rather than from a ``Letter`` because there are two writers.
    The send tools call this on their resumed pass, holding the reason they
    built. :class:`minutes.interventions.SendGate` calls it when it refuses the
    resumed pass before the tool re-enters, holding the same dict off the
    interrupt itself. One function means one shape in the declined list and one
    wording in the trail whichever layer got there first; ``refused_by`` names
    the gate when it was the gate.
    """
    tool_name = shown.get("action", "")
    kind = str(shown.get("letter_kind", "letter"))
    subject = str(shown.get("subject", ""))
    digest = str(shown.get("letter_digest", ""))
    reference = str(shown.get("reference") or shown.get("request_id") or "")

    if tool_name == "send_records_request":
        covers = shown.get("covers") or ["?", "?"]
        what = "parent declined a records request"
        detail = (
            f"Request {reference} covering {covers[0]} to {covers[-1]} was compiled and "
            f"shown to the parent, who answered {answer!r}. Nothing was released; the "
            f"letter (digest {digest}) is kept with the declined letters. The request "
            "was not marked as made, so the cadence will offer it again."
        )
    else:
        what = f"parent declined a {kind}"
        detail = (
            f'The letter "{subject}" (digest {digest}) was compiled and shown to the '
            f"parent, who answered {answer!r}. Nothing was sent, and the letter is kept "
            "whole with the declined letters."
        )
    if refused_by:
        detail += (
            f" The {refused_by} intervention refused the call before the tool re-entered, "
            "so what is kept is exactly what the parent was shown."
        )

    with _STATE_LOCK:
        _append_state(
            agent,
            STATE_DECLINED,
            {
                "reference": reference,
                "kind": kind,
                "subject": subject,
                "body": str(shown.get("body", "")),
                "declined_on": on.isoformat(),
                "letter_digest": digest,
                "citations": int(shown.get("citations") or 0),
            },
        )
        record_action(
            agent,
            AuditEntry(entry_date=on, actor=ACTOR, action=what, detail=detail),
        )


# ---------------------------------------------------------------------------
# Case state that outlives a process.
#
# case_requests and store_requests live in minutes.tools, beside the data seam,
# so the read tools and the send tools below are looking at one records-request
# state machine rather than two. They are re-exported here because this is where
# a reader of the agent layer will look for them.
# ---------------------------------------------------------------------------


def _store_requests(agent: Agent, requests: Sequence[RecordsRequest]) -> None:
    with _STATE_LOCK:
        store_requests(agent, list(requests))


# ---------------------------------------------------------------------------
# The two tools that reach outside the family.
#
# Both pause on an interrupt. Everything above the interrupt call is a pure
# recomputation -- reading the case, running the cadence, compiling the letter --
# which is what makes re-entry safe: on resume the SDK invokes the tool again
# from the top, and only the interrupt() call itself short-circuits. Every
# statement that touches the world sits BELOW the interrupt, and so runs once.
# ---------------------------------------------------------------------------


@tool(context=True)
@_guard
def send_records_request(today: str, tool_context: ToolContext) -> dict:
    """Compile a request for the district's own service records and put it to the parent.

    This is one of only two actions in Minutes that leave the family, and it
    cannot happen without a human. Calling it compiles the request the discovery
    cadence says is due and then STOPS: the parent is shown the exact letter,
    and only their answer releases it. Nothing leaves if they decline, and
    their decline is recorded.

    Releasing it does NOT send it and does NOT start any clock. Minutes cannot
    post anything: the letter goes to the outbox for the parent to send by a
    channel that proves delivery, and the district's 45-day response deadline
    only begins when the district receives it. When the parent tells you the
    date it was received, call record_request_delivery with that date -- that is
    what starts the clock. Until then the request is still a draft and there is
    no deadline to report.

    Use it when records_requests_status or pending_decisions says a request is
    due. It refuses on its own if one is not — it will not ask a district twice
    for the same period or crowd one still inside its response window.

    A records request leaves one blank only the parent can fill: why inspecting
    the records in person is impractical for them. Do not invent that fact and
    do not suggest wording for it. If the parent approves without filling it,
    the letter is held rather than released, and the tool says so.

    Args:
        today: The date the request is dated and released to the parent, ISO
            format, e.g. '2026-10-15'.
    """
    now = _parse_date(today, "today")
    agent = tool_context.agent
    case = load_case_record()

    known = refresh_states(case_requests(agent, case), now)
    due = due_requests(case.ledger, known, now, cadence_days=REQUEST_CADENCE_DAYS)
    if not due:
        return {
            "released": False,
            "reason": (
                "No records request is due today. Either one is already open and inside "
                "its response window, or not enough uncovered service time has accrued "
                f"since the last request ({REQUEST_CADENCE_DAYS} days)."
            ),
            "open_requests": [
                request.request_id
                for request in known
                if request.state in (RequestState.DRAFT, RequestState.SENT)
            ],
        }

    request = due[0]
    letter = compile_records_request(case.ledger, request, today=now)
    shown_digest = letter_digest(letter)

    # The question is itself an event, and the after-tool hook will not record
    # it: the executor skips that hook for a halted call. Written here, before
    # the pause, so an approval nobody ever answers still leaves a trail saying
    # what was compiled and what was asked. Idempotent on the digest, because
    # the tool body re-runs from the top on resume.
    _ask_approval(
        agent,
        letter,
        on=now,
        what="asked the parent to approve a records request",
        reference=request.request_id,
    )

    # THE PAUSE. Raises on the first pass; returns the parent's answer on the
    # resumed one. The digest in the name binds their answer to this exact
    # letter -- see letter_digest().
    shown = _approval_reason(
        letter,
        action="send_records_request",
        question=(
            "Release this request for the district's own service records for "
            f"{request.covers_start.isoformat()} to {request.covers_end.isoformat()}? "
            "You will post it yourself; Minutes cannot."
        ),
        context={
            "request_id": request.request_id,
            "covers": [request.covers_start.isoformat(), request.covers_end.isoformat()],
            "services": list(request.services),
            "response_would_be_due": (
                "45 calendar days from the date the district RECEIVES it, under "
                "34 CFR 300.613(a). Approving does not start that clock; telling "
                "Minutes the date it was received does."
            ),
        },
    )
    response = tool_context.interrupt(
        approval_interrupt_name("send_records_request", shown_digest), reason=shown
    )
    answer, fill = read_answer(response)

    if not is_approval(answer):
        record_parent_decline(agent, shown, answer, on=now)
        return {
            "released": False,
            "declined": True,
            "answer": answer,
            "request_id": request.request_id,
            "note": (
                "The parent read this letter and chose not to send it. That is a complete "
                "outcome, not a failure. Do not re-ask, and do not send anything else "
                "in its place."
            ),
        }

    ready = fill_placeholders(letter, fill)
    blocking = validate_letter(ready, for_sending=True)
    if blocking:
        record_action(
            agent,
            AuditEntry(
                entry_date=now,
                actor=ACTOR,
                action="held an approved records request",
                detail=(
                    f"The parent approved request {request.request_id}, but the letter "
                    f"still fails the send gate: {'; '.join(blocking)}. Nothing was released."
                ),
            ),
        )
        return {
            "released": False,
            "held": True,
            "request_id": request.request_id,
            "blocking_before_send": blocking,
            "blanks_only_the_parent_can_fill": placeholders(ready),
            "note": (
                "The parent approved but the letter is not sendable yet. Tell them exactly "
                "which blank is outstanding and ask them to supply it in their own words. "
                "Never fill it for them."
            ),
        }

    # THE REQUEST STAYS A DRAFT. mark_sent is NOT called here, and that is the
    # whole point of this branch. Its own contract says the caller must pass the
    # date a delivery-confirmed channel proves the school received it -- and on
    # this line the letter has not even been posted yet, because Minutes has no
    # mail channel and the family does. Starting the 45-day clock from the
    # approval date would make response_due early by however long the parent
    # took to post it, and everything downstream treats that date as fact: a
    # DOCUMENTED_SILENCE event is minted on it, a card states the district
    # produced nothing "inside the time 34 CFR 300.613(a) allows", and an
    # otherwise unevidenced gap becomes escalatable because of it. That is an
    # accusation manufactured out of a postal delay. The clock starts in
    # record_request_delivery, on a date only the parent can supply.
    _store_requests(agent, [*(r for r in known if r.request_id != request.request_id), request])
    item = _release(
        agent,
        ready,
        on=now,
        reference=request.request_id,
        approved_digest=shown_digest,
        evidence=EvidenceRef(
            # PARENT_OBSERVED, and the grade matters. A request the family sent
            # is a record the family made; the district has confirmed nothing by
            # receiving it. Grading it SCHOOL_CONFIRMED would quietly promote a
            # letter we wrote into the district's own admission, which is the
            # exact overstatement this product exists to refuse. The silence
            # that follows if it goes unanswered is separately graded
            # DOCUMENTED_SILENCE, by minutes.discovery, on the response date.
            provenance=Provenance.PARENT_OBSERVED,
            source=request.request_id,
            detail=(
                f"{now.isoformat()}: records request {request.request_id} released to the "
                f"parent covering {request.covers_start.isoformat()} to "
                f"{request.covers_end.isoformat()}; not yet posted, and no response "
                "deadline runs until the district's receipt is recorded."
            ),
        ),
    )

    return {
        "released": True,
        "sent": False,
        "request_id": request.request_id,
        "covers": [request.covers_start.isoformat(), request.covers_end.isoformat()],
        "response_due": None,
        "letter_digest": item["letter_digest"],
        "note": (
            "Released to the parent and waiting in the outbox. NOTHING HAS BEEN SENT and "
            "no deadline is running. Minutes cannot post: tell them to send it by a "
            "method that proves delivery, keep the proof, and then tell you the date the "
            "district received it. Call record_request_delivery with that date — the "
            "45-day clock runs from receipt, and it is what makes the deadline "
            "enforceable."
        ),
    }


_LETTER_KINDS = ("shortfall_notice", "compensatory_request", "deadline_reminder")


@tool(context=True)
@_guard
def send_letter(
    kind: str,
    today: str,
    start: str | None = None,
    end: str | None = None,
    deadline_due: str | None = None,
    *,
    tool_context: ToolContext,
) -> dict:
    """Compile a letter to the district and put it to the parent for approval.

    Like send_records_request, this STOPS and shows the parent the exact letter;
    only their answer releases it. Nothing leaves on a decline, and releasing is
    not sending: the letter waits in the outbox for the parent to post by a
    channel that proves delivery. Minutes cannot post it.

    You are naming a document, not writing one. The letter is recompiled here
    from the IEP ledger and the evidence, with a footnote on every factual
    claim, which is why this tool takes dates and a kind and no prose. There is
    no parameter through which a sentence of yours could reach a school
    district, and that is deliberate.

    Choose the kind honestly. A shortfall notice puts the arithmetic in front of
    the district and asks it to compare against its own logs; it is the right
    first letter. A compensatory request asks the IEP team to convene about
    make-up services, and is only appropriate once the gap is documented and the
    district has already had the chance to answer a notice — this tool refuses
    to compile one at all unless some service's gap is backed by records rather
    than by their absence. A deadline reminder names one date the IEP itself set
    and asks what is scheduled.

    Args:
        kind: One of 'shortfall_notice', 'compensatory_request',
            'deadline_reminder'.
        today: The date the letter is dated and released, ISO format.
        start: First day of the service period, ISO format. Required for
            'shortfall_notice' and 'compensatory_request'.
        end: Last day of the service period, ISO format. Required for
            'shortfall_notice' and 'compensatory_request'.
        deadline_due: The due date of the deadline to write about, ISO format,
            exactly as check_deadlines reported it. Required for
            'deadline_reminder'.
    """
    if kind not in _LETTER_KINDS:
        raise ValueError(f"kind={kind!r} is not one of {', '.join(_LETTER_KINDS)}")

    now = _parse_date(today, "today")
    agent = tool_context.agent
    case = load_case_record()
    known = refresh_states(case_requests(agent, case), now)

    if kind == "deadline_reminder":
        letter, reference, context = _deadline_letter(case, now, deadline_due)
    else:
        letter, reference, context = _period_letter(case, known, now, kind, start, end)

    shown_digest = letter_digest(letter)
    _ask_approval(
        agent,
        letter,
        on=now,
        what=f"asked the parent to approve a {kind}",
        reference=reference,
    )

    shown = _approval_reason(
        letter,
        action="send_letter",
        question=(
            f"Release this {kind.replace('_', ' ')} for you to send to the district? "
            "You will post it yourself; Minutes cannot."
        ),
        context={"reference": reference, **context},
    )
    response = tool_context.interrupt(
        approval_interrupt_name("send_letter", shown_digest), reason=shown
    )
    answer, fill = read_answer(response)

    if not is_approval(answer):
        record_parent_decline(agent, shown, answer, on=now)
        return {
            "released": False,
            "declined": True,
            "answer": answer,
            "kind": kind,
            "note": (
                "The parent read this letter and chose not to send it. That is a complete "
                "outcome. Do not re-ask and do not offer a different letter in its place "
                "unless they ask for one."
            ),
        }

    ready = fill_placeholders(letter, fill)
    blocking = validate_letter(ready, for_sending=True)
    if blocking:
        record_action(
            agent,
            AuditEntry(
                entry_date=now,
                actor=ACTOR,
                action=f"held an approved {kind}",
                detail=(
                    f'The parent approved "{letter.subject}", but it still fails the send '
                    f"gate: {'; '.join(blocking)}. Nothing was released."
                ),
            ),
        )
        return {
            "released": False,
            "held": True,
            "kind": kind,
            "blocking_before_send": blocking,
            "note": "The parent approved but the letter is not sendable. Nothing was sent.",
        }

    item = _release(agent, ready, on=now, reference=reference, approved_digest=shown_digest)
    return {
        "released": True,
        "sent": False,
        "kind": kind,
        "reference": reference,
        "subject": item["subject"],
        "letter_digest": item["letter_digest"],
        "note": (
            "Released to the parent and waiting in the outbox. NOTHING HAS BEEN SENT. "
            "Minutes cannot post: tell them to send it by a method that proves delivery, "
            "and to keep the dated copy."
        ),
    }


def _period_letter(
    case: CaseRecord,
    known: Sequence[RecordsRequest],
    now: date,
    kind: str,
    start: str | None,
    end: str | None,
) -> tuple[Letter, str, dict]:
    """Compile a shortfall notice or compensatory request for a service period.

    ``known`` is the request history including anything this agent has already
    sent, because a compensatory request cites an unanswered records request as
    part of why the gap is documented rather than merely alleged.

    THE ESCALATION GATE. Nothing in :mod:`minutes.decisions` ever raises a
    compensatory card — rule 3 attaches a shortfall notice and stops there — so
    until now the only thing standing between the model and a make-up-services
    demand was a sentence in a docstring. That is not a gate. A compensatory
    request is refused here unless some service's gap crosses
    :func:`minutes.decisions.crosses_material_bar`, which weighs the *documented*
    part of the gap only: without it, a period whose difference is 99%
    undocumented minutes could front a demand, and a letter whose headline
    figure is mostly minutes nobody recorded either way is the one letter that
    puts every other figure in the file in doubt.
    """
    if not start or not end:
        raise ValueError(f"kind={kind!r} needs both start and end, the service period it reports")

    first, last = _parse_date(start, "start"), _parse_date(end, "end")
    if last < first:
        raise ValueError(f"end={end} precedes start={start}; a service period runs forwards")

    result = reconcile(case.ledger, case.events, first, last)

    if kind == "compensatory_request" and not any(
        crosses_material_bar(line) for line in result.shortfalls
    ):
        raise ValueError(
            "no service's documented gap for "
            f"{first.isoformat()} to {last.isoformat()} is large enough to escalate, so a "
            "compensatory request cannot be compiled. The difference here rests on "
            "minutes with no record either way, which is a reason to ask the district for "
            "its service logs — send_records_request — or to put the arithmetic to it "
            "with kind='shortfall_notice', not to demand make-up services."
        )

    letter = (
        compile_compensatory_request(case.ledger, result, list(known), today=now)
        if kind == "compensatory_request"
        else compile_shortfall_notice(case.ledger, result, today=now)
    )
    context = {
        "period": [first.isoformat(), last.isoformat()],
        "shortfall_minutes": result.total_shortfall_minutes,
        "undocumented_minutes": sum(line.undocumented_minutes for line in result.shortfalls),
        "undocumented_note": (
            "Undocumented minutes are minutes with no record either way. They are not "
            "minutes the district is shown to have missed."
        ),
    }
    return letter, f"{kind}:{first.isoformat()}:{last.isoformat()}", context


def _deadline_letter(
    case: CaseRecord, now: date, deadline_due: str | None
) -> tuple[Letter, str, dict]:
    """Compile a reminder for exactly one deadline, named by its due date."""
    if not deadline_due:
        raise ValueError(
            "kind='deadline_reminder' needs deadline_due, the due date of the deadline to "
            "write about, exactly as check_deadlines reported it"
        )

    wanted = _parse_date(deadline_due, "deadline_due")
    statuses = evaluate_deadlines(case.ledger, now)
    matches = [status for status in statuses if status.deadline.due == wanted]

    if not matches:
        known = ", ".join(sorted({status.deadline.due.isoformat() for status in statuses}))
        raise ValueError(f"no deadline is due on {wanted.isoformat()}; the IEP sets {known}")
    if len(matches) > 1:
        raise ValueError(
            f"{len(matches)} deadlines fall on {wanted.isoformat()}, so this date does not "
            "name one letter; they cannot be written about together"
        )

    status: DeadlineStatus = matches[0]
    letter = compile_deadline_reminder(case.ledger, status, today=now)
    context = {
        "deadline": status.deadline.due.isoformat(),
        "deadline_kind": status.deadline.kind.value,
        "state": status.state.value,
        "days_remaining": status.days_remaining,
    }
    return letter, f"deadline_reminder:{status.deadline.due.isoformat()}", context


# ---------------------------------------------------------------------------
# The one fact only the family has: that the district actually received it.
# ---------------------------------------------------------------------------


@tool(context=True)
@_guard
def record_request_delivery(
    request_id: str, received_on: str, tool_context: ToolContext
) -> dict:
    """Record the date the district received a records request, starting its clock.

    Minutes cannot post anything and cannot observe a delivery, so this date
    comes from the parent and from nowhere else — the receipt from certified
    post, the district portal's confirmation, the delivery receipt on an email.
    ONLY call this with a date the parent has actually told you. Do not use the
    date they approved the letter, do not use today, and do not estimate. Every
    later statement that the district went past its deadline is computed from
    this date, so a date you invented would be an accusation you invented.

    From here the 45-day response deadline under 34 CFR 300.613(a) runs, and if
    an IEP meeting falls sooner the deadline moves to the meeting instead,
    because records are owed before any meeting regarding an IEP.

    It refuses if the request was never released to the parent, if it has
    already been recorded as delivered, or if the date precedes the day the
    letter was released — none of which can be true of a real delivery.

    Args:
        request_id: The request the parent posted, e.g. 'req-001', exactly as
            send_records_request or records_requests_status reported it.
        received_on: The date the district received it, ISO format, as the
            parent's delivery proof states it.
    """
    received = _parse_date(received_on, "received_on")
    agent = tool_context.agent
    case = load_case_record()
    known = case_requests(agent, case)

    request = next((r for r in known if r.request_id == request_id), None)
    if request is None:
        raise ValueError(
            f"no records request {request_id!r} is on this case; known requests are "
            + (", ".join(r.request_id for r in known) or "none")
        )
    if request.state is not RequestState.DRAFT:
        return {
            "recorded": False,
            "request_id": request_id,
            "state": request.state.value,
            "response_due": _iso(request.response_due),
            "note": (
                f"{request_id} is already {request.state.value}; its response clock "
                "started when its delivery was recorded and re-recording would reset a "
                "deadline that is already running."
            ),
        }

    released = next(
        (item for item in outbox(agent) if item["reference"] == request_id),
        None,
    )
    if released is None:
        raise ValueError(
            f"{request_id} has not been released to the parent, so it cannot have been "
            "received; call send_records_request and get their approval first"
        )
    if received < date.fromisoformat(released["released_on"]):
        raise ValueError(
            f"received_on {received.isoformat()} precedes the day {request_id} was "
            f"released to the parent ({released['released_on']}); a district cannot "
            "receive a letter before it exists"
        )

    # The earliest of the stacked deadlines governs, and a meeting is usually
    # the earliest. Passed only when it is strictly after receipt: mark_sent
    # rejects a meeting on or before the send date, correctly, because a
    # deadline that expired before the request would allege a breach that
    # predates it.
    meeting = meeting_pressure(evaluate_deadlines(case.ledger, received), received)
    sent = mark_sent(
        request,
        received,
        before_meeting_on=meeting if meeting and meeting > received else None,
    )
    _store_requests(agent, [*(r for r in known if r.request_id != request_id), sent])

    due = sent.response_due
    record_action(
        agent,
        AuditEntry(
            entry_date=received,
            actor=ACTOR,
            action="recorded a records request as received",
            detail=(
                f"The parent confirmed the district received {request_id} on "
                f"{received.isoformat()}, and the response clock now runs from that "
                f"date: a response is due {_iso(due) or 'unrecorded'}"
                + (
                    f", shortened from the 45-day ceiling by the IEP meeting on "
                    f"{meeting.isoformat()} under 34 CFR 300.613(a)."
                    if meeting and due == meeting
                    else " under 34 CFR 300.613(a)."
                )
            ),
            evidence=EvidenceRef(
                provenance=Provenance.PARENT_OBSERVED,
                source=request_id,
                detail=(
                    f"{received.isoformat()}: delivery of {request_id} to the district "
                    "reported by the parent from their own delivery proof."
                ),
            ),
        ),
    )

    return {
        "recorded": True,
        "request_id": request_id,
        "received_on": received.isoformat(),
        "response_due": _iso(due),
        "shortened_by_meeting": _iso(meeting) if meeting and due == meeting else None,
        "note": (
            "The clock is running from the district's receipt. Tell the parent the "
            "response date and that their delivery proof is what makes it enforceable. "
            "If nothing arrives by then, that silence becomes a dated fact about the "
            "records — never about whether a session happened."
        ),
    }


def _iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


# ---------------------------------------------------------------------------
# The trail, as the parent asks for it.
# ---------------------------------------------------------------------------


@tool(context=True)
@_guard
def audit_trail(tool_context: ToolContext) -> dict:
    """The most recent 100 things Minutes has done on this case, in order.

    This is the answer to "what have you been doing while I wasn't watching",
    and it includes what Minutes did NOT do: every tool call, every letter
    compiled, every approval asked for, and every approval refused. Hand it over
    as a record; do not summarise it into a reassurance.

    A declined letter and a held letter are in here too, and they belong there.
    Nothing was hidden because it was not acted on.

    The trail grows for the life of a case, so only the latest hundred entries
    come back. Read count and truncated: if truncated is true there is older
    history you have not been shown, and say so rather than implying this is
    everything.
    """
    entries = audit_entries(tool_context.agent)
    shown = entries[-_MAX_TRAIL_ENTRIES_RETURNED:]
    return {
        "count": len(entries),
        "showing": len(shown),
        "truncated": len(shown) < len(entries),
        "entries": [
            {
                "date": entry.entry_date.isoformat(),
                "actor": entry.actor,
                "action": entry.action,
                "detail": entry.detail,
                "evidence": entry.evidence.source if entry.evidence else None,
            }
            for entry in shown
        ],
        "letters_released_to_the_parent": len(outbox(tool_context.agent)),
        "letters_the_parent_declined": len(declined(tool_context.agent)),
    }


READ_TOOLS = (
    load_case,
    reconcile_services,
    check_deadlines,
    records_requests_status,
    draft_records_request,
    draft_shortfall_letter,
    build_monthly_statement,
    pending_decisions,
    audit_trail,
)
"""Everything the agent may do unattended. All read-only or drafting-only."""

RECORD_TOOLS = (record_request_delivery,)
"""Records a fact only the family has. Changes case state; reaches nobody.

Separate from ``READ_TOOLS`` because it is not read-only — it starts a statutory
clock — and separate from ``SEND_TOOLS`` because nothing leaves the family when
it runs, so there is nothing for a parent to approve. What guards it is that the
one input it takes is a date the parent supplied.
"""

SEND_TOOLS = (send_records_request, send_letter)
"""Every tool that puts a document to the parent for release. Each one pauses.

"Reaches the district" would overstate it: releasing a letter puts it in the
outbox for the family to post. These are the tools whose output is intended to
leave the family, and the reason each of them stops for a human.
"""

SEND_TOOL_NAMES = frozenset(tool.tool_name for tool in SEND_TOOLS)
"""The same set by name, for the layers that see a tool call before the tool does."""


# ---------------------------------------------------------------------------
# Assembly.
# ---------------------------------------------------------------------------


@dataclass
class Caseworker:
    """One family's agent, plus the machinery that makes it accountable.

    The invocation discipline this product needs — budget limits on every call,
    a limit trip written to the trail, approvals answered in a loop rather than
    once and bounded when they will not converge — lives in :meth:`ask`,
    :meth:`answer` and :meth:`work`, so there is one place to read it and one
    place to change it.

    It is a convention, not a wall. ``agent`` is a public field and calling it
    directly runs uncapped with no trail entry; Strands offers no
    constructor-level cap to lean on (``limits`` exists only on the invocation),
    so subclassing ``Agent`` would be the only way to make the discipline
    unskippable, and it would buy less than it costs — the field has to stay
    reachable for tests and for a front end that drives the event loop itself.
    Go through ``ask``/``work``.
    """

    agent: Agent
    limits: Limits
    session: SessionManager | None = None

    def ask(self, prompt: Any) -> AgentResult:
        """One invocation, under budget, with a limit trip recorded.

        Hitting a limit raises nothing in Strands: the loop stops and
        ``stop_reason`` changes. A truncated run and a finished run are
        otherwise indistinguishable, which is exactly the kind of silent failure
        an audit trail exists to catch.
        """
        result = self.agent(prompt, limits=self.limits)
        if result.stop_reason in _LIMIT_STOPS:
            record_action(
                self.agent,
                AuditEntry(
                    entry_date=date.today(),
                    actor=ACTOR,
                    action="stopped at a budget limit",
                    detail=(
                        f"The run stopped early: {result.stop_reason}. The budget was "
                        f"{self.limits}. Anything it had not yet checked was not checked."
                    ),
                ),
            )
            # Written after the last message of the run, and the session manager
            # only syncs on message-added and after-invocation -- so without this
            # the one entry saying the run was truncated is the one entry that
            # does not survive the process.
            self.sync()
        return result

    def pending(self, result: AgentResult) -> list[Interrupt]:
        """The approvals waiting on the parent, or an empty list."""
        return list(result.interrupts or []) if result.stop_reason == "interrupt" else []

    def answer(
        self, interrupts: Sequence[Interrupt], decide: Callable[[Interrupt], Any]
    ) -> AgentResult:
        """Answer every pending approval in one payload and resume.

        All of them at once, deliberately. Answering a subset resumes only the
        tool that was answered and re-enters the others, costing a round trip
        per unanswered gate.
        """
        responses = [
            {
                "interruptResponse": {
                    "interruptId": interrupt.id,
                    # None means "unanswered" to the SDK and would re-raise the
                    # same interrupt forever. A decider that returns nothing is
                    # a parent who did not answer, which is a decline.
                    "response": _answer_or_decline(decide(interrupt)),
                }
            }
            for interrupt in interrupts
        ]
        return self.ask(responses)

    def work(self, prompt: str, decide: Callable[[Interrupt], Any]) -> AgentResult:
        """Run to completion, putting every approval to ``decide`` as it arises.

        A loop, never a single ``if``: one tool can hold several gates, and a
        resumed run can interrupt again on a different tool.

        Bounded, because the loop is not guaranteed to converge on its own.
        ``limits`` are per invocation and reset on every resume, so they cap each
        round and not the number of rounds; a letter that recompiles identically
        keeps its digest and therefore its interrupt id, but a case that moves
        underneath the loop mints a fresh approval every time round. Stopping at
        :data:`MAX_APPROVAL_ROUNDS` is recorded, for the same reason a budget
        trip is: a run that gave up must not read like one that finished.
        """
        result = self.ask(prompt)
        for _ in range(MAX_APPROVAL_ROUNDS):
            pending = self.pending(result)
            if not pending:
                return result
            result = self.answer(pending, decide)

        if self.pending(result):
            record_action(
                self.agent,
                AuditEntry(
                    entry_date=date.today(),
                    actor=ACTOR,
                    action="stopped an approval loop that would not settle",
                    detail=(
                        f"{MAX_APPROVAL_ROUNDS} rounds of approvals were answered and the "
                        "run is still asking, which means the letter being compiled keeps "
                        "changing between rounds. Nothing further was answered and the "
                        "approvals still pending were not acted on."
                    ),
                ),
            )
            self.sync()
        return result

    def audit(self) -> list[AuditEntry]:
        """The complete trail, oldest first."""
        return audit_entries(self.agent)

    def outbox(self) -> list[dict]:
        """Every letter released on this case, waiting for the family to post."""
        return outbox(self.agent)

    def declined(self) -> list[dict]:
        """Every letter the parent read and chose not to send, kept whole."""
        return declined(self.agent)

    def requests(self) -> list[RecordsRequest]:
        """The records requests this case knows about, including ones we sent."""
        return case_requests(self.agent, load_case_record())

    def sync(self) -> None:
        """Persist state written after the last message of a run.

        The session manager syncs on message-added and after-invocation only, so
        anything set after the final message of a scheduled run is otherwise
        lost on process exit.
        """
        if self.session is not None:
            self.session.sync_agent(self.agent)


def _answer_or_decline(answer: Any) -> Any:
    return DECLINE if answer is None else answer


def build_caseworker(
    *,
    model: Model | str | None = None,
    session_id: str | None = None,
    storage_dir: str | Path | None = None,
    agent_id: str = "caseworker",
    trail_path: str | Path | None = None,
    limits: Limits | None = None,
    system_prompt: str = MINUTES_SYSTEM_PROMPT,
    interventions: bool = True,
    state_bucket: str | None = None,
    state_prefix: str = "cases/",
) -> Caseworker:
    """Assemble the agent, its tools, its audit hook, its interventions and its session.

    ``model`` defaults to the id in :mod:`minutes.config` — Haiku unless
    ``MINUTES_MODEL`` overrides it for a demo run. It is never hardcoded here,
    and a test passes its own model so nothing in this package needs AWS to be
    exercised.

    Passing ``session_id`` attaches a session manager, which is what makes the
    weekly wake-up real: messages, the audit trail, the outbox and the
    records-request state machine all come back in a new process, and so does a
    pending approval — a parent can be asked on Monday and answer on Thursday,
    from a process that did not exist when the question was asked.

    With ``state_bucket`` the session lives in S3 (:class:`S3SessionManager`),
    the only store that survives where this agent actually runs: on AgentCore
    every session is its own microVM and its disk goes with it. The
    ``session_id`` is then the *case*, not the microVM — two invocations weeks
    apart, on machines that never met, open the same case. Without a bucket
    the session is a directory (:class:`FileSessionManager`), which is right
    for a laptop and for tests.

    ``interventions`` is on by default and wires
    :func:`minutes.interventions.build_interventions` — the Cedar allowlist and
    the send gate — into the agent. Turning it off leaves every guarantee in the
    tool bodies intact; it exists so a test can show the difference between the
    two layers, not as a production setting.
    """
    if model is None:
        from strands.models import BedrockModel

        model = BedrockModel(model_id=EXTRACTION_MODEL, region_name=BEDROCK_REGION)

    session: SessionManager | None = None
    if session_id and state_bucket:
        session = S3SessionManager(
            session_id=session_id,
            bucket=state_bucket,
            prefix=state_prefix,
            region_name=BEDROCK_REGION,
        )
    elif session_id:
        session = FileSessionManager(
            session_id=session_id,
            storage_dir=str(storage_dir) if storage_dir else None,
        )

    # Imported here, not at the top: interventions.py reads this module's
    # approval state and writes to its trail, so a top-level import would be
    # circular.
    from .interventions import build_interventions

    agent = Agent(
        model=model,
        tools=[*READ_TOOLS, *RECORD_TOOLS, *SEND_TOOLS],
        system_prompt=system_prompt,
        hooks=[AuditTrail()],
        interventions=(
            build_interventions(principal_id=session_id or agent_id) if interventions else None
        ),
        session_manager=session,
        agent_id=agent_id,
        name="minutes-caseworker",
        description="Keeps one child's IEP obligations on the record.",
        callback_handler=None,
    )

    if trail_path is not None:
        agent.state.set(STATE_TRAIL_PATH, str(trail_path))

    return Caseworker(agent=agent, limits=limits or DEFAULT_LIMITS, session=session)
