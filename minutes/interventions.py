"""The framework's own check on the two tools that reach outside the family.

Everything that keeps a letter from leaving without the parent's say-so lives
inside ``send_records_request`` and ``send_letter`` in :mod:`minutes.agent`,
and it has to: only the tool body has the compiled letter, so only it can show
that letter to the parent and bind their answer to its digest. This module does
not move that guarantee. It makes the framework enforce it a second time, from
outside the tool, in a form a reader can check without reading the tool —
defence in depth, so that a regression in a tool body is caught by a layer that
never runs the tool body, and a policy file a judge can read states what the
agent may do at all.

Two handlers, wired into the agent through Strands' ``interventions=``:

**SendGate** runs first. Before a send tool is entered it reads the parent's
answer for *this* call off the agent's interrupt state — the SDK's own record
of what was asked and what came back, which the session manager persists — and
refuses the call if that answer is not an approval. It reads agent state and
never the tool's arguments, so nothing the model writes into a call can supply
an approval. A refusal is written to the trail through the same
:func:`~minutes.agent.record_parent_decline` the tool would have used, and the
declined letter is kept whole, taken from the interrupt the parent actually
answered.

**CedarAuthorization** runs second, over ``minutes/policy/minutes.cedar``. Cedar
is deny-by-default and the file has no wildcard: a tool is permitted by name or
it does not run, so a tool added to the agent without a policy line is refused
(and a test fails when the two lists drift). For the send tools the policy also
requires ``context.session.parent_answer`` to be ``"asking"`` or
``"approved"``, computed by the same reader the gate uses. The vended handler
cannot express "approved for this exact letter" on its own — it sees a tool
name and its input, not the agent's interrupt state — which is why the gate
exists and why the policy's clause is fed from agent state rather than
``context.input``.

**Why the gate runs before Cedar.** Both refuse a declined call. The gate knows
what a refusal means for this product and writes the decline to the trail with
the letter beside it; Cedar's refusal is a bare "denied by policy". Deny
short-circuits the chain, so the handler that records has to go first, and the
one that only denies is the backstop behind it.

**Why the send tools do not get Strands' HumanInTheLoop as well.** ``Confirm``
raises its interrupt *before* the tool runs, when the letter has not been
compiled yet: the parent would be asked to approve ``send_letter(kind=...,
today=...)`` sight unseen, and then asked again by the tool with the actual
letter in front of them. Two prompts for one decision, and the first one is
the kind of approval this product refuses to collect — an approval of a tool
call rather than of a document. The tool's own interrupt is the human loop.

**Why a send tool may be entered with no approval on file.** Its first pass
cannot release anything: every line that touches the world sits below
``tool_context.interrupt``, which raises when there is no answer to return.
Entering the tool in that state is how the parent gets asked; refusing it would
be refusing to ask. The gate calls this state ``"asking"`` and lets it through,
and the policy says why.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from strands import Agent
from strands.hooks import BeforeToolCallEvent
from strands.interventions import Deny, InterventionHandler, OnError, Proceed
from strands.vended_interventions.cedar import CedarAuthorization

from .agent import (
    ACTOR,
    SEND_TOOL_NAMES,
    AuditEntry,
    _entry_date,
    digest_from_interrupt_name,
    is_approval,
    read_answer,
    record_action,
    record_parent_decline,
)

__all__ = [
    "POLICY_PATH",
    "CedarAllowlist",
    "ParentAnswer",
    "SendGate",
    "build_interventions",
    "parent_answer",
    "policy_actions",
]

POLICY_PATH = Path(__file__).with_name("policy") / "minutes.cedar"
"""The one policy file. Ships inside the package so the deployed agent reads the same text a reviewer does."""

Phase = Literal["asking", "approved", "declined"]

_TOOL_INTERRUPT_PREFIX = "v1:tool_call:"
"""How Strands ids an interrupt raised from inside a tool: ``v1:tool_call:<toolUseId>:<uuid5(name)>``.

A ``Confirm`` from a hook on the same call would be ``v1:before_tool_call:...``
instead, which is one of two reasons the reader below filters by prefix (the
other is the interrupt's name; see :func:`~minutes.agent.digest_from_interrupt_name`).
"""


@dataclass(frozen=True)
class ParentAnswer:
    """What the agent's interrupt state says about one send-tool call.

    ``phase`` is the only field the policy sees. ``digest`` is the letter the
    answer was for, ``answer`` the parent's normalised reply, and ``shown`` the
    reason dict they were looking at — everything a refusal needs to be
    recorded as carefully as an approval.
    """

    phase: Phase
    digest: str | None = None
    answer: Any = None
    shown: Mapping[str, Any] | None = None


def parent_answer(agent: Agent, tool_name: str, tool_use_id: str) -> ParentAnswer:
    """Read the parent's answer for one call of a send tool off the agent's interrupt state.

    The interrupt state is the SDK's own record: the send tool raised an
    interrupt named ``<tool>:<digest>`` on this ``toolUseId``, and on resume the
    caller's response was stored against it. Nothing in the tool's arguments is
    consulted, and the model cannot write to this state.

    One call can carry several approval interrupts when the case moved between
    rounds (each recompile with a different digest raises a fresh one, in
    order). The parent's decision is the answer to the letter most recently put
    in front of them, so the last one wins: an unanswered latest interrupt means
    the tool will ask again (``asking``); an approving answer means the tool may
    proceed to check the digest and release (``approved``); anything else is a
    decline (``declined``). ``is_approval`` is deliberately the tool's own
    reader, so the two layers cannot disagree about what counts as yes.
    """
    prefix = f"{_TOOL_INTERRUPT_PREFIX}{tool_use_id}:"
    letters = [
        interrupt
        for interrupt in agent._interrupt_state.interrupts.values()
        if interrupt.id.startswith(prefix)
        and digest_from_interrupt_name(tool_name, interrupt.name) is not None
    ]
    if not letters:
        return ParentAnswer("asking")

    latest = letters[-1]
    digest = digest_from_interrupt_name(tool_name, latest.name)
    if latest.response is None:
        return ParentAnswer("asking", digest=digest)

    answer, _fill = read_answer(latest.response)
    shown = latest.reason if isinstance(latest.reason, Mapping) else None
    if is_approval(answer):
        return ParentAnswer("approved", digest=digest, answer=answer, shown=shown)
    return ParentAnswer("declined", digest=digest, answer=answer, shown=shown)


class SendGate(InterventionHandler):
    """Refuse a send tool whose call the parent has answered with anything but approval.

    Runs on ``before_tool_call`` only. Read and record tools pass through
    untouched; the send tools are classified by :func:`parent_answer` and
    refused in the ``declined`` phase, before the tool body is entered. The
    refusal is on the record twice over: this handler writes the decline (with
    the declined letter kept whole) through :func:`~minutes.agent.record_parent_decline`,
    and the ``AuditTrail`` after-hook writes the cancelled call itself, because
    Strands still fires the after-hook for a call a handler cancelled.

    Fails closed: a handler error is treated as a denial (``on_error='deny'``),
    so a broken gate blocks a send rather than waving it through, and the
    failure lands in the trail as a cancelled call rather than taking the
    weekly run down.
    """

    name = "minutes:send-gate"

    def __init__(self, send_tools: Sequence[str] | frozenset[str] = SEND_TOOL_NAMES) -> None:
        self._send_tools = frozenset(send_tools)

    @property
    def on_error(self) -> OnError:
        return "deny"

    def before_tool_call(self, event: BeforeToolCallEvent, **kwargs: Any) -> Proceed | Deny:
        tool_name = event.tool_use["name"]
        if tool_name not in self._send_tools:
            return Proceed(reason="not a send tool")

        verdict = parent_answer(event.agent, tool_name, event.tool_use["toolUseId"])
        if verdict.phase == "asking":
            return Proceed(reason="no answer on file for this call; the tool can only compile and ask")
        if verdict.phase == "approved":
            return Proceed(reason=f"the parent approved letter digest {verdict.digest}")

        on = _entry_date(event.tool_use.get("input"))
        if verdict.shown is not None:
            record_parent_decline(event.agent, verdict.shown, verdict.answer, on=on, refused_by=self.name)
        else:  # pragma: no cover -- an interrupt that lost its reason; still refused, still recorded
            record_action(
                event.agent,
                AuditEntry(
                    entry_date=on,
                    actor=ACTOR,
                    action=f"refused {tool_name} without approval",
                    detail=(
                        f"The parent answered {verdict.answer!r} to letter digest {verdict.digest}, "
                        f"which is not an approval, so the {self.name} intervention refused the "
                        "call before the tool re-entered. Nothing was released."
                    ),
                ),
            )
        return Deny(
            reason=(
                f"The parent read this letter (digest {verdict.digest}) and answered "
                f"{verdict.answer!r}, which is not an approval. Nothing was released and the "
                "declined letter is kept on the record. That is a complete outcome, not a "
                "failure: do not re-ask, and do not send anything else in its place."
            )
        )


class CedarAllowlist(CedarAuthorization):
    """The vended Cedar handler, fed the parent's answer from agent state.

    ``CedarAuthorization`` builds its request from the tool name, the tool
    input and a ``context.session`` that a ``context_enricher`` may extend —
    but the enricher is handed only ``{tool_name, tool_input,
    invocation_state}``, none of which identifies the call (its ``toolUseId``)
    or reaches the interrupt state. So this subclass reads the answer in
    ``before_tool_call``, where the event carries both, and the enricher hands
    it to the policy as ``context.session.parent_answer``.

    The hand-off is safe without a lock: hooks run on the agent's event-loop
    thread, and the vended ``before_tool_call`` is synchronous from the moment
    the phase is set to the moment the enricher reads it, so no other tool
    call's evaluation can interleave.

    The principal is the parent the caseworker acts for; the resource is fixed
    by the SDK to ``Resource::"agent"``. The policies do not branch on either.
    ``on_error='deny'`` for the same reason as the gate: a broken check blocks.
    """

    def __init__(
        self,
        *,
        principal_id: str,
        policies: str | Path = POLICY_PATH,
        send_tools: Sequence[str] | frozenset[str] = SEND_TOOL_NAMES,
    ) -> None:
        self._send_tools = frozenset(send_tools)
        self._answer: ParentAnswer | None = None
        super().__init__(
            policies=str(policies),
            principal={"type": "Parent", "id": principal_id},
            context_enricher=self._enrich,
            on_error="deny",
        )

    def before_tool_call(self, event: BeforeToolCallEvent, **kwargs: Any) -> Proceed | Deny:
        tool_name = event.tool_use["name"]
        self._answer = (
            parent_answer(event.agent, tool_name, event.tool_use["toolUseId"])
            if tool_name in self._send_tools
            else None
        )
        try:
            return super().before_tool_call(event, **kwargs)
        finally:
            self._answer = None

    def _enrich(self, call: dict[str, Any]) -> dict[str, Any]:
        """``context.session`` extras. ``call['tool_input']`` is deliberately never read."""
        if self._answer is None:
            return {}
        return {"parent_answer": self._answer.phase}


def build_interventions(
    *, principal_id: str, policies: str | Path = POLICY_PATH
) -> list[InterventionHandler]:
    """The handlers ``build_caseworker`` registers, in the order they must run."""
    return [SendGate(), CedarAllowlist(principal_id=principal_id, policies=policies)]


_ACTION = re.compile(r'Action::"([^"]*)"')


def policy_actions(policy_text: str) -> set[str]:
    """Every tool name the policy text mentions, for the drift check against the agent."""
    return set(_ACTION.findall(policy_text))
