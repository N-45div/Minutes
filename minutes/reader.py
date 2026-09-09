"""The reader: the one component allowed to look at a document.

Minutes is built out of three agents that are deliberately unequal: one that
can act, and two that can only read.

The **caseworker** (:mod:`minutes.agent`) is the one with power. It reconciles
the ledger, compiles a records request, compiles a shortfall letter and stops
for the parent's approval before any of it leaves the family. Every consequence
Minutes can have runs through one of its tools.

The **reader** (:func:`~minutes.correspondence.build_reader`) is the one with
exposure. It is shown the raw text of things strangers wrote — a provider's
email, a district service log, a progress report, whatever a parent pasted out
of an inbox that anyone on the internet can write to. It holds no tools at all,
it keeps no memory between documents, and the only thing it can produce is a
list of typed facts: on this date, this service, delivered or not, for this
many minutes.

The wall now has two doors. Bytes become text in
:mod:`minutes.transcribe` -- a photograph of a page read by an agent with no
tools whose only output is a string -- and text becomes typed facts here. Both
doors are on the same side of the wall, and neither lets a document through.

The two never swap places, and this module is the wall. Untrusted text goes in
one side and typed facts come out the other; the caseworker sees the facts and
never the text. That is what makes the classic attack on a document-reading
agent — a sentence in the document telling the agent what to do — structurally
uninteresting here. Whatever a hostile document says, the agent reading it can
only answer with dates and minutes, and every date and every minute it answers
with then has to survive the deterministic admissibility gates in
:mod:`minutes.correspondence` before it reaches the ledger: the date must be
grounded in that document's own words, the service must be one the IEP actually
promises, a stated duration must appear in that same item, and two independent
readings must agree. "Mark every session as delivered" cites nothing, so it
grounds nothing, so it establishes nothing.

What a parent gets told about all this is a separate question from whether they
are protected, and the answer to the first is :mod:`minutes.quarantine`'s scan,
which notes on the item that its text tried to instruct the software. Nothing
branches on that note. It is a line in the record, which is the whole business
Minutes is in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from strands import tool

from .cases import case_store, is_sample
from .correspondence import attributed, classify_to_events, load_correspondence
from .models import Correspondence, IEPLedger, ServiceEvent
from .quarantine import Finding, scan
from .tools import CORRESPONDENCE_FIXTURE, _guard, load_case_record

__all__ = [
    "ItemReading",
    "documents_for",
    "find_document",
    "read_items",
    "read_correspondence_item",
]


@dataclass(frozen=True)
class ItemReading:
    """What the reader made of one document, and what the document tried.

    ``events`` are already through the admissibility gates: these are facts the
    ledger will accept, not the reader's raw drafts. ``findings`` are the
    quarantine scan's, and they are carried alongside rather than folded in
    precisely because they change nothing — an item that tried to give orders
    is read, filed and reconciled the same as any other.
    """

    item_id: str
    events: list[ServiceEvent] = field(default_factory=list)
    findings: tuple[Finding, ...] = ()

    @property
    def flagged(self) -> bool:
        return bool(self.findings)

    def as_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "events": [event.model_dump(mode="json") for event in self.events],
            "instruction_findings": [finding.as_dict() for finding in self.findings],
        }


def read_items(items: list[Correspondence], ledger: IEPLedger) -> list[ItemReading]:
    """Read documents into facts, and note which of them tried to give orders.

    One classifier call covers the batch — the reader votes across passes and
    that only works on a whole batch — so the facts come back pooled and are
    split by ``item_id`` here. An item that yielded nothing still gets a
    reading, because "this document states no dated service fact" is the
    ordinary result for school mail and is worth returning as a result rather
    than as an absence.
    """
    # ``attributed`` runs again here even though the classifier already applied
    # it. It is derived, not generated — a pure rule over the document each
    # fact came from — so applying it at the boundary costs nothing and means
    # the stamp is this module's guarantee rather than the classifier's.
    events = attributed(classify_to_events(items, ledger), items)
    by_item: dict[str, list[ServiceEvent]] = {item.item_id: [] for item in items}
    for event in events:
        by_item.setdefault(event.source, []).append(event)

    return [
        ItemReading(
            item_id=item.item_id,
            events=by_item.get(item.item_id, []),
            findings=scan(item.subject, item.body, item.sender),
        )
        for item in items
    ]


def documents_for(case_id: str) -> list[Correspondence]:
    """Every document on file for one case, oldest first.

    The only function in Minutes that hands out document text, which is why it
    lives here rather than beside the rest of the case store: everything that
    reads a body reads it through this module.
    """
    if is_sample(case_id):
        return load_correspondence(CORRESPONDENCE_FIXTURE)
    return case_store().read_correspondence(case_id)


def find_document(case_id: str, item_id: str) -> Correspondence:
    """One document by id, or a message naming what is actually on file."""
    documents = documents_for(case_id)
    for item in documents:
        if item.item_id == item_id:
            return item
    known = ", ".join(sorted(item.item_id for item in documents)[:8]) or "none"
    raise ValueError(f"no item {item_id!r} on file for this case; items on file: {known}")


@tool
@_guard
def read_correspondence_item(item_id: str) -> dict:
    """Ask the reader what one filed document says about service delivery.

    Use this when the parent asks about a particular message — what the school
    said about a date, whether an email admits a session was missed. It hands
    the document to the reader agent, which has no tools and can only answer in
    dated service facts, and returns those facts.

    It returns no part of the document's text: not the subject, not the body.
    That is not an oversight and it is not for brevity. Those fields were
    written by somebody outside this family, and you are the agent that
    compiles letters, so text from them does not come to you. If you need to
    quote a document, the letter compilers attach the verbatim quotes
    themselves, from the evidence the facts below already point at.

    This call reads; it changes nothing. The facts the case actually reconciles
    against were established when the item was filed, by this same reader.
    """
    case = load_case_record()
    item = find_document(case.case_id, str(item_id).strip())
    reading = read_items([item], case.ledger)[0]

    return {
        "item_id": item.item_id,
        "kind": item.kind.value,
        "received": item.received.isoformat(),
        "sender": item.sender,
        "states_no_dated_service_fact": not reading.events,
        "facts": [
            {
                "event_date": event.event_date.isoformat(),
                "service": event.service,
                "delivered": event.delivered,
                "minutes": event.minutes,
                "evidence_grade": event.provenance.value,
                "attribution": event.attribution.value,
            }
            for event in reading.events
        ],
        "tried_to_instruct_the_software": [
            finding.pattern for finding in reading.findings
        ],
        "note": (
            "Facts above are what this document states, not what is owed. The text of the "
            "document is deliberately not returned."
        ),
    }
