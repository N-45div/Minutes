"""The monthly Statement -- the artifact the product is named for.

Schools send report cards about the child. Nobody sends a statement about
the school. Once a month Minutes sends exactly one thing: what was owed,
what was delivered, what is short, how each of those minutes is evidenced,
and what is genuinely waiting on the parent. The rest of the month it stays
silent, which is only tolerable if this one artifact is complete.

Nothing here calls a model. A statement is arithmetic over the
reconciliation it was handed, composed once into an ordered document and
then rendered twice -- Markdown for reading, plain text for an email body --
so the two renderings can never disagree about a number.

Three properties hold of everything below, because this is the document a
parent forwards to a district:

1. The statement reports the reconciliation and stops. It never
   characterizes a shortfall as a violation, a denial, or a material
   failure: that is an individualized adjudicative call this tool does not
   make. That is not merely a property of the strings written here --
   :func:`_screen` gates every piece of text that arrives from outside this
   module (decision-card prose, extracted service names and deadline
   descriptions) against :data:`~minutes.letters.LEGAL_CONCLUSION_PHRASES`
   and against the verified citation table, the way ``validate_letter``
   gates a letter. A card whose wording states a conclusion is held back
   with a visible note rather than rendered.
2. The lead number carries its own correction. ``shortfall_minutes`` is
   owed minus delivered, and absent evidence subtracts exactly like
   non-delivery (see :mod:`minutes.reconcile`), so wherever undocumented
   minutes are part of the shortfall the headline says so in the same
   sentence as the figure -- not two sections below it.
3. Silence is never dressed up as good news. A period in which nothing
   could be reconciled, or in which every delivered minute rests on the
   parent's own log, says exactly that in the opening lines.
"""

from __future__ import annotations

import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from datetime import date

from .letters import ALLOWED_CITATIONS, LEGAL_CONCLUSION_PHRASES
from .models import (
    DeadlineState,
    DeadlineStatus,
    DecisionCard,
    IEPLedger,
    ReconciliationResult,
    RecordsRequest,
    RequestState,
    Statement,
    StatementLine,
    Urgency,
)

__all__ = [
    "DISCLAIMER",
    "HOURS_THRESHOLD",
    "LEGAL_SERVICES_CITATION",
    "METHOD_NOTE",
    "NOTHING_SCHEDULED_HEADLINE",
    "NOTHING_SCHEDULED_NOTE",
    "NO_ACTIVITY_HEADLINE",
    "NO_ACTIVITY_NOTE",
    "NO_SHORTFALL_HEADLINE",
    "QUIET_MONTH_NOTE",
    "WITHHELD_CARD_NOTE",
    "WITHHELD_CELL",
    "build_statement",
    "render_markdown",
    "render_text",
]

# Past ten hours a minute count stops conveying scale; below it, minutes are
# the unit the IEP itself is written in and the gloss is only noise.
HOURS_THRESHOLD = 600


def _verified_citation(citation: str) -> str:
    """Take a regulation from the verified table or refuse to load.

    The statement is the document most likely to be forwarded to a district,
    so its one citation is looked up in :data:`minutes.letters.ALLOWED_CITATIONS`
    rather than typed here and trusted. If that table ever stops carrying it,
    importing this module fails loudly instead of shipping an unverified
    reference in a parent-facing artifact.
    """
    if citation not in ALLOWED_CITATIONS:  # pragma: no cover -- import-time guard
        raise ValueError(f"{citation!r} is not in the verified citation table")
    return citation


LEGAL_SERVICES_CITATION = _verified_citation("34 CFR 300.507(b)")

NO_SHORTFALL_HEADLINE = "No shortfall this period."
NO_ACTIVITY_HEADLINE = "No service from the IEP was reconciled for this period."
NO_ACTIVITY_NOTE = (
    "A period with nothing to reconcile is not the same as a period that went "
    "well, and this statement will not read it as one. It usually means the "
    "ledger holds no service covering these dates, or that the records for "
    "them have not reached Minutes yet. It is worth checking that the IEP was "
    "read in full and that this period's correspondence was loaded before "
    "treating the quiet as real."
)
NOTHING_SCHEDULED_HEADLINE = "No minutes from the IEP were scheduled for these dates."
NOTHING_SCHEDULED_NOTE = (
    "Every service on this ledger has a start and an end date, and the dates "
    "above fall outside all of them, so nothing was promised for this period "
    "and nothing is short. This is a statement about the calendar, not about "
    "how the services went."
)
QUIET_MONTH_NOTE = (
    "Nothing this month needs a decision from you. The ledger stays open and "
    "the clocks keep running; the next statement will say so if that changes."
)
METHOD_NOTE = (
    "Every figure above is arithmetic on the minutes your child's IEP itself "
    "states, reconciled against the records on file. It is a reconciliation, "
    "not a conclusion about what any of it means."
)
DISCLAIMER = (
    "Minutes is not a law firm and this statement is not legal advice. It is "
    "documentation compiled from your child's IEP and your own records, and it "
    "is not a substitute for the advice of an attorney. Your state's Parent "
    "Training and Information Center, its Protection and Advocacy agency, and "
    "the list of free or low-cost legal services the district must give you on "
    f"request ({LEGAL_SERVICES_CITATION}) are the places to take it next."
)

# What is printed in place of text that fails the gate. It is deliberately
# visible: the parent is told something was held back, never quietly shown a
# document with a hole in it.
WITHHELD_CELL = "(withheld -- see Minutes)"
WITHHELD_CARD_NOTE = (
    "An item prepared for this period is held back from this statement. Its "
    "wording characterizes what the school's conduct amounts to, or cites a "
    "regulation Minutes has not verified, and this statement reports "
    "arithmetic only. The item is still on your ledger under the id above; "
    "open it in Minutes to read it in full and decide what to do with it."
)

# The characterizations only counsel or an adjudicator may make. The shared
# list from letters.py is the authority; the entries added here are the looser
# forms that reach a statement through a decision card's free text rather than
# through a compiled letter's assembled sentences.
_EXTRA_CONCLUSION_PHRASES: tuple[str, ...] = (
    "violation",
    "violated",
    "violates",
    "unlawful",
    "illegal",
    "you are owed",
    "you have a claim",
    "you have a case",
    "file for due process",
    "denial of a fape",
    "failure to provide fape",
    "compensatory education is owed",
)

_CONCLUSION_PHRASES: tuple[str, ...] = tuple(
    dict.fromkeys(LEGAL_CONCLUSION_PHRASES + _EXTRA_CONCLUSION_PHRASES)
)

# Section-level anchors of the verified table, so text may name a subsection
# the table stores in a range form while an invented section still fails.
_CITE_RE = re.compile(r"34 CFR \d+\.\d+|20 U\.S\.C\. \d+")
# The same tokens with whatever subsection follows, for the one job of keeping
# a citation on one line when the email body is wrapped.
_CITE_TOKEN_RE = re.compile(r"(?:34 CFR \d+\.\d+|20 U\.S\.C\. \d+)\S*")
_ANCHOR_RE = re.compile(r"^(34 CFR \d+\.\d+|20 U\.S\.C\. \d+)")
_ALLOWED_ANCHORS: frozenset[str] = frozenset(
    match.group(1) for citation in ALLOWED_CITATIONS if (match := _ANCHOR_RE.match(citation))
)

_SUMMED_FIELDS = (
    "owed_minutes",
    "delivered_minutes",
    "shortfall_minutes",
    "school_confirmed_minutes",
    "parent_observed_minutes",
    "undocumented_minutes",
)

_OUTSTANDING_REQUEST_STATES = (RequestState.SENT, RequestState.UNANSWERED_OVERDUE)

_URGENCY_ORDER = {
    Urgency.DEADLINE_IMMINENT: 0,
    Urgency.TIME_SENSITIVE: 1,
    Urgency.ROUTINE: 2,
}

_URGENCY_LABEL = {
    Urgency.DEADLINE_IMMINENT: "deadline imminent",
    Urgency.TIME_SENSITIVE: "time-sensitive",
    Urgency.ROUTINE: "routine",
}

_TEXT_WIDTH = 76

# Stands in for the spaces inside a citation while a paragraph is wrapped.
_NOWRAP = "\x00"


# ---------------------------------------------------------------------------
# The gate on text that comes from outside this module
# ---------------------------------------------------------------------------


def _flatten(value: str) -> str:
    """Collapse a value onto one line.

    Service names, deadline descriptions and card facts are extracted from a
    real document by a model, so any of them may arrive carrying a newline or
    a tab. A newline inside a cell shifts the figures into the wrong column of
    a Markdown table and breaks the width arithmetic of the plain-text one.
    The wrap sentinel is stripped here so that extracted text can never be
    mistaken for one of :func:`_fill`'s protected spaces.
    """
    return " ".join(str(value).replace(_NOWRAP, " ").split())


def _states_a_conclusion(value: str) -> bool:
    lowered = _flatten(value).lower()
    return any(phrase in lowered for phrase in _CONCLUSION_PHRASES)


def _cites_outside_the_table(value: str) -> bool:
    return any(found not in _ALLOWED_ANCHORS for found in _CITE_RE.findall(_flatten(value)))


def _screen(value: str) -> bool:
    """True when this text may not appear in a parent-facing statement."""
    return _states_a_conclusion(value) or _cites_outside_the_table(value)


def _cell(value: str) -> str:
    """A single-line, screened cell value.

    A row keeps its figures and its dates even when its label is withheld:
    the arithmetic is what the statement is for, and losing a clock because a
    description was badly worded would be the worse failure.

    The screen is deliberately blunt, so it will occasionally withhold an
    innocent label -- an IEP behaviour goal that happens to use the word
    "violates", say. That trade is taken knowingly: a false positive costs one
    label and announces itself in the artifact, while a false negative puts an
    accusation this tool is not entitled to make into a document the parent
    forwards to the district.
    """
    flat = _flatten(value)
    return WITHHELD_CELL if _screen(flat) else flat


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def build_statement(
    ledger: IEPLedger,
    result: ReconciliationResult,
    deadlines: list[DeadlineStatus],
    requests: list[RecordsRequest],
    decisions: list[DecisionCard],
) -> Statement:
    """Compile one period's ledger, clocks, requests and decisions into the
    single artifact the parent receives."""
    return Statement(
        student_alias=ledger.student_alias,
        period_start=result.period_start,
        period_end=result.period_end,
        lines=_lines(result),
        open_deadlines=_open_deadlines(deadlines),
        unanswered_requests=_outstanding_requests(requests),
        decisions=_ordered_decisions(decisions),
    )


def _lines(result: ReconciliationResult) -> list[StatementLine]:
    """One line per service, summing however many periods reconciliation
    split that service into. Services absent from the reconciliation are
    absent here too: a statement never invents a line."""
    totals: dict[str, Counter[str]] = {}
    for shortfall in result.shortfalls:
        bucket = totals.setdefault(shortfall.service, Counter())
        for field in _SUMMED_FIELDS:
            bucket[field] += getattr(shortfall, field)

    lines = [StatementLine(service=service, **bucket) for service, bucket in totals.items()]
    # Biggest gap first: the line that matters should not be scrolled to.
    return sorted(lines, key=lambda line: (-line.shortfall_minutes, line.service))


def _open_deadlines(deadlines: list[DeadlineStatus]) -> list[DeadlineStatus]:
    open_ones = [status for status in deadlines if status.state is not DeadlineState.MET]
    return sorted(
        open_ones,
        key=lambda status: (status.days_remaining, status.deadline.due, status.deadline.description),
    )


def _outstanding_requests(requests: list[RecordsRequest]) -> list[RecordsRequest]:
    """Sent but not answered. A draft was never sent, so nothing is owed on it."""
    outstanding = [req for req in requests if req.state in _OUTSTANDING_REQUEST_STATES]
    return sorted(outstanding, key=lambda req: (req.sent_on or date.max, req.request_id))


def _ordered_decisions(decisions: list[DecisionCard]) -> list[DecisionCard]:
    return sorted(
        decisions,
        key=lambda card: (_URGENCY_ORDER[card.urgency], card.deadline or date.max, card.card_id),
    )


# ---------------------------------------------------------------------------
# The composed document, shared by both renderers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Heading:
    text: str
    level: int = 2


@dataclass(frozen=True)
class _Lead:
    """The one number the parent came for."""

    text: str


@dataclass(frozen=True)
class _Para:
    text: str


@dataclass(frozen=True)
class _Table:
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    aligns: tuple[str, ...] = ()

    def alignment(self) -> tuple[str, ...]:
        """Figures read as a column only if they are right-aligned; prose does not."""
        return self.aligns or ("l",) + ("r",) * (len(self.headers) - 1)


@dataclass(frozen=True)
class _Bullets:
    items: tuple[str, ...]


_Block = _Heading | _Lead | _Para | _Table | _Bullets


def _compose(statement: Statement) -> list[_Block]:
    blocks: list[_Block] = [
        _Heading(f"Minutes statement for {_flatten(statement.student_alias)}", level=1),
        _Para(
            f"Service period {_fmt_date(statement.period_start)} to "
            f"{_fmt_date(statement.period_end)}. All figures are in minutes."
        ),
        _Heading("This period"),
        _Lead(_headline(statement)),
    ]
    summary = _summary(statement)
    if summary is not None:
        blocks.append(_Para(summary))
    if not statement.lines:
        # Nothing reconciled is a pipeline result, not a calm month.
        blocks.append(_Para(NO_ACTIVITY_NOTE))
    elif _is_quiet(statement):
        blocks.append(_Para(QUIET_MONTH_NOTE))

    blocks += _decision_blocks(statement)
    blocks += _service_blocks(statement)
    blocks += _deadline_blocks(statement)
    blocks += _request_blocks(statement)
    blocks += [_Heading("How to read this"), _Para(METHOD_NOTE), _Para(DISCLAIMER)]
    return blocks


def _total(statement: Statement, field: str) -> int:
    return sum(getattr(line, field) for line in statement.lines)


def _headline(statement: Statement) -> str:
    """One number, alone on its line -- carrying whatever qualifier it needs
    to be read correctly, because this is the sentence that gets forwarded."""
    if not statement.lines:
        return NO_ACTIVITY_HEADLINE
    if statement.total_owed == 0:
        return NOTHING_SCHEDULED_HEADLINE
    if statement.total_shortfall == 0:
        return NO_SHORTFALL_HEADLINE

    short = _fmt_minutes(statement.total_shortfall)
    undocumented = _total(statement, "undocumented_minutes")
    if undocumented == statement.total_shortfall:
        return f"{short} short this period, all of it with no record either way."
    if undocumented:
        return (
            f"{short} short this period, "
            f"{_fmt_minutes(undocumented)} of it with no record either way."
        )
    return f"{short} short this period."


def _summary(statement: Statement) -> str | None:
    if not statement.lines:
        return None
    if statement.total_owed == 0:
        return NOTHING_SCHEDULED_NOTE

    confirmed = _total(statement, "school_confirmed_minutes")
    # Parent-observed minutes are never asserted as established delivery --
    # the same rule letters.py applies to a compiled letter's sentences.
    only_your_log = statement.total_delivered > 0 and confirmed == 0

    if statement.total_shortfall == 0:
        sentence = (
            f"All {_fmt_minutes(statement.total_owed)} promised over these dates "
            "are documented as delivered"
        )
        return sentence + (", entirely from your own log." if only_your_log else ".")

    delivered = f"{_fmt_minutes(statement.total_delivered)} documented as delivered"
    if only_your_log:
        delivered += ", all of it from your own log"
    short = sum(1 for line in statement.lines if line.shortfall_minutes)
    sentences = [
        f"{_fmt_minutes(statement.total_owed)} owed.",
        f"{delivered}.",
        f"{short} of {len(statement.lines)} "
        f"{_plural(len(statement.lines), 'service')} on the ledger came up short.",
    ]
    undocumented = _total(statement, "undocumented_minutes")
    if undocumented:
        sentences.append(
            f"Nobody has recorded {_fmt_minutes(undocumented)} of that shortfall "
            "either way, which is a gap in the evidence rather than a record of "
            "non-delivery."
        )
    return " ".join(sentences)


def _is_quiet(statement: Statement) -> bool:
    """A month is quiet only if something was actually reconciled and nothing
    anywhere is waiting -- not merely when the shortfall is zero.

    An empty ledger makes the shortfall trivially zero, and every real cause of
    an empty ledger (nothing extracted, no obligation covering the window, no
    correspondence loaded) is a failure rather than a calm month, so the
    presence of lines is part of the test.
    """
    pressing = {DeadlineState.OVERDUE, DeadlineState.DUE_SOON}
    return (
        bool(statement.lines)
        and statement.total_shortfall == 0
        and _total(statement, "undocumented_minutes") == 0
        and not statement.decisions
        and not statement.unanswered_requests
        and not any(status.state in pressing for status in statement.open_deadlines)
    )


def _service_blocks(statement: Statement) -> list[_Block]:
    if not statement.lines:
        return []

    service_rows = [
        (
            _cell(line.service),
            _fmt_cell(line.owed_minutes),
            _fmt_cell(line.delivered_minutes),
            _fmt_cell(line.shortfall_minutes),
        )
        for line in statement.lines
    ]
    service_rows.append(
        (
            "TOTAL",
            _fmt_cell(statement.total_owed),
            _fmt_cell(statement.total_delivered),
            _fmt_cell(statement.total_shortfall),
        )
    )

    confirmed = _total(statement, "school_confirmed_minutes")
    observed = _total(statement, "parent_observed_minutes")
    undocumented = _total(statement, "undocumented_minutes")
    evidence_rows = [
        (
            _cell(line.service),
            _fmt_cell(line.school_confirmed_minutes),
            _fmt_cell(line.parent_observed_minutes),
            _fmt_cell(line.undocumented_minutes),
        )
        for line in statement.lines
    ]
    evidence_rows.append(
        ("TOTAL", _fmt_cell(confirmed), _fmt_cell(observed), _fmt_cell(undocumented))
    )

    blocks: list[_Block] = [
        _Heading("Services"),
        _Table(("Service", "Owed", "Delivered", "Short"), tuple(service_rows)),
        _Heading("Where the evidence comes from"),
        _Table(
            ("Service", "School-confirmed", "Parent-observed", "Undocumented"),
            tuple(evidence_rows),
        ),
        _Para(
            "School-confirmed minutes come from the school's own records or its "
            "written statements. Parent-observed minutes come from your log: "
            "dated, but not the school's record. Undocumented minutes are minutes "
            "the IEP promised for which there is no evidence either way."
        ),
    ]
    if statement.total_delivered and not confirmed:
        blocks.append(
            _Para(
                "Nothing this period is confirmed by the school's own records. "
                "Every minute shown as delivered rests on your log."
            )
        )
    if undocumented:
        blocks.append(
            _Para(
                f"{_fmt_minutes(undocumented)} of what was promised have no "
                "evidence either way -- no school record and no entry in your log."
            )
        )
    return blocks


def _deadline_blocks(statement: Statement) -> list[_Block]:
    if not statement.open_deadlines:
        return [_Heading("Deadlines"), _Para("No deadlines from the IEP are open.")]

    rows = tuple(
        (
            _fmt_date(status.deadline.due),
            _cell(status.deadline.description),
            _days_phrase(status.days_remaining),
        )
        for status in statement.open_deadlines
    )
    return [
        _Heading("Deadlines"),
        _Table(("Due", "What is due", "Status"), rows, aligns=("l", "l", "r")),
    ]


def _request_blocks(statement: Statement) -> list[_Block]:
    if not statement.unanswered_requests:
        return [
            _Heading("Records requests outstanding"),
            _Para("No records request is outstanding."),
        ]

    rows = tuple(
        (
            _cell(request.request_id),
            ", ".join(_cell(service) for service in request.services) or "all services",
            _fmt_date(request.sent_on) if request.sent_on else "not sent",
            _outstanding_phrase(request, statement.period_end),
        )
        for request in statement.unanswered_requests
    )
    return [
        _Heading("Records requests outstanding"),
        _Table(("Request", "Covers", "Sent", "Outstanding"), rows, aligns=("l", "l", "l", "l")),
        _Para(
            "An unanswered request stays on the ledger with the date it was sent "
            "and the date a response was due, so the gap is dated rather than "
            "remembered."
        ),
    ]


def _decision_blocks(statement: Statement) -> list[_Block]:
    if not statement.decisions:
        return [_Heading("Waiting on you"), _Para("Nothing is waiting on you this month.")]

    blocks: list[_Block] = [_Heading("Waiting on you")]
    for card in statement.decisions:
        blocks += _decision(card, statement.period_end)
    return blocks


def _decision(card: DecisionCard, as_of: date) -> list[_Block]:
    """One card, or a note in its place.

    A card is a single recommendation: its title, its reasons and its facts
    argue for one action together. Half of a card, with the sentence that
    failed the gate quietly removed, would misrepresent the rest -- so a card
    that states a conclusion is held back whole, and the parent is told it was.
    The urgency and any deadline still show, because those are the card's
    clock and losing them would cost the parent something real.
    """
    timing = f"Urgency: {_URGENCY_LABEL[card.urgency]}."
    if card.deadline is not None:
        timing += (
            f" Deadline {_fmt_date(card.deadline)}"
            f" ({_days_phrase(_days_between(as_of, card.deadline))})."
        )

    passthrough = [card.title, card.why_now, *card.facts, card.recommended_action]
    if card.draft is not None:
        passthrough.append(card.draft.subject)
    if any(_screen(value) for value in passthrough):
        return [
            _Heading(f"An item is held back ({_cell(card.card_id)})", level=3),
            _Para(WITHHELD_CARD_NOTE),
            _Para(timing),
        ]

    blocks: list[_Block] = [
        _Heading(_flatten(card.title), level=3),
        _Para(f"Why now: {_flatten(card.why_now)}"),
    ]
    if card.facts:
        blocks.append(_Bullets(tuple(_flatten(fact) for fact in card.facts)))
    blocks.append(_Para(timing))
    blocks.append(_Para(f"Suggested next step: {_flatten(card.recommended_action)}"))
    if card.draft is not None:
        blocks.append(
            _Para(f'A draft letter is ready to review: "{_flatten(card.draft.subject)}".')
        )
    return blocks


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_markdown(statement: Statement) -> str:
    """The human-facing artifact."""
    chunks: list[str] = []
    for block in _compose(statement):
        match block:
            case _Heading(text=text, level=level):
                chunks.append(f"{'#' * level} {text}")
            case _Lead(text=text):
                chunks.append(f"**{text}**")
            case _Para(text=text):
                chunks.append(text)
            case _Table() as table:
                chunks.append(_markdown_table(table))
            case _Bullets(items=items):
                chunks.append("\n".join(f"- {item}" for item in items))
    return "\n\n".join(chunks) + "\n"


def render_text(statement: Statement) -> str:
    """The same statement as a plain-text email body."""
    chunks: list[str] = []
    for block in _compose(statement):
        match block:
            case _Heading(text=text, level=level):
                if level >= 3:
                    chunks.append(f"* {text}")
                else:
                    rule = "=" if level == 1 else "-"
                    chunks.append(f"{text.upper()}\n{rule * len(text)}")
            case _Lead(text=text):
                # The lead has to lead in this rendering too: plain text has no
                # bold, so it is ruled off above and below instead.
                body = _fill(text)
                rule = "-" * max(len(line) for line in body.splitlines())
                chunks.append(f"{rule}\n{body}\n{rule}")
            case _Para(text=text):
                chunks.append(_fill(text))
            case _Table() as table:
                chunks.append(_text_table(table))
            case _Bullets(items=items):
                chunks.append(
                    "\n".join(
                        _fill(item, initial_indent="  - ", subsequent_indent="    ")
                        for item in items
                    )
                )
    return "\n\n".join(chunks) + "\n"


def _fill(text: str, **kwargs: str) -> str:
    """Wrap a paragraph without ever breaking a citation across two lines.

    ``34 CFR 300.507(b)`` split as ``34`` / ``CFR 300.507(b)`` is still legible
    to a person, but the parent forwards this body to a district and a broken
    reference is one more thing for the reader to doubt.
    """
    protected = _CITE_TOKEN_RE.sub(lambda m: m.group(0).replace(" ", _NOWRAP), text)
    return textwrap.fill(protected, _TEXT_WIDTH, **kwargs).replace(_NOWRAP, " ")


def _markdown_table(table: _Table) -> str:
    rule = ["---:" if align == "r" else "---" for align in table.alignment()]
    lines = ["| " + " | ".join(table.headers) + " |", "| " + " | ".join(rule) + " |"]
    # A literal pipe in an extracted name would otherwise open a new column and
    # slide every figure on the row one place to the left.
    lines += ["| " + " | ".join(cell.replace("|", r"\|") for cell in row) + " |" for row in table.rows]
    return "\n".join(lines)


def _text_table(table: _Table) -> str:
    widths = [max(len(cell) for cell in column) for column in zip(table.headers, *table.rows)]
    aligns = table.alignment()

    def line(cells: tuple[str, ...]) -> str:
        return "  ".join(
            cell.rjust(width) if align == "r" else cell.ljust(width)
            for cell, width, align in zip(cells, widths, aligns)
        ).rstrip()

    return "\n".join(
        [line(table.headers), "  ".join("-" * width for width in widths), *(line(row) for row in table.rows)]
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _fmt_minutes(minutes: int) -> str:
    """Prose form: minutes, with hours once minutes stop being legible."""
    text = f"{minutes:,} {_plural(minutes, 'minute')}"
    if minutes >= HOURS_THRESHOLD:
        text += f" ({_fmt_hours(minutes)})"
    return text


def _fmt_cell(minutes: int) -> str:
    """Table form: the bare figure, with a compact hour gloss when large."""
    text = f"{minutes:,}"
    if minutes >= HOURS_THRESHOLD:
        text += f" ({_hours_figure(minutes)}h)"
    return text


def _hours_figure(minutes: int) -> str:
    hours = minutes / 60
    return f"{hours:.0f}" if abs(hours - round(hours)) < 0.05 else f"{hours:.1f}"


def _fmt_hours(minutes: int) -> str:
    figure = _hours_figure(minutes)
    return f"{figure} {'hour' if figure == '1' else 'hours'}"


def _fmt_date(day: date) -> str:
    return f"{day:%b} {day.day}, {day.year}"


def _days_phrase(days: int) -> str:
    if days > 0:
        return f"{days} {_plural(days, 'day')} remaining"
    if days == 0:
        return "due today"
    return f"{-days} {_plural(-days, 'day')} overdue"


def _outstanding_phrase(request: RecordsRequest, as_of: date) -> str:
    """How long a request has been waiting as of the close of the period.

    A statement is compiled after its period closes, so a request sent in the
    meantime is ordinary. It is reported rather than dropped -- the parent
    should see it is on the ledger -- but it has waited no days yet, and a
    negative count would be nonsense.
    """
    if request.sent_on is None:
        return "not sent"
    if request.sent_on > as_of:
        return "sent after this period"
    days = _days_between(request.sent_on, as_of)
    phrase = f"{days} {_plural(days, 'day')}"
    if request.response_due is not None and request.response_due < as_of:
        phrase += f", response was due {_fmt_date(request.response_due)}"
    return phrase


def _days_between(start: date, end: date) -> int:
    return (end - start).days


def _plural(count: int, word: str) -> str:
    return word if count == 1 else f"{word}s"
