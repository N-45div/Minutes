"""Letter compilation — the integrity mechanism.

A parent who sends the school an accusation that turns out to be wrong loses
the only thing they have: credibility with the people who decide. So Minutes
does not *write* letters, it *assembles* them. Every claim a body makes *about
the district* enters through :meth:`_Draft.cite`, which refuses to emit a
sentence it cannot footnote to an :class:`~minutes.models.EvidenceRef`. A claim
with no evidence is not softened or hedged — it is dropped, and the letter
reads as if the thought never occurred. That is why this module makes no LLM
call: there is no step at which prose could be invented around a fact that
isn't in the ledger.

The one other way a number reaches a body is :meth:`_Draft.derived`, for
arithmetic the parent performs over figures already cited above it ("we count
120 minutes as undocumented"). Those sentences assert nothing about the school
— they report the state of the parent's own records — so they carry no marker,
but they are recorded on the draft so that nothing enters a body unrecorded.

Four rules ride on top of the footnotes:

* School-confirmed and documented-silence facts are stated plainly.
* Parent-observed facts are always attributed ("We recorded at home that ...")
  and never asserted as established fact.
* Owed minutes with no evidence either way are described as *undocumented*.
  Undocumented is a statement about the parent's records, never an assertion
  that the district did not deliver.
* Minutes falling on a date the records note the child was absent are stated
  in the service's own paragraph and excluded from the difference the letter
  claims. That sentence is worth more to the family than the minutes it gives
  up: a district that can answer one line of a demand with "she was not in
  school that day" has been handed a reason to doubt every other line. The
  wording stays neutral — an absence is a fact about attendance, never an
  admission about the school and never a complaint about the child — and it
  says only what the record says. It names the dates the record names; it does
  not say a session was scheduled on one, because the ledger holds no schedule
  and a claim nothing in the evidence supports is exactly what this module
  exists to make impossible. Nor can an absence record become the evidence
  behind an escalation: those minutes are already out of the difference, so the
  record that gave them up may not double as the reason to ask for the rest
  (:func:`_is_evidenced_gap`).

A letter also only ever quotes an IEP provision whose own term overlaps the
period it is talking about (:func:`_obligation_for`). An expired or not-yet-
started provision is not a basis for minutes owed, and there is no fallback to
"the nearest match": the service is named and set aside instead.

Legal references are drawn only from :data:`ALLOWED_CITATIONS`, a fixed,
human-verified table, and :func:`validate_letter` checks the body's own prose
against it after normalizing citation spelling, so "34 C.F.R. 300.999",
"34 CFR  300.999", "20 USC 1499" and "Section 300.999 of Title 34" all fail the
same way. Reporter citations in a body are checked at the volume-and-page level
against the same table, which is what rejects a superseded reporter (Van Duyn's
vacated 481 F.3d 770) wherever it appears.

Nothing here applies law to the family's facts or states a legal conclusion;
the letters say what the records show and what the parent asks for.
:data:`LEGAL_CONCLUSION_PHRASES` and :data:`ACCUSATORY_PHRASES` are checked by
the same gate. Both are defense in depth, not the guarantee — a blocklist can
never be complete, and the guarantee is the cite-or-stay-silent construction
above. :func:`validate_letter` is the gate a letter must pass before a parent
is ever shown it; :func:`placeholders` reports the blanks a parent must fill
before one is sent.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from .models import (
    Deadline,
    DeadlineKind,
    DeadlineState,
    DeadlineStatus,
    EvidenceRef,
    IEPLedger,
    Letter,
    LetterCitation,
    LetterKind,
    Provenance,
    ReconciliationResult,
    RecordsRequest,
    RequestState,
    ServiceObligation,
    ServiceShortfall,
)
from .reconcile import ABSENCE_NOTE, NON_DELIVERY_NOTE, evidence_date

__all__ = [
    "ACCUSATORY_PHRASES",
    "ALLOWED_CITATIONS",
    "DISCLAIMER",
    "LEGAL_CONCLUSION_PHRASES",
    "PARENT_ATTRIBUTION_PHRASES",
    "compile_compensatory_request",
    "compile_deadline_reminder",
    "compile_records_request",
    "compile_shortfall_notice",
    "placeholders",
    "validate_letter",
]

# ---------------------------------------------------------------------------
# The verified citation table.
#
# Every entry below was read from the regulation or opinion text during the
# build's legal research. Letters may cite nothing else. An LLM never chooses a
# section number at runtime — the compilers pick from these constants, and
# validate_letter rejects anything else, which is what makes a fabricated
# citation impossible rather than merely unlikely.
# ---------------------------------------------------------------------------

_RECORDS_ACCESS_CITATIONS = frozenset(
    {
        "34 CFR 300.501(a)",
        "34 CFR 300.611(b)",
        "34 CFR 300.611(c)",
        "34 CFR 300.613(a)",
        "34 CFR 300.613(b)(1)-(3)",
        "34 CFR 300.613(c)",
        "34 CFR 300.614",
        "34 CFR 300.615",
        "34 CFR 300.616",
        "34 CFR 300.617(a) and (b)",
        "34 CFR 300.618 and 300.619",
        "34 CFR 300.624(a) and (b)",
        "34 CFR 99.3 (definition of 'Education records'), para. (a)(1)-(2)",
        "34 CFR 99.3 (definition of 'Education records'), para. (b)(1)",
        "34 CFR 99.3 (definition of 'Education records'), para. (b)(4)(i)-(iii)",
        "34 CFR 99.10(a)",
        "34 CFR 99.10(b)",
        "34 CFR 99.10(c)",
        "34 CFR 99.10(d)",
        "34 CFR 99.10(e)",
        "34 CFR 99.11(a) and (b)",
    }
)

_SERVICE_DELIVERY_CITATIONS = frozenset(
    {
        "20 U.S.C. 1401(9)(D)",
        "20 U.S.C. 1414(d)(1)(A)(i)(III)",
        "20 U.S.C. 1414(d)(4)(A)(i)",
        "34 CFR 300.17(d)",
        "34 CFR 300.34(a)",
        "34 CFR 300.101(a)",
        "34 CFR 300.320(a)(3)(i)-(ii)",
        "34 CFR 300.320(a)(4)",
        "34 CFR 300.320(a)(7)",
        "34 CFR 300.323(a)",
        "34 CFR 300.323(c)(2)",
        "34 CFR 300.323(d)(1)-(2)",
        "34 CFR 300.324(a)(4)(i)",
        "34 CFR 300.324(a)(6)",
        "34 CFR 300.324(b)(1)(i)",
        "34 CFR 300.324(b)(1)(ii)(A)",
    }
)

_TIMELINE_CITATIONS = frozenset(
    {
        "20 U.S.C. 1414(a)(2)(B)",
        "20 U.S.C. 1415(b)(3)",
        "20 U.S.C. 1415(c)(1)",
        "34 CFR 300.11(a)",
        "34 CFR 300.11(b)",
        "34 CFR 300.301(c)(1)(i)",
        "34 CFR 300.301(c)(1)(ii)",
        "34 CFR 300.303(a)(1)-(2)",
        "34 CFR 300.303(b)(1)",
        "34 CFR 300.303(b)(2)",
        "34 CFR 300.322(a)(1)-(2) and (b)(1)(i)",
        "34 CFR 300.322(f)",
        "34 CFR 300.323(c)(1)",
        "34 CFR 300.503(a)(1)-(2)",
        "34 CFR 300.503(b)(1)-(7)",
        "34 CFR 300.503(c)(1)-(2)",
        "34 CFR 300.504(a)(1)-(4)",
    }
)

_REMEDY_CITATIONS = frozenset(
    {
        "20 U.S.C. 1415(f)(3)(E)(i)-(ii)",
        "34 CFR 300.151(b)(1)-(2)",
        "34 CFR 300.152(a) and (b)(1)",
        "34 CFR 300.153(a) and (b)",
        "34 CFR 300.153(c) and (d)",
        "34 CFR 300.506",
        "34 CFR 300.507(a)(2)",
        "34 CFR 300.507(b)",
        "34 CFR 300.510(a)(1) and (b)(1)",
        "34 CFR 300.513(a)(2)(i)-(iii)",
        "34 CFR 300.513(a)(2)(ii)",
        "34 CFR 300.515(a)",
    }
)

# Case authority is part of the verified vocabulary but is deliberately never
# placed in a compiled letter's legal_basis: characterizing this child's facts
# against a materiality standard is an adjudicative call for counsel or a
# hearing officer, not for a document assembler. Holding the citations here
# still lets validate_letter reject the traps — Van Duyn's superseded 481 F.3d
# 770 reporter, for one, which secondary sources hand out freely.
_CASE_AUTHORITY_CITATIONS = frozenset(
    {
        "Amanda J. v. Clark County School District, 267 F.3d 877 (9th Cir. 2001)",
        "Endrew F. v. Douglas Cnty. Sch. Dist. RE-1, 580 U.S. 386, 399 (2017)",
        "Gonzaga University v. Doe, 536 U.S. 273 (2002)",
        "Houston Indep. Sch. Dist. v. Bobby R., 200 F.3d 341, 349 (5th Cir. 2000)",
        "L.J. ex rel. N.N.J. v. Sch. Bd. of Broward Cnty., 927 F.3d 1203, 1211-13 (11th Cir. 2019)",
        "Letter to Clarke, 48 IDELR 77 (OSEP Mar. 8, 2007)",
        "M.C. v. Central Regional School District, 81 F.3d 389, 397 (3d Cir. 1996)",
        "Reid ex rel. Reid v. District of Columbia, 401 F.3d 516 (D.C. Cir. 2005)",
        "Schaffer ex rel. Schaffer v. Weast, 546 U.S. 49 (2005)",
        "Sumter Cnty. Sch. Dist. 17 v. Heffernan ex rel. TH, 642 F.3d 478, 484 (4th Cir. 2011)",
        "Van Duyn ex rel. Van Duyn v. Baker Sch. Dist. 5J, 502 F.3d 811, 822 (9th Cir. 2007)",
    }
)

ALLOWED_CITATIONS: frozenset[str] = (
    _RECORDS_ACCESS_CITATIONS
    | _SERVICE_DELIVERY_CITATIONS
    | _TIMELINE_CITATIONS
    | _REMEDY_CITATIONS
    | _CASE_AUTHORITY_CITATIONS
)

DISCLAIMER = (
    "Minutes compiled this letter from your child's IEP and from records you provided. "
    "Minutes is not a law firm, does not give legal advice, and is not a substitute for "
    "the advice of an attorney. Nothing in this letter states a legal conclusion about "
    "the district's conduct. For advice about your situation, contact an attorney, your "
    "state's Parent Training and Information Center, or your state Protection and "
    "Advocacy agency; if you request the information, or if a due process complaint is "
    "filed, the district must inform you of free or low-cost legal and other relevant "
    "services in your area (34 CFR 300.507(b))."
)

# A compiled letter reports arithmetic and asks questions. These are the
# characterizations only counsel or an adjudicator may make.
#
# This list and ACCUSATORY_PHRASES below are defense in depth, not the product's
# guarantee. Any blocklist loses to paraphrase; what actually holds is that a
# body sentence about the district can only come from _Draft.cite, which has no
# way to say anything an EvidenceRef does not already say.
LEGAL_CONCLUSION_PHRASES: tuple[str, ...] = (
    "denial of fape",
    "denied fape",
    "denied a fape",
    "denial of a free appropriate public education",
    "denied a free appropriate public education",
    "denies a free appropriate public education",
    "denies my child a free appropriate public education",
    "denying a free appropriate public education",
    "material failure",
    "materially failed",
    "failure to implement",
    "failed to implement",
    "violated the idea",
    "violation of the idea",
    "violated 34 cfr",
    "violated federal law",
    "violated state law",
    "broke federal law",
    "broke the law",
    "against the law",
    "constitutes a violation",
    "is a violation",
    "is unlawful",
    "is illegal",
    "you are entitled to",
    "we are entitled to",
    "we are owed",
    "the district owes",
    "legally required to compensate",
)

# Phrasings that would turn a reconciliation into an accusation. What the
# parent has is an arithmetic gap and an absence of records, which is a
# different and more defensible thing, so none of these may reach any body —
# not only the four this module compiles, but any Letter another module
# assembles and hands to validate_letter.
ACCUSATORY_PHRASES: tuple[str, ...] = (
    "were missed",
    "was missed",
    "missed sessions",
    "missed minutes",
    "were not delivered",
    "was not delivered",
    "never delivered",
    "did not deliver",
    "failed to deliver",
    "failed to provide",
    "refused to provide",
    "withheld services",
    "denied services",
    "shortchanged",
    "skipped sessions",
    "you owe",
    "owes us",
    "owes my child",
    # "the district owes" lives in LEGAL_CONCLUSION_PHRASES above; it is a
    # conclusion about entitlement, not only a characterization of delivery.
)

# A parent-observed claim must carry one of these in its own sentence.
PARENT_ATTRIBUTION_PHRASES: tuple[str, ...] = (
    "we recorded at home",
    "our home log",
    "our own log",
    "we observed at home",
    "our record at home",
    "parent observation",
    "parent-observed",
)

_MARKER_RE = re.compile(r"\[(\d+)\]")

# A blank only the parent can fill. Emitted with a sentinel so a sender can
# find every one of them mechanically instead of by reading.
_PLACEHOLDER_RE = re.compile(r"\[\[PARENT:\s*(.*?)\]\]", re.DOTALL)


# ---------------------------------------------------------------------------
# Citation matching.
#
# A citation gate that recognizes exactly one spelling is not a gate. Both the
# verified table and any prose being checked against it go through
# _normalize_authority first, so "34 C.F.R. § 300.999", "34 CFR  300.999" and
# "Section 300.999 of Title 34" all reduce to the same anchor and all fail
# together.
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
# The negative lookahead keeps "U.S. Court" and "C.F.R.A." out of it.
_CFR_SPELLING_RE = re.compile(r"\bC\.\s*F\.\s*R\.?(?![a-z])", re.IGNORECASE)
_USC_SPELLING_RE = re.compile(r"\bU\.\s*S\.\s*C\.?(?![a-z])", re.IGNORECASE)
_CFR_WORD_RE = re.compile(r"\bCFR\b", re.IGNORECASE)
_USC_WORD_RE = re.compile(r"\bUSC\b", re.IGNORECASE)
_SECTION_SYMBOL_RE = re.compile(r"§+\s*")
_SECTION_OF_TITLE_RE = re.compile(
    r"\b[Ss]ections?\s+(\d+(?:\.\d+)?)\s+of\s+[Tt]itle\s+(\d{1,2})\b"
)

# "34 CFR 300.613" / "20 USC 1414" once spelling is normalized.
_REG_ANCHOR_RE = re.compile(r"\b(\d{1,2})\s+(CFR|USC)\s+(\d+(?:\.\d+)?)")
# A trailing section that shares the preceding title, as in "34 CFR 300.618 and 300.619".
_TRAILING_SECTION_RE = re.compile(r"\band\s+(\d+\.\d+)\b")
# Case and agency authority, matched at volume-and-page level so a pin cite
# ("502 F.3d 811, 822") checks against the same key as the bare reporter.
_REPORTER_RE = re.compile(
    r"\b\d+\s+(?:F\.\s?\d?d|F\.\s?Supp\.(?:\s?\d?d)?|U\.S\.|S\.\s?Ct\.|IDELR|EHLR|LRP)\s+\d+"
)


def _normalize_authority(text: str) -> str:
    """Reduce every spelling of a citation this project has seen to one form."""
    normalized = _WHITESPACE_RE.sub(" ", text)
    normalized = _CFR_SPELLING_RE.sub("CFR", normalized)
    normalized = _USC_SPELLING_RE.sub("USC", normalized)
    normalized = _CFR_WORD_RE.sub("CFR", normalized)
    normalized = _USC_WORD_RE.sub("USC", normalized)
    normalized = _SECTION_SYMBOL_RE.sub("", normalized)
    return _SECTION_OF_TITLE_RE.sub(r"\2 CFR \1", normalized)


def _regulation_anchors(text: str) -> set[str]:
    """Every "<title> <corpus> <section>" anchor named in ``text``."""
    normalized = _normalize_authority(text)
    anchors: set[str] = set()
    for match in _REG_ANCHOR_RE.finditer(normalized):
        title, corpus, section = match.groups()
        anchors.add(f"{title} {corpus} {section}")
        for trailing in _TRAILING_SECTION_RE.findall(normalized[match.end() :]):
            anchors.add(f"{title} {corpus} {trailing}")
    return anchors


def _reporter_citations(text: str) -> set[str]:
    """Every reporter citation in ``text``, keyed by volume and first page."""
    return {
        _WHITESPACE_RE.sub(" ", hit) for hit in _REPORTER_RE.findall(_normalize_authority(text))
    }


def _collect(extractor, entries) -> frozenset[str]:
    found: set[str] = set()
    for entry in entries:
        found |= extractor(entry)
    return frozenset(found)


# Section-level anchors, so prose may write "34 CFR 300.613(b)(2)" while the
# table stores "34 CFR 300.613(b)(1)-(3)" — an invented section still fails.
_ALLOWED_ANCHORS: frozenset[str] = _collect(_regulation_anchors, ALLOWED_CITATIONS)
# Reporter keys, so a body may pin-cite an opinion in the table but cannot
# reach for one that isn't — including the superseded Van Duyn reporter.
_ALLOWED_REPORTERS: frozenset[str] = _collect(_reporter_citations, ALLOWED_CITATIONS)


# ---------------------------------------------------------------------------
# Assembly primitives
# ---------------------------------------------------------------------------


@dataclass
class _Draft:
    """Body text plus the citations bound to it.

    The only way a factual claim reaches the body is :meth:`cite`, and
    :meth:`cite` returns ``None`` when it is handed no evidence. Callers drop
    what comes back as ``None``, so an unevidenced claim disappears from the
    letter instead of being written in weaker words.
    """

    blocks: list[str] = field(default_factory=list)
    citations: list[LetterCitation] = field(default_factory=list)
    derived_lines: list[str] = field(default_factory=list)

    def text(self, block: str) -> None:
        """Add prose that asserts no fact about the school (framing, requests,
        quotations of the regulations themselves)."""
        self.blocks.append(block.strip())

    def derived(self, sentence: str) -> str:
        """Record arithmetic the parent performs over figures already cited.

        These sentences carry no marker because they make no claim about the
        district — "we count 120 minutes as undocumented" is a statement about
        the state of the parent's records, and stays true whether or not the
        district has a log. Recording them here keeps the module's invariant
        literally true: nothing reaches a body without passing through
        :meth:`cite`, :meth:`derived`, or :meth:`text`.
        """
        sentence = sentence.strip()
        self.derived_lines.append(sentence)
        return sentence

    def cite(self, sentence: str, evidence: Sequence[EvidenceRef]) -> str | None:
        sentence = sentence.strip()
        if not evidence:
            return None
        markers = []
        for ref in evidence:
            marker = f"[{len(self.citations) + 1}]"
            self.citations.append(LetterCitation(marker=marker, claim=sentence, evidence=ref))
            markers.append(marker)
        return sentence + "".join(markers)

    def paragraph(self, *sentences: str | None) -> None:
        """Join the sentences that survived citation into one paragraph."""
        kept = [s for s in sentences if s]
        if kept:
            self.blocks.append(" ".join(kept))

    def body(self) -> str:
        return "\n\n".join(b for b in self.blocks if b)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _parent_line(ledger: IEPLedger) -> str:
    """Aliases usually end in an initial ('Maya R.'), so the sentence has to
    close itself carefully."""
    line = f"I am the parent of {ledger.student_alias}"
    return line if line.endswith(".") else line + "."


def _iep_ref(ledger: IEPLedger, quote: str) -> EvidenceRef:
    """The IEP is the district's own signed document, so quoting it is a
    school-confirmed fact."""
    return EvidenceRef(
        provenance=Provenance.SCHOOL_CONFIRMED,
        source=f"iep-{ledger.iep_date.isoformat()}",
        detail=quote,
    )


def _obligation_for(
    ledger: IEPLedger, service: str, period_start: date, period_end: date
) -> ServiceObligation | None:
    """The IEP provision for ``service`` whose own term overlaps the period.

    A provision that expired before the period began, or that had not started
    when the period ended, is not a basis for minutes owed in that period —
    quoting it would produce exactly the overstatement against the school this
    product exists to prevent. So there is no fallback to "the nearest match":
    when nothing overlaps, callers get ``None`` and route the service into the
    named-but-not-quantified branch instead.
    """
    name = service.strip().lower()
    covering = [
        o
        for o in ledger.obligations
        if o.service.strip().lower() == name
        and o.start_date <= period_end
        and o.end_date >= period_start
    ]
    return covering[0] if covering else None


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _provision_sentence(ledger: IEPLedger, obligation: ServiceObligation) -> str:
    return (
        f"The IEP dated {ledger.iep_date.isoformat()} provides {obligation.service} at "
        f"{obligation.minutes_per_session} minutes per session, "
        f"{_plural(obligation.sessions_per_period, 'session')} per {obligation.period.value}, "
        f"delivered by {_article(obligation.provider_role)} {obligation.provider_role} "
        f"in the {obligation.setting}."
    )


def _first_deadline(ledger: IEPLedger, kind: DeadlineKind) -> Deadline | None:
    matches = sorted((d for d in ledger.deadlines if d.kind is kind), key=lambda d: d.due)
    return matches[0] if matches else None


def _by_provenance(evidence: Sequence[EvidenceRef], provenance: Provenance) -> list[EvidenceRef]:
    return [e for e in evidence if e.provenance is provenance]


def _absence_refs(line: ServiceShortfall, provenance: Provenance) -> list[EvidenceRef]:
    """The records on this line that note the student as absent, one grade at a time.

    ``EvidenceRef`` carries no attribution field, so the refs are selected on
    the note :mod:`minutes.reconcile` writes into the detail and publishes as
    ``ABSENCE_NOTE``. Importing that constant rather than re-typing the
    sentence is what keeps the two modules from drifting into disagreement
    about which footnote supports the exclusion.
    """
    return [ref for ref in _by_provenance(line.evidence, provenance) if ABSENCE_NOTE in ref.detail]


def _delivery_refs(line: ServiceShortfall, provenance: Provenance) -> list[EvidenceRef]:
    """The records on this line that support a claim of minutes DELIVERED.

    A footnote list has to support the sentence it hangs from. A record saying
    a session did not happen — an absence record above all, whose minutes this
    letter has just given up — cannot be offered as a source for the minutes
    the district delivered. Selected on the notes :mod:`minutes.reconcile`
    publishes for exactly this, so the two modules agree by import rather than
    by memory.
    """
    return [
        ref
        for ref in _by_provenance(line.evidence, provenance)
        if NON_DELIVERY_NOTE not in ref.detail and ABSENCE_NOTE not in ref.detail
    ]


def _dates_phrase(dates: list[date]) -> str:
    """The dates named, as "on 2026-10-14" — never a bare plural.

    "absent on dates in this period", said of one record, reads as several
    absences. The dates are in the footnotes already, so naming them is both
    shorter and unarguable. Refs this module did not write carry no date
    (:func:`~minutes.reconcile.evidence_date` returns None), and the fallback
    claims no count at all.
    """
    if not dates:
        return "during this period"
    rendered = [day.isoformat() for day in dates]
    if len(rendered) == 1:
        return f"on {rendered[0]}"
    return "on " + ", ".join(rendered[:-1]) + f" and {rendered[-1]}"


def _absence_dates(refs: Sequence[EvidenceRef]) -> list[date]:
    return sorted({day for ref in refs if (day := evidence_date(ref)) is not None})


def _excused_sentences(draft: _Draft, ledger: IEPLedger, line: ServiceShortfall) -> list[str]:
    """State the minutes excluded for absence, and what they rest on.

    The parent gives these minutes up, so the letter says so out loud instead
    of quietly reporting a smaller difference. A district that finds one line
    of a demand answerable with "she was not in school that day" has a reason
    to doubt the rest; a district that reads the family struck those minutes
    itself has none.

    Split by evidence grade like every other claim here: the district's own
    record is stated plainly, the family's log is attributed to the family.
    Neither sentence characterizes anybody, and neither says more than the
    record does — the dates are named, and nothing is asserted about what was
    scheduled on them, because the ledger holds no schedule to assert it from.
    An absence is a fact about attendance: not an admission by the school, not
    a complaint about the child.

    If neither grade produced a citable record, the exclusion sentence goes
    with them. It is the one claim in this module that is about the CHILD, so
    it may not stand as unfootnoted arithmetic once the records under it are
    gone — cite or stay silent applies hardest to the sentence a family would
    least like to be asked to prove.
    """
    if not line.excused_minutes:
        return []

    alias = ledger.student_alias
    school = _absence_refs(line, Provenance.SCHOOL_CONFIRMED)
    parent = _absence_refs(line, Provenance.PARENT_OBSERVED)
    stated = [
        draft.cite(
            f"District records note {alias} as absent {_dates_phrase(_absence_dates(school))}.",
            school,
        ),
        draft.cite(
            f"We recorded at home that {alias} was absent "
            f"{_dates_phrase(_absence_dates(parent))}.",
            parent,
        ),
    ]
    kept = [sentence for sentence in stated if sentence]
    if not kept:
        return []

    dates = _absence_dates(school + parent)
    fell_on = "a date noted as an absence" if len(dates) == 1 else "dates noted as an absence"
    return kept + [
        draft.derived(
            f"{line.excused_minutes} minutes are excluded from the difference below as minutes "
            f"owed on {fell_on}, rather than asked of the district."
        )
    ]


def _totals_sentence(draft: _Draft, line: ServiceShortfall) -> str:
    """The line's whole arithmetic in one sentence, with every term the reader
    needs to reproduce it: owed, delivered, anything excluded for an absence,
    and the difference those leave."""
    excluded = (
        f", {line.excused_minutes} minutes excluded as falling on a noted absence"
        if line.excused_minutes
        else ""
    )
    return draft.derived(
        f"For this period: {line.owed_minutes} minutes owed, {line.delivered_minutes} minutes "
        f"documented as delivered{excluded}, a difference of {line.shortfall_minutes} minutes."
    )


def _silence_ref(request: RecordsRequest) -> EvidenceRef:
    """An overdue, unanswered request is itself dated evidence — that is the
    whole point of sending it early."""
    return EvidenceRef(
        provenance=Provenance.DOCUMENTED_SILENCE,
        source=request.request_id,
        detail=(
            f"Records request {request.request_id} covering "
            f"{request.covers_start.isoformat()} to {request.covers_end.isoformat()}, "
            f"sent {request.sent_on.isoformat() if request.sent_on else 'date not recorded'}, "
            f"response due {request.response_due.isoformat() if request.response_due else 'date not recorded'}; "
            "no response recorded as received."
        ),
    )


def _answered_ref(request: RecordsRequest) -> EvidenceRef:
    return EvidenceRef(
        provenance=Provenance.SCHOOL_CONFIRMED,
        source=request.request_id,
        detail=(
            f"Records request {request.request_id} sent "
            f"{request.sent_on.isoformat() if request.sent_on else 'date not recorded'}, "
            f"answered {request.answered_on.isoformat() if request.answered_on else 'date not recorded'}."
        ),
    )


def _request_history_sentence(request: RecordsRequest) -> tuple[str, list[EvidenceRef]] | None:
    """Render one records request as a dated fact, or nothing if it was never
    sent (a draft is not evidence of anything)."""
    if request.state is RequestState.UNANSWERED_OVERDUE:
        sentence = (
            f"Records request {request.request_id}, covering "
            f"{request.covers_start.isoformat()} through {request.covers_end.isoformat()}, "
            f"was sent on {request.sent_on.isoformat() if request.sent_on else 'a date not recorded'} "
            f"with a response due {request.response_due.isoformat() if request.response_due else 'on a date not recorded'}, "
            "and no response has been recorded as received."
        )
        return sentence, [_silence_ref(request)]
    if request.state is RequestState.ANSWERED:
        sentence = (
            f"Records request {request.request_id}, covering "
            f"{request.covers_start.isoformat()} through {request.covers_end.isoformat()}, "
            f"was answered on {request.answered_on.isoformat() if request.answered_on else 'a date not recorded'}."
        )
        return sentence, [_answered_ref(request)]
    return None


def _period_totals(matched: Sequence[ServiceShortfall]) -> tuple[int, int, int, int]:
    """Owed, delivered, excused and the difference, over the quotable lines only."""
    return (
        sum(line.owed_minutes for line in matched),
        sum(line.delivered_minutes for line in matched),
        sum(line.excused_minutes for line in matched),
        sum(line.shortfall_minutes for line in matched),
    )


def _terms_clause(ledger: IEPLedger, owed: int, delivered: int, excused: int) -> str:
    """Every term the summary figure was computed from, in one clause.

    A summary that names two terms and subtracts three is the fastest way to
    make a district doubt a letter: 8,520 minus 1,365 is not 7,050 unless the
    105 excused minutes are named too. The per-service paragraphs above already
    name them (:func:`_totals_sentence`), so the summary does as well, and the
    headline figure stays reproducible from the words around it.
    """
    return (
        f"the IEP provides {owed} minutes for this period, {delivered} minutes are documented "
        f"as delivered, and {excused} minutes fall on dates a record notes "
        f"{ledger.student_alias} was absent and are excluded"
    )


def _matched_shortfalls(ledger: IEPLedger, result: ReconciliationResult) -> list[ServiceShortfall]:
    """Reconciliation lines the IEP on file can actually be quoted for.

    A letter totals only these. Minutes from a service with no provision in the
    ledger are real to the parent but unquotable here, and a summary figure
    that silently included them would be a number the letter cannot source.
    """
    return [
        line
        for line in result.shortfalls
        if _obligation_for(ledger, line.service, line.period_start, line.period_end) is not None
    ]


def _is_evidenced_gap(line: ServiceShortfall) -> bool:
    """Do the district's own records speak to THE GAP — not merely to the period?

    A gap made only of minutes with no evidence either way is a gap in the
    *parent's* records, and what that calls for is a records request, not a
    request for compensatory services.

    The test is therefore on the short minutes themselves, not on whether some
    record of some kind sits on the line. A line can hold a shelf of
    school-confirmed records that document only the sessions that were
    delivered, and leave every short minute undocumented; asking a district to
    convene a team over minutes nobody has recorded either way overstates what
    the family holds. ``shortfall_minutes - undocumented_minutes`` is exactly
    the part of the gap a district record already speaks to, and it is what
    escalation rests on.

    An absence record cannot be that evidence even in principle, and this
    subtraction is why: :mod:`minutes.reconcile` has already taken those
    minutes out of the shortfall, so a record used to give minutes up can
    never double as the reason to ask for the rest.

    A documented silence is the one thing that escalates without documenting
    the gap, because it is evidence *about* the gap: an overdue, unanswered
    records request standing where a service log should be. By contract with
    ``reconcile.py`` a silence never shrinks the undocumented bucket, so it has
    to be recognized here or it would never escalate at all.
    """
    if line.shortfall_minutes <= 0:
        return False
    if line.shortfall_minutes > line.undocumented_minutes:
        return True
    return any(
        ref.provenance is Provenance.DOCUMENTED_SILENCE for ref in line.evidence
    )


def _reconciliation_blocks(draft: _Draft, ledger: IEPLedger, result: ReconciliationResult) -> None:
    """Write one paragraph per reconciled service.

    A service with no matching provision in the IEP on file is named and set
    aside rather than described: the letter cannot quote what the ledger does
    not hold. Callers total :func:`_matched_shortfalls`, never the whole
    reconciliation.
    """
    omitted: list[str] = []

    for line in result.shortfalls:
        obligation = _obligation_for(ledger, line.service, line.period_start, line.period_end)
        if obligation is None:
            omitted.append(line.service)
            continue

        provision = draft.cite(_provision_sentence(ledger, obligation), [_iep_ref(ledger, obligation.source_quote)])
        arithmetic = draft.derived(
            f"On that basis our reconciliation counts {line.owed_minutes} minutes owed for "
            f"{line.period_start.isoformat()} through {line.period_end.isoformat()}."
        )

        confirmed = None
        if line.school_confirmed_minutes:
            confirmed = draft.cite(
                f"District records account for {line.school_confirmed_minutes} of those minutes as delivered.",
                _delivery_refs(line, Provenance.SCHOOL_CONFIRMED),
            )

        observed = None
        if line.parent_observed_minutes:
            observed = draft.cite(
                f"We recorded at home a further {line.parent_observed_minutes} minutes as delivered; "
                "that is our observation and not a district record.",
                _delivery_refs(line, Provenance.PARENT_OBSERVED),
            )

        # Stated before the undocumented count, because the count is a share of
        # a difference these minutes are already out of.
        excused = _excused_sentences(draft, ledger, line)

        silence = None
        undocumented_note = None
        if line.undocumented_minutes:
            silence = draft.cite(
                f"{line.undocumented_minutes} of the minutes owed have no delivery record from the district.",
                _by_provenance(line.evidence, Provenance.DOCUMENTED_SILENCE),
            )
            undocumented_note = draft.derived(
                f"We count {line.undocumented_minutes} minutes as undocumented, which describes the state of "
                "our records rather than what took place at school."
            )

        totals = _totals_sentence(draft, line)

        draft.text(f"{line.service} — {line.period_start.isoformat()} to {line.period_end.isoformat()}")
        draft.paragraph(
            provision, arithmetic, confirmed, observed, *excused, silence, undocumented_note, totals
        )

    if omitted:
        many = len(omitted) > 1
        draft.text(
            ("The following services are" if many else "The following service is")
            + " tracked in our records but "
            + ("have" if many else "has")
            + " no matching provision in the IEP on file covering that period, so "
            + ("they are" if many else "it is")
            + " left out of this letter rather than described from memory: "
            + ", ".join(omitted)
            + "."
        )


def _signature_block(ledger: IEPLedger) -> str:
    return (
        "Please reply in writing.\n\n"
        f"Parent/guardian of {ledger.student_alias}\n"
        "Signature: ______________________    Date: ______________\n"
        "Preferred contact: ______________________"
    )


# ---------------------------------------------------------------------------
# The compiled letters
# ---------------------------------------------------------------------------


def compile_records_request(ledger: IEPLedger, request: RecordsRequest, *, today: date | None = None) -> Letter:
    """The active-discovery letter: exercise the records right on a cadence.

    Its value is not only the documents it may return. A dated, delivery-proved
    request freezes the record set (34 CFR 99.10(e)) and starts a clock whose
    expiry is itself evidence.
    """
    today = today or date.today()
    draft = _Draft()

    draft.text(
        f"Date: {today.isoformat()}\n"
        "To: Records Custodian and Director of Special Education\n"
        f"Re: Education records of {ledger.student_alias}, school year {ledger.school_year}\n"
        f"Request reference: {request.request_id}"
    )

    draft.text(
        f"{_parent_line(ledger)} Under 34 CFR 300.613(a) and 34 CFR 99.10(a) I am "
        "requesting to inspect and review, and to receive copies of, the education records described "
        f"below covering {request.covers_start.isoformat()} through {request.covers_end.isoformat()}."
    )

    history = _request_history_sentence(request)
    if history is not None:
        sentence, refs = history
        draft.paragraph(draft.cite(sentence, refs), "This letter repeats that request.")

    services = list(request.services) or [o.service for o in ledger.obligations]
    bullets = ["Services at issue:"]
    for name in services:
        obligation = _obligation_for(ledger, name, request.covers_start, request.covers_end)
        if obligation is None:
            bullets.append(
                f"- {name}: the IEP on file records no provision for this service covering "
                f"{request.covers_start.isoformat()} through {request.covers_end.isoformat()}, "
                "so no quotation is included here."
            )
            continue
        line = draft.cite(_provision_sentence(ledger, obligation), [_iep_ref(ledger, obligation.source_quote)])
        if line is not None:
            bullets.append(f"- {line}")
    draft.text("\n".join(bullets))

    draft.text(
        "Records requested. For each service listed above, and for the period stated above, all records "
        "documenting delivery of that service, including:\n"
        "- service logs, encounter logs, and session notes;\n"
        "- data sheets and progress-monitoring data;\n"
        "- attendance records for the service, and records of sessions canceled, rescheduled, shortened, "
        "or not provided as scheduled;\n"
        "- records of make-up sessions offered, scheduled, or held;\n"
        "- provider schedules and assignment records, including any period covered by a substitute or "
        "contracted provider;\n"
        "- communications concerning the scheduling or delivery of the service; and\n"
        "- all records submitted for Medicaid or other third-party billing for services to this child."
    )

    draft.text(
        "This request reaches records held by any party acting for the district, including contracted and "
        "independent providers (34 CFR 300.611(b) and 34 CFR 300.611(c); 34 CFR 99.3). I also request the "
        "list of the types and locations of education records collected, maintained, or used by the agency "
        "(34 CFR 300.616), the record of parties who have obtained access to these records (34 CFR 300.614), "
        "and, for any coded entries in the logs, the explanations and interpretations provided for by "
        "34 CFR 300.613(b) and 34 CFR 99.10(c)."
    )

    timing = [
        "Timing. 34 CFR 300.613(a) provides that the agency must comply without unnecessary delay and "
        "before any meeting regarding an IEP, any hearing, or any resolution session, and in no case more "
        "than 45 days after the request is made. 34 CFR 99.10(b) sets a parallel ceiling of 45 days from "
        "receipt and requires compliance within a reasonable period of time."
    ]
    annual_review = _first_deadline(ledger, DeadlineKind.ANNUAL_REVIEW)
    if annual_review is not None:
        noticed = draft.cite(
            f'The IEP records the following date: "{annual_review.description}", '
            f"no later than {annual_review.due.isoformat()}.",
            [_iep_ref(ledger, annual_review.source_quote)],
        )
        if noticed is not None:
            timing.append(noticed)
            timing.append(
                "If a meeting regarding the IEP is scheduled before the 45-day ceiling expires, the earlier "
                "date is the one that governs production."
            )
    draft.paragraph(*timing)

    draft.text(
        "Copies. I am asking for copies rather than an on-site appointment. 34 CFR 300.613(b) and "
        "34 CFR 99.10(d) provide for copies, or other arrangements, where failure to provide them would "
        "effectively prevent the parent from exercising the right to inspect and review.\n"
        "[[PARENT: state here the circumstance that would prevent on-site inspection — for example work "
        "schedule, distance, childcare, disability, or the volume of records. Minutes does not supply this "
        "fact for you.]]"
    )

    draft.text(
        "Fees. 34 CFR 300.617 and 34 CFR 99.11 provide that no fee may be charged to search for or to "
        "retrieve records, and that a fee for copies is permitted only where it would not effectively "
        "prevent the parent from exercising the right to inspect and review."
    )

    draft.text(
        "Preservation. 34 CFR 99.10(e) provides that an agency shall not destroy education records while "
        "a request to inspect and review them is outstanding. I ask that all records within the scope of "
        "this request be preserved, including records held by contracted providers."
    )

    draft.text(
        "If any part of this request is declined, or if responsive records do not exist, please say so in "
        "writing and identify which records are withheld or absent and on what basis."
    )

    draft.text(_signature_block(ledger))

    return Letter(
        kind=LetterKind.RECORDS_REQUEST,
        subject=(
            f"Request to inspect and review education records — {ledger.student_alias}, "
            f"{request.covers_start.isoformat()} to {request.covers_end.isoformat()}"
        ),
        body=draft.body(),
        citations=draft.citations,
        legal_basis=[
            "34 CFR 300.613(a)",
            "34 CFR 300.613(b)(1)-(3)",
            "34 CFR 300.611(b)",
            "34 CFR 300.611(c)",
            "34 CFR 300.614",
            "34 CFR 300.616",
            "34 CFR 300.617(a) and (b)",
            "34 CFR 300.501(a)",
            "34 CFR 99.3 (definition of 'Education records'), para. (a)(1)-(2)",
            "34 CFR 99.10(a)",
            "34 CFR 99.10(b)",
            "34 CFR 99.10(c)",
            "34 CFR 99.10(d)",
            "34 CFR 99.10(e)",
            "34 CFR 99.11(a) and (b)",
        ],
        disclaimer=DISCLAIMER,
    )


def compile_shortfall_notice(
    ledger: IEPLedger, result: ReconciliationResult, *, today: date | None = None
) -> Letter:
    """Put the arithmetic in front of the school and ask it to reconcile.

    It states owed, documented, and undocumented minutes, and asks a question.
    It does not characterize the difference — that is an adjudicative call this
    product does not make.
    """
    today = today or date.today()
    draft = _Draft()

    draft.text(
        f"Date: {today.isoformat()}\n"
        "To: Case Manager and Director of Special Education\n"
        f"Re: Service delivery reconciliation for {ledger.student_alias}, "
        f"{result.period_start.isoformat()} to {result.period_end.isoformat()}"
    )

    draft.text(
        f"{_parent_line(ledger)} I keep a record of the services the IEP provides and "
        "of what we can document as delivered. Below is that reconciliation for this period, with a source "
        "noted for every figure. I am writing to ask the district to compare it against its own records."
    )

    matched = _matched_shortfalls(ledger, result)
    owed, delivered, excused, total = _period_totals(matched)

    _reconciliation_blocks(draft, ledger, result)

    if not matched:
        draft.text(
            "No service in this period could be matched to a provision in the IEP on file, so this letter "
            "makes no findings. Please send the current IEP and the service records for the period above."
        )
    elif total == 0 and excused:
        # "The records reconcile" would be false here and would put a
        # disclaimer of concern in the district's file for a period in which
        # part of the promise was not delivered at all. The three figures are
        # stated instead, and nothing is inferred from them.
        draft.text(
            f"Across the services above, {_terms_clause(ledger, owed, delivered, excused)}, which "
            "leaves no difference for this period to state. I am not asking the district to account "
            "for the excluded minutes. I am asking the district to compare these figures against its "
            "own service logs and to tell me in writing where they differ."
        )
    elif total == 0:
        draft.text(
            "For this period the records we hold reconcile with the minutes the IEP provides. I am not "
            "raising a concern about delivery for this period; I am asking the district to confirm the "
            "figures against its own service logs so that the two records stay aligned."
        )
    else:
        opening = (
            f"Across the services above, {_terms_clause(ledger, owed, delivered, excused)}, leaving a "
            f"difference of {total} minutes for this period."
            if excused
            else (
                f"Across the services above the difference between the minutes the IEP provides and the "
                f"minutes documented as delivered is {total} minutes for this period."
            )
        )
        draft.text(
            f"{opening} That is arithmetic from the IEP's "
            "own numbers and from the records available to me; it is not a conclusion about the district's "
            "conduct, and where a figure is marked undocumented I am not stating that a session did or did "
            "not happen."
        )

    draft.text(
        "I am asking for three things:\n"
        "1. The district's service logs, session notes, and attendance records for these services for the "
        "period above, so the two records can be compared line by line (34 CFR 300.613(a)).\n"
        "2. A written statement of any minutes the district records as delivered that are not reflected "
        "above, with the dates and the provider.\n"
        "3. An IEP team meeting, or a written response, to reconcile any remaining difference."
    )

    draft.text(
        "For reference, 34 CFR 300.17(d) defines a free appropriate public education as including services "
        "provided in conformity with an IEP that meets the requirements of 34 CFR 300.320 through "
        "34 CFR 300.324, 34 CFR 300.101(a) provides that such an education must be available to eligible "
        "children, and 34 CFR 300.323(a) provides that an IEP must be in effect for the child at the "
        "beginning of each school year. 34 CFR 300.320(a)(7) requires the IEP to state the anticipated "
        "frequency, location, and duration of each service, and 34 CFR 300.323(c)(2) provides that, as soon "
        "as possible following development of the IEP, services are made available in accordance with it. "
        "If the district declines any part of this request, I ask for prior written notice under "
        "34 CFR 300.503."
    )

    draft.text(_signature_block(ledger))

    return Letter(
        kind=LetterKind.SHORTFALL_NOTICE,
        subject=(
            f"Service delivery reconciliation — {ledger.student_alias}, "
            f"{result.period_start.isoformat()} to {result.period_end.isoformat()}"
        ),
        body=draft.body(),
        citations=draft.citations,
        legal_basis=[
            "34 CFR 300.17(d)",
            "20 U.S.C. 1401(9)(D)",
            "34 CFR 300.101(a)",
            "34 CFR 300.323(a)",
            "34 CFR 300.323(c)(2)",
            "34 CFR 300.320(a)(7)",
            "34 CFR 300.320(a)(4)",
            "34 CFR 300.323(d)(1)-(2)",
            "34 CFR 300.324(a)(4)(i)",
            "34 CFR 300.324(a)(6)",
            "34 CFR 300.503(a)(1)-(2)",
            "34 CFR 300.613(a)",
        ],
        disclaimer=DISCLAIMER,
    )


def compile_compensatory_request(
    ledger: IEPLedger,
    result: ReconciliationResult,
    requests: list[RecordsRequest],
    *,
    today: date | None = None,
) -> Letter:
    """Ask the IEP team to consider compensatory services for a documented gap.

    The letter carries the reconciliation, the dated history of what was asked
    of the district and what came back, and a request for a team decision. It
    never names an amount the family is owed: the amount is individualized and
    is for the IEP team, an SEA complaint officer, or a hearing officer.

    It only *asks* when the gap is evidenced — school-confirmed records or a
    documented silence standing where a record should be. A gap made entirely
    of minutes with no evidence either way is a hole in the parent's own
    records, and the answer to that is to ask the district for its records, not
    to ask the team for compensatory services.
    """
    today = today or date.today()
    draft = _Draft()

    # A letter that asks for nothing must not be titled as if it asked for
    # something: with no documented gap there is nothing to request, and the
    # subject line has to say so before the school opens it.
    matched = _matched_shortfalls(ledger, result)
    owed, delivered, excused, total = _period_totals(matched)
    evidenced = any(_is_evidenced_gap(line) for line in matched)
    requesting = total > 0 and evidenced
    subject = (
        f"Request for an IEP team meeting to consider compensatory services — {ledger.student_alias}, "
        f"{result.period_start.isoformat()} to {result.period_end.isoformat()}"
        if requesting
        else (
            f"Service delivery reconciliation, no compensatory services requested — {ledger.student_alias}, "
            f"{result.period_start.isoformat()} to {result.period_end.isoformat()}"
        )
    )

    draft.text(
        f"Date: {today.isoformat()}\n"
        "To: Director of Special Education and IEP Team\n"
        f"Re: {subject}"
    )

    if requesting:
        draft.text(
            f"{_parent_line(ledger)} I am requesting an IEP team meeting to review service "
            "delivery for the period above and to consider whether compensatory services are appropriate. "
            "The reconciliation below is drawn from the IEP's own numbers and from the records available "
            "to me, with a source noted for every figure."
        )
    else:
        draft.text(
            f"{_parent_line(ledger)} I am writing about service delivery for the period above. The "
            "reconciliation below is drawn from the IEP's own numbers and from the records available to "
            "me, with a source noted for every figure."
        )

    _reconciliation_blocks(draft, ledger, result)

    history_sentences = []
    for request in requests:
        rendered = _request_history_sentence(request)
        if rendered is None:
            continue
        sentence, refs = rendered
        cited = draft.cite(sentence, refs)
        if cited is not None:
            history_sentences.append(cited)
    if history_sentences:
        draft.text("Records requested from the district:")
        draft.paragraph(*history_sentences)

    if not matched:
        draft.text(
            "No service in this period could be matched to a provision in the IEP on file, so this letter "
            "makes no request and states no figures. Please send the current IEP and the service records "
            "for the period above."
        )
    elif total == 0 and excused:
        # A period excused to zero is not a period that reconciles. Saying it
        # does would put a written concession in the district's file for
        # minutes that were never delivered.
        draft.text(
            f"Across the services above, {_terms_clause(ledger, owed, delivered, excused)}, which "
            "leaves no difference for this period, so I am not requesting compensatory services. I am "
            "asking the district to confirm these figures against its own service logs, and to tell me "
            "in writing if its records differ so that I can correct mine."
        )
    elif total == 0:
        draft.text(
            "For this period the records I hold reconcile with the minutes the IEP provides, so I am not "
            "requesting compensatory services. I am asking the district to confirm the figures against its "
            "own service logs, and to tell me in writing if its records differ so that I can correct mine."
        )
    elif not requesting:
        opening = (
            f"Across the services above, {_terms_clause(ledger, owed, delivered, excused)}, leaving an "
            f"arithmetic difference of {total} minutes. Every one of those minutes is a minute for "
            "which I hold no record either way."
            if excused
            else (
                f"The arithmetic difference for this period is {total} minutes, and every one of those "
                "minutes is a minute for which I hold no record either way."
            )
        )
        draft.text(
            f"{opening} That is a gap in my records, not a record of "
            "anything that did or did not happen at school, so I am not requesting compensatory services on "
            "the strength of it. I am asking the district for its service logs, session notes, and "
            "attendance records for these services for the period above (34 CFR 300.613(a)), so that the "
            "difference can be resolved against the district's own documents."
        )
    else:
        opening = (
            f"Across the services above, {_terms_clause(ledger, owed, delivered, excused)}, leaving an "
            f"arithmetic difference of {total} minutes."
            if excused
            else (
                f"The arithmetic difference for this period is {total} minutes between the minutes the IEP "
                "provides and the minutes documented as delivered."
            )
        )
        draft.text(
            f"{opening} I am not stating what that difference means "
            "or what amount of service would answer it. I am asking the IEP team to review it, to determine "
            "whether compensatory services are appropriate, and to record that determination in writing. "
            "34 CFR 300.151(b)(1)-(2) recognizes compensatory services as a remedy an SEA may order where "
            "it finds a failure to provide appropriate services; whether anything of the sort is warranted "
            "here is for the team, and then for the State, to decide, not for me to assert."
        )
        draft.text(
            "For the meeting, I ask the district to bring the service logs, session notes, attendance "
            "records, and progress-monitoring data for these services for the period above, together with "
            "any record of make-up sessions offered, scheduled, or held, and any documentation of an "
            "amendment to the IEP under 34 CFR 300.324(a)(4) or 34 CFR 300.324(a)(6) affecting these "
            "services."
        )

    if requesting:
        draft.text(
            "Please offer meeting dates within the next two weeks. If the district declines to convene the "
            "team or declines any part of this request, I ask for prior written notice under 34 CFR 300.503."
        )

    draft.text(_signature_block(ledger))

    return Letter(
        kind=LetterKind.COMPENSATORY_REQUEST,
        subject=subject,
        body=draft.body(),
        citations=draft.citations,
        legal_basis=[
            "34 CFR 300.323(c)(2)",
            "34 CFR 300.17(d)",
            "20 U.S.C. 1401(9)(D)",
            "34 CFR 300.320(a)(7)",
            "34 CFR 300.320(a)(3)(i)-(ii)",
            "34 CFR 300.324(a)(4)(i)",
            "34 CFR 300.324(a)(6)",
            "34 CFR 300.324(b)(1)(ii)(A)",
            "34 CFR 300.503(a)(1)-(2)",
            "34 CFR 300.151(b)(1)-(2)",
            "34 CFR 300.613(a)",
        ],
        disclaimer=DISCLAIMER,
    )


_DEADLINE_NOTES: dict[DeadlineKind, str] = {
    DeadlineKind.ANNUAL_REVIEW: (
        "For reference, 34 CFR 300.324(b)(1)(i) provides that the IEP Team reviews the IEP periodically, "
        "but not less than annually, and 34 CFR 300.323(a) provides that an IEP must be in effect for the "
        "child at the beginning of each school year."
    ),
    DeadlineKind.REEVALUATION: (
        "For reference, 34 CFR 300.303(b)(2) provides that a reevaluation must occur at least once every "
        "three years unless the parent and the public agency agree that a reevaluation is unnecessary."
    ),
    DeadlineKind.PROGRESS_REPORT: (
        "For reference, 34 CFR 300.320(a)(3) requires the IEP to state how progress toward the annual "
        "goals will be measured and when periodic reports will be provided; the schedule quoted above is "
        "the one this IEP sets."
    ),
}

_DEADLINE_LEGAL_BASIS: dict[DeadlineKind, list[str]] = {
    DeadlineKind.ANNUAL_REVIEW: [
        "34 CFR 300.324(b)(1)(i)",
        "20 U.S.C. 1414(d)(4)(A)(i)",
        "34 CFR 300.323(a)",
        "34 CFR 300.322(a)(1)-(2) and (b)(1)(i)",
        "34 CFR 300.503(a)(1)-(2)",
    ],
    DeadlineKind.REEVALUATION: [
        "34 CFR 300.303(b)(2)",
        "34 CFR 300.303(a)(1)-(2)",
        "20 U.S.C. 1414(a)(2)(B)",
        "34 CFR 300.503(a)(1)-(2)",
    ],
    DeadlineKind.PROGRESS_REPORT: [
        "34 CFR 300.320(a)(3)(i)-(ii)",
        "20 U.S.C. 1414(d)(1)(A)(i)(III)",
        "34 CFR 300.324(b)(1)(ii)(A)",
    ],
    DeadlineKind.OTHER: [],
}


def compile_deadline_reminder(
    ledger: IEPLedger, status: DeadlineStatus, *, today: date | None = None
) -> Letter:
    """Name a date the IEP itself set, and ask what is scheduled.

    A missed timeline is not, on its own, a finding about anything
    (34 CFR 300.513(a)(2) makes procedural misses actionable only through
    specific harm channels), so this letter quotes the date and asks a
    question. Asking in writing is also what creates the dated record.

    The letter counts days from *its own dateline*, never from
    ``status.days_remaining``. A DeadlineStatus is computed at some moment and
    a letter is compiled at another; restating a stale count as a fact about
    this letter's date is a false, school-adverse claim the letter cannot
    source. Where the status and the dateline disagree, the dateline wins,
    because the dateline is what the district will read.
    """
    today = today or date.today()
    deadline = status.deadline
    days_from_dateline = (deadline.due - today).days
    draft = _Draft()

    draft.text(
        f"Date: {today.isoformat()}\n"
        "To: Case Manager and Director of Special Education\n"
        f"Re: {deadline.description} — {ledger.student_alias}"
    )

    quoted = draft.cite(
        f'The IEP dated {ledger.iep_date.isoformat()} records the following: "{deadline.description}", '
        f"no later than {deadline.due.isoformat()}.",
        [_iep_ref(ledger, deadline.source_quote)],
    )

    upcoming_ask = (
        "I am asking the district to confirm in writing the date, time, and location, and who will "
        "attend, so that I can arrange to take part."
    )
    if status.state is DeadlineState.MET:
        follow_up = (
            f"Our file records this as met on {status.met_on.isoformat()}."
            if status.met_on is not None
            else "Our file records this as met."
        )
        ask = "No action is requested. This letter is a record for the file."
    elif days_from_dateline < 0:
        follow_up = (
            f"As of the date of this letter that date is "
            f"{_plural(abs(days_from_dateline), 'day')} in the past."
        )
        ask = (
            "I am asking the district to confirm in writing whether this has taken place and, if it has "
            "not, on what date it will be scheduled."
        )
    elif days_from_dateline == 0:
        follow_up = "That date is the date of this letter."
        ask = upcoming_ask
    else:
        follow_up = (
            f"As of the date of this letter that date is "
            f"{_plural(days_from_dateline, 'day')} away."
        )
        ask = upcoming_ask

    draft.paragraph(_parent_line(ledger), quoted, follow_up, ask)

    note = _DEADLINE_NOTES.get(deadline.kind)
    if note:
        draft.text(note)

    if status.state is not DeadlineState.MET and deadline.kind in (
        DeadlineKind.ANNUAL_REVIEW,
        DeadlineKind.REEVALUATION,
    ):
        draft.text(
            "If the district declines to convene the team or to conduct this, I ask for prior written "
            "notice under 34 CFR 300.503."
        )

    if status.state is DeadlineState.MET:
        draft.text(
            "This letter states a date recorded in the IEP and how our file stands against it. It makes no "
            "request and asserts nothing about the district."
        )
    else:
        draft.text(
            "This letter states a date recorded in the IEP and asks what is scheduled. It does not assert "
            "that any requirement has been broken."
        )

    draft.text(_signature_block(ledger))

    return Letter(
        kind=LetterKind.DEADLINE_REMINDER,
        subject=f"{deadline.description} — {ledger.student_alias}, due {deadline.due.isoformat()}",
        body=draft.body(),
        citations=draft.citations,
        legal_basis=list(_DEADLINE_LEGAL_BASIS.get(deadline.kind, [])),
        disclaimer=DISCLAIMER,
    )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def placeholders(letter: Letter) -> list[str]:
    """The blanks in ``letter`` that only the parent can fill.

    Minutes will not supply a fact about the family's circumstances, so a
    compiled letter can carry an instruction to the parent. Those are emitted
    with a sentinel rather than as ordinary prose so a sender can enumerate
    them mechanically; ``validate_letter(letter, for_sending=True)`` refuses a
    letter that still has one.
    """
    return [prompt.strip() for prompt in _PLACEHOLDER_RE.findall(letter.body)]


def _body_citation_violations(body: str) -> list[str]:
    """Regulation and reporter citations in the prose, checked against the table."""
    violations: list[str] = []
    for anchor in sorted(_regulation_anchors(body) - _ALLOWED_ANCHORS):
        violations.append(f"body cites {anchor}, which is outside the verified citation table")
    for reporter in sorted(_reporter_citations(body) - _ALLOWED_REPORTERS):
        violations.append(
            f"body cites {reporter}, which is outside the verified citation table"
        )
    return violations


def validate_letter(letter: Letter, *, for_sending: bool = False) -> list[str]:
    """Return every integrity violation in ``letter``; empty means it is sound.

    Checked, in order:

    1. A ``[n]`` marker in the body with no matching citation (a claim nobody
       can trace).
    2. A citation whose marker never appears in the body (a footnote with no
       claim).
    3. A marker defined by more than one citation.
    4. A ``legal_basis`` entry outside :data:`ALLOWED_CITATIONS`.
    5. A regulation or reporter cited in the body's own prose that is outside
       the verified table, after both are normalized for spelling — the check
       that makes an invented section number or a superseded reporter
       impossible anywhere in a letter, not only in ``legal_basis``.
    6. A parent-observed citation whose claim is not attributed as a parent
       observation.
    7. A legal conclusion, or an accusatory characterization, in the body.
    8. A missing disclaimer.

    ``for_sending=True`` adds the one check that is about a draft's readiness
    rather than its soundness: an unfilled parent placeholder. A compiled
    records request is sound (it invents nothing) but is not sendable until the
    parent has replaced the blank, and only the caller knows which of the two
    questions it is asking.
    """
    violations: list[str] = []

    body_markers = [f"[{n}]" for n in _MARKER_RE.findall(letter.body)]
    body_marker_set = set(body_markers)
    cited_markers = [c.marker for c in letter.citations]
    cited_marker_set = set(cited_markers)

    for marker in sorted(body_marker_set - cited_marker_set):
        violations.append(f"body carries marker {marker} with no matching citation")

    for marker in sorted(cited_marker_set - body_marker_set):
        violations.append(f"citation {marker} never appears in the body")

    for marker in sorted({m for m in cited_markers if cited_markers.count(m) > 1}):
        violations.append(f"marker {marker} is defined by more than one citation")

    for entry in letter.legal_basis:
        if entry not in ALLOWED_CITATIONS:
            violations.append(f"legal_basis entry is outside the verified citation table: {entry}")

    violations.extend(_body_citation_violations(letter.body))

    for citation in letter.citations:
        if citation.evidence.provenance is Provenance.PARENT_OBSERVED:
            claim = citation.claim.lower()
            if not any(phrase in claim for phrase in PARENT_ATTRIBUTION_PHRASES):
                violations.append(
                    f"citation {citation.marker} rests on a parent observation but its claim is not "
                    "attributed as one"
                )

    lowered_body = letter.body.lower()
    for phrase in LEGAL_CONCLUSION_PHRASES:
        if phrase in lowered_body:
            violations.append(f"body states a legal conclusion: {phrase!r}")

    for phrase in ACCUSATORY_PHRASES:
        if phrase in lowered_body:
            violations.append(f"body states an accusation the records do not support: {phrase!r}")

    if not letter.disclaimer.strip():
        violations.append("letter carries no disclaimer")

    if for_sending:
        for prompt in placeholders(letter):
            violations.append(f"letter contains an unfilled parent placeholder: {prompt}")

    return violations
