"""The typed obligations ledger.

An IEP is extracted once into these models; everything downstream —
reconciliation, deadline clocks, letters, the monthly Statement — is
deterministic code over this ledger. No generated text may assert a
fact that is not present here (the cite-or-stay-silent rule), which is
why every extracted item carries the verbatim ``source_quote`` it came
from.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field

# Approximate school-calendar factors used only for weekly summaries.
# Reconciliation itself operates on scheduled sessions, not these factors.
_WEEKS_PER_PERIOD = {
    "day": 1 / 5,  # a daily service delivers 5 sessions per school week
    "week": 1.0,
    "month": 4.0,
    "quarter": 9.0,
    "year": 36.0,
}


class Period(str, Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class ServiceObligation(BaseModel):
    """One promised service line from the IEP, normalized to minutes."""

    service: str = Field(description="Service name as written in the IEP, e.g. 'Speech-Language Therapy'")
    minutes_per_session: int = Field(gt=0)
    sessions_per_period: int = Field(gt=0)
    period: Period = Field(description="The period the session count applies to")
    provider_role: str = Field(description="Who delivers it, e.g. 'Licensed Speech-Language Pathologist'")
    setting: str = Field(description="Where it is delivered, e.g. 'therapy room', 'general education classroom'")
    start_date: date
    end_date: date
    source_quote: str = Field(description="Verbatim sentence(s) from the IEP this obligation comes from")

    @property
    def minutes_per_week(self) -> float:
        """Promised minutes per school week (calendar approximation)."""
        weeks = _WEEKS_PER_PERIOD[self.period.value]
        return self.minutes_per_session * self.sessions_per_period / weeks


class DeadlineKind(str, Enum):
    ANNUAL_REVIEW = "annual_review"
    REEVALUATION = "reevaluation"
    PROGRESS_REPORT = "progress_report"
    OTHER = "other"


class Deadline(BaseModel):
    """A date the school is obligated to act by."""

    kind: DeadlineKind
    due: date
    description: str = Field(description="What must happen by this date")
    source_quote: str = Field(description="Verbatim sentence(s) from the IEP this deadline comes from")


class Provenance(str, Enum):
    """Evidence grade of an observed fact. Escalation letters may cite
    SCHOOL_CONFIRMED and DOCUMENTED_SILENCE facts without qualification;
    PARENT_OBSERVED facts are always attributed as parent observations."""

    SCHOOL_CONFIRMED = "school_confirmed"
    PARENT_OBSERVED = "parent_observed"
    DOCUMENTED_SILENCE = "documented_silence"


class ServiceEvent(BaseModel):
    """One observed delivery or non-delivery fact."""

    event_date: date
    service: str
    minutes: int = Field(ge=0, description="Minutes delivered; 0 for a missed session")
    delivered: bool
    provenance: Provenance
    source: str = Field(description="Where this fact came from, e.g. an email id, a parent log entry, a records-request id")


class Accommodation(BaseModel):
    description: str
    source_quote: str


class IEPLedger(BaseModel):
    """Everything the IEP promises, machine-readable."""

    student_alias: str = Field(description="Pseudonymous student identifier; never the real full name")
    school_year: str
    iep_date: date
    obligations: list[ServiceObligation]
    deadlines: list[Deadline]
    accommodations: list[Accommodation]

    @property
    def promised_minutes_per_week(self) -> float:
        return sum(o.minutes_per_week for o in self.obligations)


# ---------------------------------------------------------------------------
# Evidence, correspondence, and the artifacts compiled from them.
#
# Everything below is the frozen contract between engine modules. Each module
# consumes and returns these types and nothing else, so no module needs to
# know how any other module is implemented.
# ---------------------------------------------------------------------------


class EvidenceRef(BaseModel):
    """A pointer to why a claim is believed. No compiled document may assert
    a fact that lacks one of these."""

    provenance: Provenance
    source: str = Field(description="Identifier of the originating record, e.g. 'email-2026-10-14-01', 'req-002'")
    detail: str = Field(description="The dated fact or verbatim quote this reference stands on")


class CorrespondenceKind(str, Enum):
    SCHOOL_EMAIL = "school_email"
    PARENT_LOG = "parent_log"
    PROGRESS_REPORT = "progress_report"
    SERVICE_LOG = "service_log"


class Correspondence(BaseModel):
    """One inbound item: a school message, a returned service log, or a
    parent's quick note."""

    item_id: str
    received: date
    kind: CorrespondenceKind
    sender: str
    subject: str
    body: str


class ServiceShortfall(BaseModel):
    """Owed minus delivered for one service over one period."""

    service: str
    period_start: date
    period_end: date
    owed_minutes: int = Field(ge=0)
    delivered_minutes: int = Field(ge=0)
    shortfall_minutes: int = Field(ge=0)
    school_confirmed_minutes: int = Field(ge=0)
    parent_observed_minutes: int = Field(ge=0)
    undocumented_minutes: int = Field(ge=0, description="Owed minutes with no evidence either way")
    evidence: list[EvidenceRef]


class ReconciliationResult(BaseModel):
    period_start: date
    period_end: date
    shortfalls: list[ServiceShortfall]
    events_considered: int = Field(ge=0)

    @property
    def total_shortfall_minutes(self) -> int:
        return sum(s.shortfall_minutes for s in self.shortfalls)


class DeadlineState(str, Enum):
    UPCOMING = "upcoming"
    DUE_SOON = "due_soon"
    OVERDUE = "overdue"
    MET = "met"


class DeadlineStatus(BaseModel):
    deadline: Deadline
    state: DeadlineState
    days_remaining: int = Field(description="Negative once the due date has passed")
    met_on: date | None = None


class RequestState(str, Enum):
    DRAFT = "draft"
    SENT = "sent"
    ANSWERED = "answered"
    UNANSWERED_OVERDUE = "unanswered_overdue"


class RecordsRequest(BaseModel):
    """A request for the school's own service-delivery records. An overdue,
    unanswered request is itself evidence."""

    request_id: str
    covers_start: date
    covers_end: date
    services: list[str]
    state: RequestState = RequestState.DRAFT
    sent_on: date | None = None
    response_due: date | None = None
    answered_on: date | None = None


class LetterKind(str, Enum):
    RECORDS_REQUEST = "records_request"
    SHORTFALL_NOTICE = "shortfall_notice"
    COMPENSATORY_REQUEST = "compensatory_request"
    DEADLINE_REMINDER = "deadline_reminder"


class LetterCitation(BaseModel):
    marker: str = Field(description="Footnote marker as it appears in the body, e.g. '[1]'")
    claim: str = Field(description="The sentence in the body this marker supports")
    evidence: EvidenceRef


class Letter(BaseModel):
    """A compiled document. Assembled from ledger facts and evidence, never
    free-written: every factual claim in ``body`` carries a marker present in
    ``citations`` (the cite-or-stay-silent rule)."""

    kind: LetterKind
    subject: str
    body: str
    citations: list[LetterCitation]
    legal_basis: list[str] = Field(default_factory=list, description="Regulatory references, e.g. '34 CFR 300.323(a)'")
    disclaimer: str = "This letter is documentation compiled from your child's IEP and your records. It is not legal advice."


class Urgency(str, Enum):
    ROUTINE = "routine"
    TIME_SENSITIVE = "time_sensitive"
    DEADLINE_IMMINENT = "deadline_imminent"


class DecisionCard(BaseModel):
    """The only thing that ever interrupts the parent."""

    card_id: str
    title: str
    why_now: str
    facts: list[str]
    recommended_action: str
    urgency: Urgency
    deadline: date | None = None
    draft: Letter | None = None


class StatementLine(BaseModel):
    service: str
    owed_minutes: int = Field(ge=0)
    delivered_minutes: int = Field(ge=0)
    shortfall_minutes: int = Field(ge=0)
    school_confirmed_minutes: int = Field(ge=0)
    parent_observed_minutes: int = Field(ge=0)
    undocumented_minutes: int = Field(ge=0)


class Statement(BaseModel):
    """The monthly artifact: owed, delivered, shortfall, evidence, what's next."""

    student_alias: str
    period_start: date
    period_end: date
    lines: list[StatementLine]
    open_deadlines: list[DeadlineStatus]
    unanswered_requests: list[RecordsRequest]
    decisions: list[DecisionCard]

    @property
    def total_owed(self) -> int:
        return sum(line.owed_minutes for line in self.lines)

    @property
    def total_delivered(self) -> int:
        return sum(line.delivered_minutes for line in self.lines)

    @property
    def total_shortfall(self) -> int:
        return sum(line.shortfall_minutes for line in self.lines)


class AuditEntry(BaseModel):
    """Every action the agent takes, recorded as it happens. The paper trail
    is the product."""

    entry_date: date
    actor: str = Field(description="Which agent or module acted")
    action: str
    detail: str
    evidence: EvidenceRef | None = None
