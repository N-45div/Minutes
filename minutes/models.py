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
