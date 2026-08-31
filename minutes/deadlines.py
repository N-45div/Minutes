"""The clocks.

A parent's leverage expires quietly. The annual review passes, the
triennial slips, the progress report that never arrived is forgotten by
the time the next one is due — and nobody is told, because the only party
tracking those dates is the party they bind. This module is the agent's
calendar: it evaluates every date the IEP obligates the school to act by
and reports each one's state, so Minutes can wake the parent *before* a
right lapses instead of documenting the loss afterwards.

Everything here is deterministic arithmetic over the ledger. No LLM call,
no network, no judgement call: a date either has passed or it has not.

Four rules this module holds to:

1. **It states facts about dates, never legal conclusions.** A missed
   deadline is not automatically a denial of FAPE — under 34 CFR
   300.513(a)(2) a procedural violation matters only if it impeded the
   child's right to FAPE, significantly impeded the parent's opportunity
   to participate, or caused a deprivation of educational benefit. That
   determination belongs to a hearing officer, not to this code. A
   ``DeadlineStatus`` says what date it is and whether the date passed.

2. **MET requires evidence.** A deadline is only marked ``MET`` when the
   caller supplies a date on which it was satisfied. Nothing here infers
   satisfaction from silence — the whole product exists because silence
   is not compliance.

3. **Silence is not evidence of failure either.** Rule 2 read backwards.
   ``OVERDUE`` says "the IEP states this date, the date has passed, and
   no evidence of satisfaction was supplied" — which is only honest for a
   date the IEP actually commits to. So the clocks run on the dates the
   IEP *states* and on nothing else: this module never manufactures a
   deadline out of a plan, a projection or an anniversary and then
   reports the absence of a record about it as a missed obligation. See
   :func:`derive_deadlines` for each candidate derivation and why it is
   refused.

4. **Lateness survives satisfaction.** A progress report delivered 29
   days late is ``MET`` — the obligation was discharged, and it should
   not interrupt anyone — but ``days_remaining`` stays negative and
   ``met_on`` records the delivery date. A pattern of chronically late
   reports is exactly the evidence this product exists to preserve, so a
   caller that filters ``MET`` statuses out wholesale throws that pattern
   away. Read ``days_remaining`` on ``MET`` statuses; ``met_on >
   deadline.due`` is a late delivery, and it is the only place that fact
   is recorded. (``DeadlineState`` is a frozen contract shared across
   engine modules and has no MET_LATE member, so this is documented here
   rather than encoded.)
"""

from __future__ import annotations

from datetime import date

from .models import (
    Deadline,
    DeadlineKind,
    DeadlineState,
    DeadlineStatus,
    IEPLedger,
    Urgency,
)

# How far ahead a deadline starts asking for the parent's attention.
#
# The default is 30 days. The per-kind overrides below exist because the
# researched IDEA timelines make the required lead time wildly unequal —
# a warning that arrives too late to act on is the same as no warning.
DUE_SOON_DAYS = 30

DUE_SOON_DAYS_BY_KIND: dict[DeadlineKind, int] = {
    # An annual review is a *meeting*. 34 CFR 300.322(a)(1)-(2) requires
    # the agency to notify parents early enough to ensure an opportunity
    # to attend and to schedule at a mutually agreed time and place, and
    # a parent who wants the service records in hand first is working
    # against the 45-day ceiling in 34 CFR 300.613(a). Sixty days is the
    # smallest window in which a parent can both request records and get
    # a meeting date agreed.
    DeadlineKind.ANNUAL_REVIEW: 60,
    # A reevaluation is an assessment cycle, not a date. Informed
    # parental consent must be obtained first (34 CFR 300.300(c)(1)(i)),
    # then the assessments are conducted, then the team meets. No federal
    # regulation sets a completion timeline for a *re*evaluation — 34 CFR
    # 300.303 states when one is required, not how long it may take — so
    # 90 days is this product's lead-time choice, not a legal deadline:
    # a quarter is the smallest span that fits consent, assessment and
    # the meeting that follows.
    DeadlineKind.REEVALUATION: 90,
    # A progress report requires nothing of the parent in advance; it
    # either arrives or it does not. The cadence is whatever this child's
    # own IEP states — 34 CFR 300.320(a)(3)(ii) sets no federal minimum
    # frequency — and on a quarterly cadence a 30-day window would keep
    # a report flagged for a third of the year, which is noise. Two weeks
    # is enough to notice a report has not come.
    DeadlineKind.PROGRESS_REPORT: 14,
    DeadlineKind.OTHER: DUE_SOON_DAYS,
}

if set(DUE_SOON_DAYS_BY_KIND) != set(DeadlineKind):  # pragma: no cover - import guard
    raise RuntimeError(
        "every DeadlineKind needs an explicit due-soon window; missing: "
        f"{sorted(k.value for k in set(DeadlineKind) - set(DUE_SOON_DAYS_BY_KIND))}"
    )

# Inside this window a due-soon deadline stops being a heads-up and
# becomes the thing the parent has to deal with this week.
IMMINENT_DAYS = 7


def deadline_key(deadline: Deadline) -> str:
    """The unambiguous ``met`` key for one deadline.

    Returns ``kind:YYYY-MM-DD:description`` — all three fields, because
    the first two are not unique. An IEP can state two obligations of the
    same kind on the same date (four services projected to begin the same
    Monday, two "other" commitments due the same Friday), and a key that
    named only kind and date would resolve to whichever sorted first,
    silently crediting one obligation with another's evidence. That is a
    fabricated compliance credit on one deadline and a false accusation
    on the other, from a single call the docstrings recommend, so the key
    carries the description that tells them apart.

    Callers building a ``met`` mapping should use this rather than
    hand-assembling keys. See :func:`evaluate_deadlines` for the three
    shorter forms that are also accepted, and for what happens when one
    of them is ambiguous (it raises; it never guesses).
    """
    return f"{deadline.kind.value}:{deadline.due.isoformat()}:{deadline.description}"


def derive_deadlines(ledger: IEPLedger) -> list[Deadline]:
    """The set of dates the clocks run on: the ones the IEP states.

    **Deadlines the IEP states are returned unmodified**, in a canonical
    order (due date, then kind, then description) so the working set is
    deterministic across runs. The extracted ledger already carries the
    annual-review date, the triennial date and the progress-report dates
    as the IEP itself writes them, each with its verbatim
    ``source_quote``, and a stated date is better evidence than a
    computed one. Nothing here overrides or recomputes them.

    **Nothing is derived.** Every candidate derivation was examined
    against the verified research and refused, because each one would
    either invent a fact about a real child or convert an absence of
    records into an accusation:

    * *An annual-review anniversary.* No federal regulation sets a
      numeric "within 365 days of the last IEP meeting" rule. The federal
      obligations are "not less than annually" (34 CFR 300.324(b)(1)(i))
      and "in effect at the beginning of each school year" (34 CFR
      300.323(a)). The 12-month computation is a state-level convention,
      so computing one from ``iep_date`` would present an implementation
      convention as federal law.
    * *A triennial reevaluation date.* 34 CFR 300.303(b)(2) runs three
      years from the last evaluation, and the ledger does not carry an
      evaluation date. Deriving one from ``iep_date`` would be inventing
      a fact about a real child.
    * *A service's projected start date.* 34 CFR 300.320(a)(7) requires
      the IEP to state the *projected* date for the beginning of each
      service. A projected date is a plan, and no federal regulation
      converts it into a date the school breaches the day after. Clocking
      it as a deadline would mean that every service the parent simply
      has not reported on turns ``OVERDUE`` on day+1 and stays
      permanently imminent — an accusation manufactured out of missing
      input, which is precisely rule 3 in this module's docstring.
      Non-delivery is measured where the evidence lives: reconciliation
      accrues promised minutes from ``ServiceObligation.start_date`` and
      reports the portion with no record either way as *undocumented*
      rather than as a breach, which is the honest rendering of silence.

    Kept as the module's single entry point for the working set — so
    :func:`evaluate_deadlines` has one place to draw from, and so the
    reasoning above sits where the next contributor tempted to compute a
    deadline will read it.
    """
    return sorted(ledger.deadlines, key=lambda d: (d.due, d.kind.value, d.description))


def evaluate_deadlines(
    ledger: IEPLedger,
    today: date,
    met: dict[str, date] | None = None,
) -> list[DeadlineStatus]:
    """Evaluate every deadline the IEP states against ``today``.

    Runs over :func:`derive_deadlines`, so results come back in due-date
    order (ties broken by kind then description) and the sequence is
    stable across runs.

    States:

    * ``MET`` — ``met`` supplies a date it was satisfied on. Only
      evidence produces this state; nothing is inferred. ``MET`` does not
      mean "on time": ``days_remaining`` stays negative and ``met_on``
      keeps the delivery date for a late satisfaction (see rule 4 in the
      module docstring — do not filter ``MET`` statuses out wholesale, or
      a pattern of chronic lateness disappears).
    * ``OVERDUE`` — ``today`` is past ``due`` and no satisfaction date
      was supplied. Being past due is a fact about a calendar, not a
      finding of violation.
    * ``DUE_SOON`` — due within this kind's lead-time window, inclusive
      at both ends: the due date itself is ``DUE_SOON``, and so is a
      deadline exactly ``DUE_SOON_DAYS_BY_KIND[kind]`` days out.
    * ``UPCOMING`` — further out than that.

    ``days_remaining`` is always ``due - today``, negative once the date
    has passed, and is reported for met deadlines too so a statement can
    show how close to the wire a satisfied obligation ran.

    **``met`` key scheme.** Four forms are accepted, and each key marks
    exactly one deadline:

    1. ``"kind:YYYY-MM-DD:description"`` — what :func:`deadline_key`
       produces, and the only form guaranteed unique. Use it whenever the
       caller knows which obligation was satisfied.
    2. ``"kind:YYYY-MM-DD"`` (e.g. ``"progress_report:2026-11-06"``) —
       the same thing without the description. Convenient, but an IEP may
       state two deadlines of one kind on one date; when this form
       matches more than one, it raises ``ValueError`` naming the
       collision rather than picking one.
    3. The deadline's ``description``, verbatim.
    4. A bare :class:`DeadlineKind` value (e.g. ``"annual_review"``).

    Forms 3 and 4 can match several deadlines — an IEP typically states
    four progress-report dates with one shared description — so the
    satisfaction date resolves them: the key marks the latest deadline
    already due on or before that date, or, if none was yet due, the
    earliest one after it (an obligation met early). One date is evidence
    of one satisfaction, so a single key never marks a run of deadlines
    met. Pass a form-1 key per deadline to mark several.

    Three things raise ``ValueError`` instead of being absorbed, all for
    the same reason: a caller who believes something was satisfied and is
    silently ignored ends up with a statement that under-reports the
    school's compliance, and a wrong ledger is worse than a loud failure.

    * A key that matches **no** deadline.
    * Two keys that resolve to the **same** deadline — one of the two
      satisfaction dates would otherwise be dropped on the floor.
    * A satisfaction date that is **not plausible**: after ``today``
      (nothing has been satisfied in the future) or before
      ``ledger.iep_date`` (nothing discharged an obligation that did not
      yet exist). Both are caller bugs a downstream Statement cannot
      detect on its own.
    """
    deadlines = derive_deadlines(ledger)
    met_by_index = _resolve_met(deadlines, met or {})
    _check_met_dates(deadlines, met_by_index, today, ledger.iep_date)

    statuses: list[DeadlineStatus] = []
    for index, deadline in enumerate(deadlines):
        days_remaining = (deadline.due - today).days
        met_on = met_by_index.get(index)

        if met_on is not None:
            state = DeadlineState.MET
        elif days_remaining < 0:
            state = DeadlineState.OVERDUE
        elif days_remaining <= due_soon_window(deadline.kind):
            state = DeadlineState.DUE_SOON
        else:
            state = DeadlineState.UPCOMING

        statuses.append(
            DeadlineStatus(
                deadline=deadline,
                state=state,
                days_remaining=days_remaining,
                met_on=met_on,
            )
        )
    return statuses


def urgency_for(status: DeadlineStatus) -> Urgency:
    """Map a deadline's state onto the urgency of interrupting the parent.

    ``OVERDUE`` is ``DEADLINE_IMMINENT``: a stated date that has passed
    unmet is the case where every further day of silence costs
    something — the state-complaint window in 34 CFR 300.153(c) is one
    year from the violation and runs whether or not anyone is watching.
    A ``DUE_SOON`` deadline inside :data:`IMMINENT_DAYS` is treated the
    same, because at that range there is no longer time to schedule
    anything. Met and upcoming deadlines are routine — they belong in the
    monthly Statement, not in an interruption; a satisfied obligation
    never interrupts anyone, however late it arrived.
    """
    if status.state is DeadlineState.OVERDUE:
        return Urgency.DEADLINE_IMMINENT
    if status.state is DeadlineState.DUE_SOON:
        if status.days_remaining <= IMMINENT_DAYS:
            return Urgency.DEADLINE_IMMINENT
        return Urgency.TIME_SENSITIVE
    return Urgency.ROUTINE


def next_action_date(statuses: list[DeadlineStatus]) -> date | None:
    """The soonest date still requiring attention, or ``None``.

    The earliest due date among deadlines that are not ``MET``. Met
    deadlines are done; every other state is outstanding, overdue ones
    included — so a returned date in the past is the signal that the
    parent is already late, not a bug. ``None`` means the ledger holds
    nothing outstanding.
    """
    outstanding = [s.deadline.due for s in statuses if s.state is not DeadlineState.MET]
    return min(outstanding) if outstanding else None


def due_soon_window(kind: DeadlineKind) -> int:
    """Lead-time window in days for one kind of deadline.

    A direct lookup with no fallback: a ``DeadlineKind`` with no window
    of its own is a gap in the reasoning above, and it should fail
    loudly rather than quietly inherit 30 days. The module-level guard
    beside :data:`DUE_SOON_DAYS_BY_KIND` catches that at import.
    """
    return DUE_SOON_DAYS_BY_KIND[kind]


# ---------------------------------------------------------------------------
# ``met`` key resolution
# ---------------------------------------------------------------------------


def _parse_dated_key(key: str) -> tuple[DeadlineKind, date, str | None] | None:
    """Parse ``kind:YYYY-MM-DD[:description]``, or ``None`` if not that form.

    The description is everything after the second colon, so a
    description that itself contains colons round-trips through
    :func:`deadline_key` unharmed.
    """
    kind_part, separator, rest = key.partition(":")
    if not separator:
        return None
    try:
        kind = DeadlineKind(kind_part)
    except ValueError:
        return None
    due_part, separator, description = rest.partition(":")
    try:
        due = date.fromisoformat(due_part)
    except ValueError:
        return None
    return kind, due, (description if separator else None)


def _candidates(key: str, deadlines: list[Deadline]) -> list[int]:
    """Indices a key could refer to; empty when it matches nothing.

    The two dated forms are read first and return at most one index — a
    dated key matching several deadlines raises, so an ambiguous key can
    never reach :func:`_closest`, which exists only for the two forms
    this module documents as ambiguous. A dated key that matches nothing
    falls through, so a description shaped like ``kind:date`` still finds
    its deadline. Description is checked before bare kind so a
    description that happens to read like a kind value still resolves to
    the deadline that carries it.
    """
    parsed = _parse_dated_key(key)
    if parsed is not None:
        kind, due, description = parsed
        indices = [
            i
            for i, d in enumerate(deadlines)
            if d.kind is kind and d.due == due and (description is None or d.description == description)
        ]
        if len(indices) > 1:
            collisions = ", ".join(repr(deadlines[i].description) for i in indices)
            raise ValueError(
                f"met key {key!r} is ambiguous: it matches {len(indices)} deadlines "
                f"({collisions}); use deadline_key(deadline) to name one exactly"
            )
        if indices:
            return indices

    by_description = [i for i, d in enumerate(deadlines) if d.description == key]
    if by_description:
        return by_description

    try:
        kind = DeadlineKind(key)
    except ValueError:
        return []
    return [i for i, d in enumerate(deadlines) if d.kind is kind]


def _closest(deadlines: list[Deadline], indices: list[int], met_on: date) -> int:
    """The one deadline a satisfaction date most plausibly discharges.

    The latest deadline already due on or before ``met_on``; failing
    that, the earliest one still ahead of it, which is an obligation met
    early. Ties resolve to the first index, and the caller has already
    sorted the deadlines, so this is deterministic. Only reached by the
    description and bare-kind forms; the dated forms name one deadline or
    raise.
    """
    already_due = [i for i in indices if deadlines[i].due <= met_on]
    if already_due:
        return max(already_due, key=lambda i: (deadlines[i].due, -i))
    return min(indices, key=lambda i: (deadlines[i].due, i))


def _resolve_met(deadlines: list[Deadline], met: dict[str, date]) -> dict[int, date]:
    """Resolve the caller's ``met`` mapping onto deadline indices.

    Each key resolves independently of the others, and keys are processed
    in sorted order, so neither the outcome nor the wording of an error
    depends on the caller's dict insertion order. Two keys that land on
    the same deadline raise: one of the two satisfaction dates would
    otherwise have to be discarded, and discarding it silently is the
    failure mode this module refuses everywhere else.
    """
    resolved: dict[int, date] = {}
    claimed_by: dict[int, str] = {}

    for key in sorted(met):
        indices = _candidates(key, deadlines)
        if not indices:
            raise ValueError(
                f"met key {key!r} matches no deadline in this ledger; "
                "use deadline_key(deadline), the deadline's description, "
                "or a DeadlineKind value"
            )
        index = indices[0] if len(indices) == 1 else _closest(deadlines, indices, met[key])
        if index in claimed_by:
            raise ValueError(
                f"met keys {claimed_by[index]!r} and {key!r} both resolve to the same "
                f"deadline ({deadlines[index].description!r} due "
                f"{deadlines[index].due.isoformat()}); one satisfaction date would be "
                "discarded — use deadline_key(deadline) to name each deadline exactly"
            )
        claimed_by[index] = key
        resolved[index] = met[key]

    return resolved


def _check_met_dates(
    deadlines: list[Deadline],
    met_by_index: dict[int, date],
    today: date,
    iep_date: date,
) -> None:
    """Reject satisfaction dates that cannot be true.

    A date after ``today`` claims something has already happened that has
    not, and a date before the IEP was written claims an obligation was
    discharged before it existed. Either one silently retires a live
    deadline, and nothing downstream can tell that it was wrong.
    """
    for index in sorted(met_by_index):
        met_on = met_by_index[index]
        deadline = deadlines[index]
        if met_on > today:
            raise ValueError(
                f"met date {met_on.isoformat()} for {deadline.description!r} is in the "
                f"future relative to today ({today.isoformat()}); a deadline is marked "
                "met by evidence it was satisfied, not by an intention to satisfy it"
            )
        if met_on < iep_date:
            raise ValueError(
                f"met date {met_on.isoformat()} for {deadline.description!r} predates "
                f"this IEP ({iep_date.isoformat()}); it cannot have discharged an "
                "obligation that did not yet exist"
            )
