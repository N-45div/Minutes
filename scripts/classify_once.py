"""Run correspondence classification ONCE and cache the events as a fixture.

Reconciliation, the Statement and every letter develop against the cached
JSON at zero token cost.

Two modes:

    python scripts/classify_once.py [fixture_name]
        The real thing. Reads the semester with the classifier
        (CLASSIFIER_MODEL, CLASSIFIER_PASSES passes) and writes the cache.
        This costs tokens and is run once.

    python scripts/classify_once.py [fixture_name] --regroom
        No LLM, no network. Reads the cache back, turns every cached fact
        into the draft that produced it, and re-runs the deterministic tail
        of the pipeline over it.

``--regroom`` exists because the two halves of this module fail differently.
The model readings are expensive and frozen; the guards around them —
grounding, admissibility, provenance, duration — are free, deterministic, and
the part that gets fixed when a defect is found. Regrooming re-applies the
current guards to the readings already paid for, so a tightened guard reaches
the shipped artifact instead of only the source. What it cannot do is recover
a fact the model never emitted, or re-run the cross-pass vote: each cached
fact is one already-established reading, so the quorum is 1 on this path. A
regroomed cache can only ever be the same size or smaller than the run it
came from.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minutes.correspondence import (  # noqa: E402
    EventDraft,
    classify_to_events,
    load_cached_events,
    load_correspondence,
    student_absence_events,
    _drafts_to_events,
)
from minutes.models import IEPLedger, ServiceEvent  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def regroom(name: str, items, ledger: IEPLedger) -> list[ServiceEvent]:
    """Re-apply the deterministic guards to the readings already cached."""
    drafts = [
        EventDraft(
            item_id=event.source,
            event_date=event.event_date,
            service=event.service,
            delivered=event.delivered,
            minutes=event.minutes,
        )
        for event in load_cached_events(name)
    ]
    return _drafts_to_events(drafts, items, ledger, min_readings=1)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name = args[0] if args else "maya_fall_2026"
    regrooming = "--regroom" in sys.argv[1:]

    ledger_path = FIXTURES / "cache" / "iep_maya_ledger.json"
    if not ledger_path.exists():
        raise SystemExit(f"{ledger_path} missing; run scripts/extract_once.py first")

    ledger = IEPLedger.model_validate_json(ledger_path.read_text(encoding="utf-8"))
    items = load_correspondence(name)

    if regrooming:
        before = len(load_cached_events(name))
        events = regroom(name, items, ledger)
        print(f"regroomed (no LLM): {before} cached facts -> {len(events)} after the current guards")
    else:
        events = classify_to_events(items, ledger)

    cache = FIXTURES / "cache"
    cache.mkdir(exist_ok=True)
    out = cache / f"{name}_events.json"
    out.write_text(
        json.dumps([e.model_dump(mode="json") for e in events], indent=2),
        encoding="utf-8",
    )

    print(f"cached -> {out}")
    print(f"items classified: {len(items)} | events: {len(events)}")
    print(f"items yielding no event: {len(items) - len({e.source for e in events})}")

    by_provenance = Counter(e.provenance.value for e in events)
    for provenance, count in sorted(by_provenance.items()):
        print(f"  {provenance}: {count}")

    for obligation in ledger.obligations:
        rows = [e for e in events if e.service == obligation.service]
        delivered = [e for e in rows if e.delivered]
        print(
            f"  - {obligation.service}: {len(delivered)} delivered / "
            f"{len(rows) - len(delivered)} missed, "
            f"{sum(e.minutes for e in delivered)} minutes documented"
        )

    # The handoff reconciliation and the letter compiler have to act on: these
    # misses are the child's absence, not the school's failure, and nothing in
    # ServiceEvent says so.
    excusable = student_absence_events(events, items)
    print(f"\nnon-deliveries attributed to student absence: {len(excusable)}")
    for event in excusable:
        print(f"  - {event.event_date} {event.service} ({event.source}, {event.provenance.value})")
    print("  ^ letters must qualify or exclude these; ServiceEvent carries no fault field.")


if __name__ == "__main__":
    main()
