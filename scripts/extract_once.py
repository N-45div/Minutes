"""Run IEP extraction ONCE and cache the ledger as a fixture.

Everything downstream develops against the cached JSON at zero token
cost. Re-run only when the fixture IEP or the extraction prompt changes.

Usage: python scripts/extract_once.py [fixture_name]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minutes.extraction import extract_ledger  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "iep_maya"
    iep_text = (FIXTURES / f"{name}.md").read_text(encoding="utf-8")

    ledger = extract_ledger(iep_text)

    cache = FIXTURES / "cache"
    cache.mkdir(exist_ok=True)
    out = cache / f"{name}_ledger.json"
    out.write_text(ledger.model_dump_json(indent=2), encoding="utf-8")

    print(f"cached -> {out}")
    print(f"student: {ledger.student_alias} | {ledger.school_year}")
    print(f"obligations: {len(ledger.obligations)}")
    for o in ledger.obligations:
        print(f"  - {o.service}: {o.minutes_per_session} min x {o.sessions_per_period}/{o.period.value} = {o.minutes_per_week:.0f} min/week")
    print(f"deadlines: {len(ledger.deadlines)}")
    for d in ledger.deadlines:
        print(f"  - {d.kind.value}: {d.due} ({d.description[:60]})")
    print(f"accommodations: {len(ledger.accommodations)}")
    print(f"promised minutes/week: {ledger.promised_minutes_per_week:.0f}")


if __name__ == "__main__":
    main()
