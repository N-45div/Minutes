"""Ledger model tests — pure unit tests plus cached-fixture validation.

No test in this file makes a network call.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from minutes.models import (
    IEPLedger,
    Period,
    ServiceObligation,
)

CACHE = Path(__file__).resolve().parents[1] / "fixtures" / "cache" / "iep_maya_ledger.json"


def _obligation(**overrides) -> ServiceObligation:
    base = dict(
        service="Speech-Language Therapy",
        minutes_per_session=30,
        sessions_per_period=2,
        period=Period.WEEK,
        provider_role="Licensed Speech-Language Pathologist",
        setting="therapy room",
        start_date=date(2026, 9, 8),
        end_date=date(2027, 6, 11),
        source_quote="30 minutes per session, 2 sessions per week",
    )
    base.update(overrides)
    return ServiceObligation(**base)


def test_weekly_minutes_for_weekly_service():
    assert _obligation().minutes_per_week == 60


def test_weekly_minutes_for_daily_service():
    daily = _obligation(minutes_per_session=60, sessions_per_period=1, period=Period.DAY)
    assert daily.minutes_per_week == 300


def test_weekly_minutes_for_monthly_service():
    monthly = _obligation(minutes_per_session=30, sessions_per_period=1, period=Period.MONTH)
    assert monthly.minutes_per_week == pytest.approx(7.5)


def test_rejects_zero_minute_obligation():
    with pytest.raises(ValueError):
        _obligation(minutes_per_session=0)


@pytest.mark.skipif(not CACHE.exists(), reason="run scripts/extract_once.py to build the cache")
def test_cached_ledger_is_valid_and_complete():
    ledger = IEPLedger.model_validate(json.loads(CACHE.read_text(encoding="utf-8")))

    services = {o.service.lower() for o in ledger.obligations}
    assert any("speech" in s for s in services)
    assert any("occupational" in s for s in services)
    assert len(ledger.obligations) == 4
    assert len(ledger.deadlines) >= 2

    # Every extracted fact must carry its verbatim citation.
    assert all(o.source_quote.strip() for o in ledger.obligations)
    assert all(d.source_quote.strip() for d in ledger.deadlines)

    # 60 (speech) + 45 (OT) + 300 (daily SAI) + ~7.5 (monthly counseling)
    assert ledger.promised_minutes_per_week == pytest.approx(412.5, abs=1.0)
