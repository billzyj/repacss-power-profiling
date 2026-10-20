"""Tests for the profile-driven RAPL collector."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from inband.collectors import rapl
from inband.collectors.rapl import RAPLCollector, counter_delta_uj, wrap_safe_seconds


def _fixed_clock(monkeypatch, readings):
    clock = iter(readings)
    monkeypatch.setattr(rapl, "time", SimpleNamespace(monotonic=lambda: next(clock)))


def test_counter_delta_without_and_with_a_wrap():
    assert counter_delta_uj(1_000, 4_000, 10_000) == (3_000, False)
    assert counter_delta_uj(9_000, 500, 10_000) == (1_500, True)


def test_wrap_bound_for_the_audited_counter_ranges():
    assert wrap_safe_seconds(65_532_610_987) == pytest.approx(65.53, abs=0.01)
    assert wrap_safe_seconds(262_143_328_850) == pytest.approx(262.14, abs=0.01)


def test_probe_reports_the_skip_reason_for_an_unverified_node(tmp_path: Path):
    collector = RAPLCollector(partition="h100", node="rpc-91-1", sysfs_root=tmp_path)
    probe = collector.probe()
    assert probe.available is False
    assert probe.reason == "skipped: unverified_node"
    assert collector.manifest_payload()["reason"] == "unverified_node"


def test_rows_are_keyed_by_domain_path(fake_powercap, monkeypatch):
    root = fake_powercap("h100_host_rapl")
    collector = RAPLCollector(partition="h100", node="rpg-93-1", sysfs_root=root)
    assert collector.probe().available
    _fixed_clock(monkeypatch, [10.0] * 5 + [11.0] * 5)

    runtime = collector.prepare_runtime()
    first = collector.collect_rows(runtime, 0.0)
    assert len(first) == 5
    assert all(row["delta_uj"] is None for row in first)

    # Both DRAM domains are named "dram"; their deltas must stay separate.
    (root / "intel-rapl:0/intel-rapl:0:0/energy_uj").write_text("3000000\n", encoding="utf-8")
    (root / "intel-rapl:2/intel-rapl:2:0/energy_uj").write_text("1500000\n", encoding="utf-8")
    (root / "intel-rapl:0/energy_uj").write_text("500\n", encoding="utf-8")
    second = {row["domain_path"]: row for row in collector.collect_rows(runtime, 1.0)}

    assert second["intel-rapl:0/intel-rapl:0:0"]["delta_uj"] == 2_000_000
    assert second["intel-rapl:0/intel-rapl:0:0"]["power_watts"] == 2.0
    assert second["intel-rapl:2/intel-rapl:2:0"]["delta_uj"] == 500_000
    assert second["intel-rapl:0"]["wrapped"] is True
    assert second["intel-rapl:0"]["delta_uj"] == 262_143_328_850 - 1_000_000 + 500
    assert second["intel-rapl:1"]["domain_name"] == "psys"
    assert second["intel-rapl:1"]["domain_group"] == "platform"
    assert not any(row["wrap_ambiguous"] for row in second.values())


def test_a_gap_beyond_the_wrap_bound_is_flagged(fake_powercap, monkeypatch):
    root = fake_powercap("zen4_rapl")
    collector = RAPLCollector(partition="zen4", node="rpc-91-1", sysfs_root=root)
    _fixed_clock(monkeypatch, [0.0] * 4 + [70.0] * 4)

    runtime = collector.prepare_runtime()
    collector.collect_rows(runtime, 0.0)
    rows = collector.collect_rows(runtime, 70.0)

    assert len(rows) == 4
    assert all(row["wrap_ambiguous"] for row in rows)
