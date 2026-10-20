"""Tests for the profile-driven RAPL collector."""

from __future__ import annotations

from pathlib import Path

import pytest

from inband.collectors.rapl import RAPLCollector, counter_delta_uj, wrap_safe_seconds


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


def test_columns_and_readings_are_keyed_by_domain_path(fake_powercap):
    root = fake_powercap("h100_host_rapl")
    collector = RAPLCollector(partition="h100", node="rpg-93-1", sysfs_root=root)
    assert collector.probe().available
    # Both DRAM domains are named "dram"; their paths keep them apart.
    assert collector.csv_columns() == [
        "intel-rapl:0",
        "intel-rapl:0/intel-rapl:0:0",
        "intel-rapl:1",
        "intel-rapl:2",
        "intel-rapl:2/intel-rapl:2:0",
    ]

    runtime = collector.prepare_runtime()
    (root / "intel-rapl:2/intel-rapl:2:0/energy_uj").write_text("1500000\n", encoding="utf-8")
    (readings,) = collector.collect_rows(runtime)

    assert readings["intel-rapl:2/intel-rapl:2:0"] == 1_500_000
    assert readings["intel-rapl:0"] == 1_000_000


def test_an_unreadable_counter_is_left_out_of_the_sample(fake_powercap):
    root = fake_powercap("zen4_rapl")
    collector = RAPLCollector(partition="zen4", node="rpc-91-1", sysfs_root=root)
    runtime = collector.prepare_runtime()
    (root / "intel-rapl:1/energy_uj").write_text("not a number\n", encoding="utf-8")

    (readings,) = collector.collect_rows(runtime)

    assert sorted(readings) == ["intel-rapl:0", "intel-rapl:0/intel-rapl:0:0", "intel-rapl:1/intel-rapl:1:0"]
