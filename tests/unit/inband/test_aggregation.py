"""Tests for aggregating staged in-band RAPL data into job energy."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from inband.aggregation import Window, aggregate_attempt
from tests.unit.inband.staging_fixtures import GPU_DOMAINS, T0, ZEN4_DOMAINS, stage_node

FULL = Window(start=T0, end=T0 + timedelta(seconds=10))


def _aggregate(job_dir: Path, nodes, window=FULL):
    rows = []
    result = aggregate_attempt(
        job_dir, job_id="777", attempt_id="repacss-777-100", expected_nodes=nodes, window=window, emit=rows.append
    )
    return result, rows


def test_complete_nodes_give_job_totals_and_matching_raw_rows(tmp_path: Path):
    job_dir = tmp_path / "staging" / "repacss-777-100"
    for node in ("rpg-93-1", "rpg-93-2"):
        stage_node(job_dir, tmp_path / "spool", node)

    result, rows = _aggregate(job_dir, ["rpg-93-1", "rpg-93-2"])

    assert result["status"] == "complete"
    assert result["energy_j"] == {"cpu_package": 4400.0, "host_dram": 440.0, "gpu": None}  # 2 x 220 W / 22 W x 10 s
    assert result["average_power_w"] == {"cpu_package": 440.0, "host_dram": 44.0, "gpu": None}
    node = result["nodes"]["rpg-93-1"]
    assert node["domains"]["intel-rapl:1"]["energy_j"] == 10000.0  # psys: reported, never totalled
    assert node["coverage"] == 1.0
    assert len(rows) == 2 * 5 * 10 * 2  # nodes x domains x samples x (power, energy)
    package_energy = sum(row["value"] for row in rows if row["metric_name"] == "energy" and row["scope_type"] == "cpu_package")
    assert package_energy == pytest.approx(4400.0)
    assert rows[0]["scope_id"] == "package-0@intel-rapl:0"


def test_intervals_crossing_the_job_window_are_prorated(tmp_path: Path):
    job_dir = tmp_path / "staging" / "repacss-777-100"
    stage_node(job_dir, tmp_path / "spool", "rpg-93-1")
    window = Window(start=T0 + timedelta(seconds=2.5), end=T0 + timedelta(seconds=7.5))

    result, _ = _aggregate(job_dir, ["rpg-93-1"], window)

    package = result["nodes"]["rpg-93-1"]["domains"]["intel-rapl:0"]
    assert package["energy_j"] == pytest.approx(500.0)  # 100 W x 5 s
    assert package["average_power_w"] == pytest.approx(100.0)


def test_a_missing_node_leaves_job_totals_empty(tmp_path: Path):
    job_dir = tmp_path / "staging" / "repacss-777-100"
    stage_node(job_dir, tmp_path / "spool", "rpg-93-1")

    result, _ = _aggregate(job_dir, ["rpg-93-1", "rpg-93-2"])

    assert result["status"] == "partial"
    assert result["missing_nodes"] == ["rpg-93-2"]
    assert result["energy_j"]["cpu_package"] is None
    assert result["nodes"]["rpg-93-1"]["cpu_package_energy_j"] == 2200.0


def test_an_ambiguous_wrap_removes_that_domain_and_its_totals(tmp_path: Path):
    job_dir = tmp_path / "staging" / "repacss-777-100"
    stage_node(job_dir, tmp_path / "spool", "rpg-93-1", ambiguous=["intel-rapl:2"])

    result, _ = _aggregate(job_dir, ["rpg-93-1"])

    node = result["nodes"]["rpg-93-1"]
    assert node["domains"]["intel-rapl:2"]["energy_j"] is None
    assert node["cpu_package_energy_j"] is None
    assert node["host_dram_energy_j"] == 220.0
    assert result["status"] == "partial"


def test_a_chunk_that_differs_from_its_checksum_invalidates_the_node(tmp_path: Path):
    job_dir = tmp_path / "staging" / "repacss-777-100"
    node_dir = stage_node(job_dir, tmp_path / "spool", "rpg-93-1")
    with (node_dir / "chunks" / "rapl" / "00000001.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    result, rows = _aggregate(job_dir, ["rpg-93-1"])

    assert result["nodes"]["rpg-93-1"]["state"] == "invalid"
    assert result["status"] == "failed"
    assert rows == []


def test_zen4_reports_cpu_package_without_a_dram_total(tmp_path: Path):
    job_dir = tmp_path / "staging" / "repacss-777-100"
    stage_node(job_dir, tmp_path / "spool", "rpc-91-1", domains=ZEN4_DOMAINS)

    result, _ = _aggregate(job_dir, ["rpc-91-1"])

    assert result["status"] == "complete"
    assert result["energy_j"] == {"cpu_package": 3100.0, "host_dram": None, "gpu": None}
    assert any("host_dram" in note for note in result["notes"])
    assert result["nodes"]["rpc-91-1"]["domains"]["intel-rapl:0/intel-rapl:0:0"]["group"] == "core"


def test_gpu_energy_is_a_separate_total(tmp_path: Path):
    job_dir = tmp_path / "staging" / "repacss-777-100"
    stage_node(job_dir, tmp_path / "spool", "rpg-93-1", gpu_domains=GPU_DOMAINS)

    result, rows = _aggregate(job_dir, ["rpg-93-1"])

    assert result["status"] == "complete"
    assert result["energy_j"] == {"cpu_package": 2200.0, "host_dram": 220.0, "gpu": 9400.0}  # 940 W x 10 s
    gpu0 = result["nodes"]["rpg-93-1"]["domains"]["GPU-00000000-test"]
    assert (gpu0["name"], gpu0["group"], gpu0["energy_j"]) == ("gpu0", "gpu", 600.0)
    assert {row["collector_family"] for row in rows} == {"rapl", "nvml"}
    assert result["nodes"]["rpg-93-1"]["nvml_manifest_status"] == "ready"


def test_a_gpu_counter_reset_removes_that_gpu_and_the_gpu_total(tmp_path: Path):
    job_dir = tmp_path / "staging" / "repacss-777-100"
    stage_node(job_dir, tmp_path / "spool", "rpg-93-1", gpu_domains=GPU_DOMAINS, ambiguous=["GPU-00000002-test"])

    result, _ = _aggregate(job_dir, ["rpg-93-1"])

    assert result["nodes"]["rpg-93-1"]["domains"]["GPU-00000002-test"]["energy_j"] is None
    assert result["energy_j"]["gpu"] is None
    assert result["energy_j"]["cpu_package"] == 2200.0
