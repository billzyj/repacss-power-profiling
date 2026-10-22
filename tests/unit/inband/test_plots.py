"""Tests for the plots of an in-band job report."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pytest

from inband.aggregation import EXPORT_COLUMNS
from inband.plots import bin_power, node_components, plot_report, ring_segments
from inband.storage import read_json_file
from shared.slurm.finalize_queue import enqueue
from shared.slurm.finalizer import FinalizerConfig, build_record, export_dir, process_queue
from tests.unit.inband.staging_fixtures import GPU_DOMAINS, T0, stage_node

START = T0.timestamp()


def _write_rows(path: Path, rows) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        for second, node, metric, group, joules, watts in rows:
            stamp = datetime.fromtimestamp(START + second, tz=timezone.utc).isoformat()
            writer.writerow(dict(zip(EXPORT_COLUMNS, (stamp, node, metric, group, joules, watts))))
    return path


def test_bins_average_energy_over_covered_time_including_counter_steps(tmp_path: Path):
    # A 100 W GPU whose counter advances every other second: rows of 200 J, then 0 J without power.
    rows = [(second, "n1", "gpu0", "gpu", 200.0 if second % 2 else 0.0, 200.0 if second % 2 else "") for second in range(1, 11)]
    rows += [(second, "n1", "package-0", "cpu_package", 100.0, 100.0) for second in range(1, 11)]
    power, groups = bin_power(_write_rows(tmp_path / "raw.csv", rows), START, START + 10, 2)

    assert groups == {("n1", "gpu0"): "gpu", ("n1", "package-0"): "cpu_package"}
    assert power[("n1", "package-0")] == [100.0, 100.0]
    # The 0 J rows count their seconds, so each bin averages the steps back to 100 W.
    assert power[("n1", "gpu0")] == [100.0, 100.0]


def test_components_sum_the_packages_and_keep_each_gpu():
    power = {
        ("n1", "package-0"): [100.0, None],
        ("n1", "package-1"): [120.0, 130.0],
        ("n1", "package-0/dram"): [10.0, 10.0],
        ("n1", "psys"): [900.0, 900.0],
        ("n1", "gpu10"): [5.0, 5.0],
        ("n1", "gpu2"): [7.0, 7.0],
        ("n2", "package-0"): [1.0, 1.0],
    }
    groups = {key: group for key, group in zip(power, ["cpu_package", "cpu_package", "host_dram", "platform", "gpu", "gpu", "cpu_package"])}
    assert node_components(power, groups, "n1") == [
        ("CPU", "cpu_package", [220.0, None]),  # a bin lacking one package is left empty
        ("Memory", "host_dram", [10.0, 10.0]),
        ("GPU (gpu2)", "gpu", [7.0, 7.0]),
        ("GPU (gpu10)", "gpu", [5.0, 5.0]),
    ]


def test_ring_segments_sum_the_nodes_in_kwh():
    nodes = {
        "n1": {"cpu_package_energy_j": 3.6e6, "host_dram_energy_j": None, "gpu_energy_j": 7.2e6},
        "n2": {"cpu_package_energy_j": 3.6e6, "host_dram_energy_j": 0.0},
    }
    assert ring_segments(nodes) == {"GPU": 2.0, "CPU": 2.0}


def test_plot_report_notes_a_missing_window(tmp_path: Path):
    pytest.importorskip("matplotlib")
    _write_rows(tmp_path / "raw_power.csv", [])
    record = {"job_id": "7", "nodes": ["n1"], "start_epoch": None, "end_epoch": None}
    notes = plot_report(tmp_path, {"nodes": {"n1": {"domains": {"x": {}}}}}, record)
    assert notes == ["Power plots were not drawn: the job window is unknown."]
    assert not list(tmp_path.glob("*.pdf"))


@pytest.mark.parametrize("nodes", [["rpg-93-1"], ["rpg-93-1", "rpg-93-2"]])
def test_the_finalizer_exports_plots_with_the_report(tmp_path: Path, nodes):
    pytest.importorskip("matplotlib")
    config = FinalizerConfig(queue_root=tmp_path / "queue", staging_root=tmp_path / "staging", report_root=tmp_path / "reports")
    for node in nodes:
        stage_node(config.staging_root / "repacss-777-100", tmp_path / "spool", node, gpu_domains=GPU_DOMAINS)
    record = build_record(
        job_id="777", attempt_id="repacss-777-100", cluster="repacss", user="alice", nodes=nodes,
        start_epoch=int(START), end_epoch=int(START) + 10, now=1000.0, exclusive="NODE",
    )
    enqueue(config.queue_root, record)
    assert process_queue(config, now=1000.0) == 1

    target = export_dir(config, record)
    expected = {"energy_ring.pdf", "power_timeseries.pdf"} | ({"power_timeseries_by_node.pdf"} if len(nodes) > 1 else set())
    assert {path.name for path in target.glob("*.pdf")} == expected
    assert all(path.stat().st_size > 1000 for path in target.glob("*.pdf"))
    assert not [note for note in read_json_file(target / "summary.json")["notes"] if "Plots" in note or "plots" in note]
