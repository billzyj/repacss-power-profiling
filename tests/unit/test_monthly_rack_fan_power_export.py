from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts.export_monthly_rack_fan_power import (
    build_position_group_map,
    build_positional_query,
    build_positional_overview,
    build_query_groups,
    build_query_targets,
    build_total_fan_power_batch_query,
    build_total_fan_power_percentile_query,
    query_node_total_fan_power,
    write_run_report,
)
from shared.constants.nodes import RACK_CONFIGS


def test_build_query_targets_uses_configured_rack_compute_nodes() -> None:
    targets = build_query_targets([91, 93, 97])

    assert [target.rack for target in targets] == (
        [91] * len(RACK_CONFIGS[91]["compute_nodes"])
        + [93] * len(RACK_CONFIGS[93]["compute_nodes"])
        + [97] * len(RACK_CONFIGS[97]["compute_nodes"])
    )
    assert [target.hostname for target in targets] == (
        RACK_CONFIGS[91]["compute_nodes"]
        + RACK_CONFIGS[93]["compute_nodes"]
        + RACK_CONFIGS[97]["compute_nodes"]
    )
    assert {target.database for target in targets if target.hostname.startswith("rpc")} == {"zen4"}
    assert {target.database for target in targets if target.hostname.startswith("rpg")} == {"h100"}


@contextmanager
def _fake_connection(_database: str, _schema: str = "idrac"):
    class _Client:
        db_connection = object()

    yield _Client()


def test_query_node_total_fan_power_queries_only_total_fan_power(monkeypatch) -> None:
    captured_queries: list[str] = []

    def fake_read_sql_query(query, _connection):
        captured_queries.append(query)
        return pd.DataFrame(
            {
                "timestamp": ["2026-05-15 00:00:00"],
                "hostname": ["rpc-91-1"],
                "source": ["idrac"],
                "fqdd": ["System.Embedded.1"],
                "value": [42.0],
                "units": ["W"],
            }
        )

    monkeypatch.setattr("scripts.export_monthly_rack_fan_power.get_pooled_connection", _fake_connection)
    monkeypatch.setattr(pd, "read_sql_query", fake_read_sql_query)

    df = query_node_total_fan_power(
        rack=91,
        hostname="rpc-91-1",
        database="zen4",
        start_time=datetime(2026, 5, 15, 0, 0, 0),
        end_time=datetime(2026, 6, 15, 0, 0, 0),
    )

    assert len(captured_queries) == 1
    assert "idrac.totalfanpower" in captured_queries[0].lower()
    assert "n.hostname = 'rpc-91-1'" in captured_queries[0]
    assert df["rack"].tolist() == [91]
    assert df["metric"].tolist() == ["TotalFanPower"]


def test_build_query_groups_batches_nodes_by_rack_database() -> None:
    groups = build_query_groups(build_query_targets([91, 93]))

    assert [(group.rack, group.database, len(group.hostnames)) for group in groups] == [
        (91, "zen4", len(RACK_CONFIGS[91]["compute_nodes"])),
        (93, "h100", len(RACK_CONFIGS[93]["compute_nodes"])),
    ]


def test_position_group_map_splits_normal_racks_and_keeps_rack_93_gpu_nodes_together() -> None:
    rack_91_groups = build_position_group_map(91, RACK_CONFIGS[91]["compute_nodes"])
    rack_93_groups = build_position_group_map(93, RACK_CONFIGS[93]["compute_nodes"])

    assert rack_91_groups["lower"] == tuple(f"rpc-91-{slot}" for slot in range(1, 8))
    assert rack_91_groups["upper"] == tuple(f"rpc-91-{slot}" for slot in range(13, 21))
    assert "middle" not in rack_91_groups
    assert rack_93_groups == {"gpu_all": tuple(f"rpg-93-{slot}" for slot in range(1, 9))}


def test_positional_query_compares_fan_power_and_system_input_by_time_bucket() -> None:
    query = build_positional_query(
        rack=91,
        grouped_hostnames={
            "lower": ("rpc-91-1", "rpc-91-2"),
            "upper": ("rpc-91-13", "rpc-91-14"),
        },
        granularity="hour",
        start_time=datetime(2026, 6, 2, 0, 0, 0),
        end_time=datetime(2026, 6, 9, 0, 0, 0),
        sample_seconds=300,
    )

    assert "date_trunc('hour', fan.timestamp)" in query
    assert "idrac.totalfanpower fan" in query.lower()
    assert "idrac.systeminputpower sys" in query.lower()
    assert "WHEN n.hostname IN ('rpc-91-1', 'rpc-91-2') THEN 'lower'" in query
    assert "fan_ratio_median" in query


def test_positional_overview_summarizes_upper_minus_lower_delta() -> None:
    comparison = pd.DataFrame(
        {
            "rack": [91, 91, 91],
            "granularity": ["hour", "hour", "hour"],
            "fan_median_delta_upper_minus_lower": [10.0, -5.0, 15.0],
            "fan_ratio_delta_upper_minus_lower": [0.1, -0.1, 0.2],
        }
    )

    overview = build_positional_overview(comparison)

    assert overview.loc[0, "rack"] == 91
    assert overview.loc[0, "upper_gt_lower_percent"] == 66.66666666666666
    assert overview.loc[0, "mean_fan_median_delta_upper_minus_lower"] == 20.0 / 3.0


def test_batch_query_uses_hostname_in_clause_for_rack_nodes() -> None:
    query = build_total_fan_power_batch_query(
        ["rpc-91-1", "rpc-91-2"],
        start_time=datetime(2026, 5, 15, 0, 0, 0),
        end_time=datetime(2026, 6, 15, 0, 0, 0),
    )

    assert "FROM idrac.totalfanpower p" in query
    assert "n.hostname IN ('rpc-91-1', 'rpc-91-2')" in query
    assert "ORDER BY n.hostname ASC, p.timestamp ASC" in query


def test_percentile_query_returns_distribution_points_not_raw_samples() -> None:
    query = build_total_fan_power_percentile_query(
        ["rpc-91-1", "rpc-91-2"],
        start_time=datetime(2026, 5, 15, 0, 0, 0),
        end_time=datetime(2026, 6, 15, 0, 0, 0),
    )

    assert "generate_series(0, 100)" in query
    assert "percentile_cont(gs.percentile / 100.0)" in query
    assert "MOD(EXTRACT(EPOCH FROM p.timestamp)::bigint, 300) = 0" in query
    assert "sample_count" in query


def test_write_run_report_records_artifacts_and_failures(tmp_path: Path) -> None:
    report_path = write_run_report(
        output_dir=tmp_path,
        start_time=datetime(2026, 5, 15, 0, 0, 0),
        end_time=datetime(2026, 6, 15, 0, 0, 0),
        racks=[91, 92],
        target_count=40,
        group_count=2,
        worker_count=2,
        mode="percentiles",
        sample_seconds=300,
        row_count=0,
        summary_path=tmp_path / "summary.csv",
        raw_path=tmp_path / "raw.csv.gz",
        figure_paths=[tmp_path / "rack_91.png", tmp_path / "rack_92.png"],
        failures=[{"rack": 91, "database": "zen4", "schema": "idrac", "error": "DNS failed"}],
    )

    text = report_path.read_text(encoding="utf-8")
    assert "Worker count: 2" in text
    assert "Rows returned: 0" in text
    assert "rack_91.png" in text
    assert "DNS failed" in text
