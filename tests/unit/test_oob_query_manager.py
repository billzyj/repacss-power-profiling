from __future__ import annotations

from contextlib import contextmanager

import pandas as pd

from oob.query_manager import QueryManager


@contextmanager
def _fake_connection(_database: str, _schema: str = None):
    class _Client:
        db_connection = object()

    yield _Client()


def test_query_manager_applies_global_limit_for_recent_queries(monkeypatch):
    manager = QueryManager("zen4")

    def fake_query_builder(metric, hostname, start_time, end_time, limit=100):
        assert hostname == "rpc-91-1"
        assert start_time is None
        assert end_time is None
        assert limit == 2
        return f"query-for-{metric}"

    monkeypatch.setattr(
        manager,
        "_get_node_type_and_query_func",
        lambda hostname: ("compute", fake_query_builder, "zen4", "idrac"),
    )
    monkeypatch.setattr(manager, "_get_metrics_for_node_type", lambda *_args: ["MetricA", "MetricB"])
    monkeypatch.setattr("oob.query_manager.get_pooled_connection", _fake_connection)

    frames = {
        "query-for-MetricA": pd.DataFrame(
            {
                "timestamp": [
                    "2025-01-01T00:00:00Z",
                    "2025-01-01T00:00:10Z",
                ],
                "hostname": ["rpc-91-1", "rpc-91-1"],
                "value": [100, 110],
                "units": ["W", "W"],
            }
        ),
        "query-for-MetricB": pd.DataFrame(
            {
                "timestamp": [
                    "2025-01-01T00:00:05Z",
                    "2025-01-01T00:00:15Z",
                ],
                "hostname": ["rpc-91-1", "rpc-91-1"],
                "value": [200, 210],
                "units": ["W", "W"],
            }
        ),
    }

    def fake_read_sql_query(query, _connection):
        return frames[query].copy()

    monkeypatch.setattr(pd, "read_sql_query", fake_read_sql_query)

    result = manager.get_power_metrics("rpc-91-1", limit=2)

    assert len(result) == 2
    assert result["metric"].tolist() == ["MetricA", "MetricB"]
    assert result["timestamp"].tolist() == ["2025-01-01T00:00:10Z", "2025-01-01T00:00:15Z"]


def test_query_manager_preserves_all_rows_for_time_window_queries(monkeypatch):
    manager = QueryManager("zen4")

    def fake_query_builder(metric, hostname, start_time, end_time, limit=100):
        assert limit == 3
        assert start_time == "2025-01-01 00:00:00"
        assert end_time == "2025-01-01 01:00:00"
        return f"query-for-{metric}"

    monkeypatch.setattr(
        manager,
        "_get_node_type_and_query_func",
        lambda hostname: ("compute", fake_query_builder, "zen4", "idrac"),
    )
    monkeypatch.setattr(manager, "_get_metrics_for_node_type", lambda *_args: ["MetricA", "MetricB"])
    monkeypatch.setattr("oob.query_manager.get_pooled_connection", _fake_connection)

    frames = {
        "query-for-MetricA": pd.DataFrame({"timestamp": ["2025-01-01T00:00:00Z"], "value": [100], "units": ["W"]}),
        "query-for-MetricB": pd.DataFrame({"timestamp": ["2025-01-01T00:00:05Z"], "value": [200], "units": ["W"]}),
    }

    monkeypatch.setattr(pd, "read_sql_query", lambda query, _connection: frames[query].copy())

    result = manager.get_power_metrics(
        "rpc-91-1",
        start_time="2025-01-01 00:00:00",
        end_time="2025-01-01 01:00:00",
        limit=3,
    )

    assert len(result) == 2
    assert result["metric"].tolist() == ["MetricA", "MetricB"]
