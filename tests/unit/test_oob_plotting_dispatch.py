from __future__ import annotations

from datetime import datetime

import pandas as pd

from oob.slurm import epilog_handler
from shared.models import SlurmJobContext


def _raw_df(hostnames):
    rows = []
    for hostname in hostnames:
        rows.append(
            {
                "timestamp": "2026-05-11T00:00:00Z",
                "hostname": hostname,
                "value": 100.0,
                "units": "W",
                "metric": "SystemInputPower",
            }
        )
    return pd.DataFrame(rows)


def test_plot_time_series_dispatches_multi_node_to_aggregate(monkeypatch, tmp_path) -> None:
    calls = []

    monkeypatch.setattr(epilog_handler, "plot_single_node_time_series", lambda *_args: calls.append("single") or True)
    monkeypatch.setattr(epilog_handler, "plot_multi_node_time_series_aggregate", lambda *_args: calls.append("aggregate") or True)

    assert epilog_handler.plot_time_series(_raw_df(["rpc-97-1", "rpc-97-2"]), tmp_path / "out.pdf")
    assert calls == ["aggregate"]


def test_handle_oob_job_writes_multi_node_total_and_by_node_plots(monkeypatch, tmp_path) -> None:
    raw_df = _raw_df(["rpc-97-1", "rpc-97-2"])
    calls = []

    class _Backend:
        def query_job(self, _context):
            return raw_df

    monkeypatch.setattr(epilog_handler, "MonsterDBBackend", lambda: _Backend())
    monkeypatch.setattr(epilog_handler, "summarize_job_power", lambda df, *_args: (df, {}, {}, {}))
    monkeypatch.setattr(epilog_handler, "save_csv", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(epilog_handler, "plot_pie", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        epilog_handler,
        "save_multi_node_time_series_total_csv",
        lambda _df, path: calls.append(path.name) or True,
    )
    monkeypatch.setattr(
        epilog_handler,
        "plot_multi_node_time_series_aggregate",
        lambda _df, path: calls.append(path.name) or True,
    )
    monkeypatch.setattr(
        epilog_handler,
        "plot_multi_node_time_series_per_node",
        lambda _df, path: calls.append(path.name) or True,
    )
    monkeypatch.setattr(
        epilog_handler,
        "plot_multi_node_time_series_per_node_files",
        lambda _df, path: calls.append(path.name) or [],
    )
    monkeypatch.setattr(epilog_handler, "plot_single_node_time_series", lambda *_args: calls.append("single") or True)

    context = SlurmJobContext(
        job_id="123",
        user="alice",
        nodelist="rpc-97-[1-2]",
        nodes=["rpc-97-1", "rpc-97-2"],
        start_time=datetime(2026, 5, 11, 0, 0, 0),
        end_time=datetime(2026, 5, 11, 1, 0, 0),
    )

    epilog_handler.handle_oob_job(context, tmp_path)

    assert calls == [
        "power_timeseries_total.csv",
        "power_timeseries_total.pdf",
        "power_timeseries_by_node.pdf",
        "power_timeseries_nodes",
    ]


def test_build_multi_node_time_series_total_sums_nodes() -> None:
    raw_df = _raw_df(["rpc-97-1", "rpc-97-2"])

    total_df = epilog_handler.build_multi_node_time_series_total(raw_df)

    assert len(total_df) == 1
    assert total_df.iloc[0]["metric"] == "SystemInputPower"
    assert total_df.iloc[0]["total_power_w"] == 200.0
    assert total_df.iloc[0]["node_count"] == 2
