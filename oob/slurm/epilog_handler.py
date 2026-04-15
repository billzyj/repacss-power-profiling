#!/usr/bin/env python3
"""OOB Slurm epilog handler for job-scoped exports."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from shared.analysis.energy import compute_energy_kwh_for_hostname
from shared.models import SlurmJobContext
from shared.slurm.resolver import expand_nodelist, resolve_job_context_from_env
from shared.utils.conversions import convert_power_series_to_watts
from oob.backends.monster_db.backend import MonsterDBBackend

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _energy_to_pie_segments(energy_by_metric: Dict[str, float], is_h100: bool) -> Dict[str, float]:
    cpu = energy_by_metric.get("TotalCPUPower", 0.0)
    memory = energy_by_metric.get("TotalMemoryPower", 0.0)
    storage = energy_by_metric.get("TotalStoragePower", 0.0)
    fan = energy_by_metric.get("TotalFanPower", 0.0)
    out_ = energy_by_metric.get("SystemOutputPower", 0.0)
    in_ = energy_by_metric.get("SystemInputPower", 0.0)
    total = in_
    psu_loss = max(0.0, in_ - out_)
    gpu = energy_by_metric.get("PowerConsumption", 0.0) if is_h100 else 0.0
    if total <= 0:
        return {}
    components = cpu + memory + storage + fan + psu_loss + (gpu if is_h100 else 0.0)
    others = max(0.0, total - components)
    result = {
        "CPU": cpu,
        "Memory": memory,
        "Storage": storage,
        "Fan": fan,
        "PSU loss": psu_loss,
        "Others": others,
    }
    if is_h100:
        result["GPU"] = gpu
    return {key: value for key, value in result.items() if value > 0}


def summarize_job_power(
    raw_df: pd.DataFrame,
    nodelist: List[str],
    start_time: datetime,
    end_time: datetime,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float], Dict[str, float]]:
    energy_by_metric: Dict[str, float] = {}
    energy_gpu_per_fqdd: Dict[str, float] = {}
    if raw_df.empty:
        return raw_df, {}, {}, {}

    start_str = start_time.astimezone().isoformat()
    end_str = end_time.astimezone().isoformat()

    for node in nodelist:
        node_df = raw_df[raw_df["hostname"] == node].copy() if "hostname" in raw_df.columns else pd.DataFrame()
        if node_df.empty:
            continue
        for metric in node_df["metric"].dropna().unique():
            sub = node_df[node_df["metric"] == metric].copy()
            unit_metric = sub["units"].iloc[0] if "units" in sub.columns and len(sub) else "W"
            if metric == "PowerConsumption" and "fqdd" in sub.columns and "timestamp" in sub.columns:
                node_gpu_total = 0.0
                for fqdd in sub["fqdd"].dropna().unique():
                    sub_fqdd = sub[sub["fqdd"] == fqdd]
                    energy = compute_energy_kwh_for_hostname(sub_fqdd, unit_metric, node, start_str, end_str)
                    energy_gpu_per_fqdd[str(fqdd)] = energy_gpu_per_fqdd.get(str(fqdd), 0.0) + energy
                    node_gpu_total += energy
                energy_by_metric[metric] = energy_by_metric.get(metric, 0.0) + node_gpu_total
                continue
            if metric == "PowerConsumption" and "timestamp" in sub.columns:
                sub = (
                    sub.groupby(["timestamp", "hostname"], as_index=False)["value"]
                    .sum()
                    .assign(units=sub["units"].iloc[0] if "units" in sub.columns else "mW")
                )
            energy = compute_energy_kwh_for_hostname(sub, unit_metric, node, start_str, end_str)
            energy_by_metric[metric] = energy_by_metric.get(metric, 0.0) + energy

    pie_segments = _energy_to_pie_segments(energy_by_metric, is_h100=any(node.lower().startswith("rpg") for node in nodelist))
    return raw_df, pie_segments, energy_by_metric, energy_gpu_per_fqdd


def _summary_row(raw_columns: List[str], metric: str, value: float, fqdd: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {column: "" for column in raw_columns}
    if "fqdd" in row:
        row["fqdd"] = fqdd
    if "value" in row:
        row["value"] = value
    if "units" in row:
        row["units"] = "kWh"
    if "metric" in row:
        row["metric"] = metric
    return row


def _energy_summary_rows(
    energy_by_metric: Dict[str, float],
    pie_segments: Dict[str, float],
    energy_gpu_per_fqdd: Dict[str, float],
    is_h100: bool,
    raw_columns: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rows.append(_summary_row(raw_columns, "ENERGY_SUMMARY_kWh", 0.0, ""))
    for name in ["TotalCPUPower", "TotalMemoryPower", "TotalStoragePower", "TotalFanPower", "SystemInputPower", "SystemOutputPower"]:
        if name in energy_by_metric and energy_by_metric[name] is not None:
            rows.append(_summary_row(raw_columns, f"Energy_{name}", energy_by_metric[name], ""))
    if is_h100 and "PowerConsumption" in energy_by_metric:
        rows.append(_summary_row(raw_columns, "Energy_GPU_total", energy_by_metric["PowerConsumption"], ""))
    if "PSU loss" in pie_segments:
        rows.append(_summary_row(raw_columns, "Energy_PSU_loss", pie_segments["PSU loss"], ""))
    if "Others" in pie_segments:
        rows.append(_summary_row(raw_columns, "Energy_Others", pie_segments["Others"], ""))
    for fqdd, energy in sorted(energy_gpu_per_fqdd.items()):
        rows.append(_summary_row(raw_columns, f"Energy_GPU_fqdd_{fqdd}", energy, str(fqdd)))
    return pd.DataFrame(rows, columns=raw_columns) if rows else pd.DataFrame()


def _utc_series_to_local_with_offset(series: pd.Series) -> pd.Series:
    local_tz = datetime.now().astimezone().tzinfo
    utc = pd.to_datetime(series, utc=True, errors="coerce")
    local = utc.dt.tz_convert(local_tz)

    def fmt(value):
        if pd.isna(value):
            return ""
        zone = value.strftime("%z")
        return value.strftime("%Y-%m-%d %H:%M:%S") + (zone[:3] if len(zone) >= 3 else "")

    return local.apply(fmt)


def save_csv(
    raw_df: pd.DataFrame,
    path: Path,
    energy_by_metric: Optional[Dict[str, float]] = None,
    pie_segments: Optional[Dict[str, float]] = None,
    energy_gpu_per_fqdd: Optional[Dict[str, float]] = None,
    is_h100: bool = False,
) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw_df.empty:
        return False
    out = raw_df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = _utc_series_to_local_with_offset(out["timestamp"])
    out.to_csv(path, index=False)
    if energy_by_metric is None or pie_segments is None:
        return True
    summary = _energy_summary_rows(energy_by_metric, pie_segments, energy_gpu_per_fqdd or {}, is_h100, list(raw_df.columns))
    if summary.empty:
        return True
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    summary.to_csv(path, mode="a", header=False, index=False)
    return True


def plot_time_series(raw_df: pd.DataFrame, path: Path) -> bool:
    if raw_df.empty or "timestamp" not in raw_df.columns or "value" not in raw_df.columns:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from shared.utils.plot_style import (
            METRIC_ID_TO_DISPLAY,
            POWER_DISTRIBUTION_TIME_SERIES_COLORS,
            TIME_SERIES_GPU_FQDD_COLORS,
            apply_paper_style,
        )
    except ImportError:
        return False
    apply_paper_style()
    df = raw_df.copy()
    local_tz = datetime.now().astimezone().tzinfo
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert(local_tz)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(local_tz)
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return False
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric in df["metric"].unique():
        sub_all = df[df["metric"] == metric]
        unit_metric = sub_all["units"].iloc[0] if "units" in sub_all.columns and len(sub_all) else "W"
        display_label = METRIC_ID_TO_DISPLAY.get(metric, metric)
        color = POWER_DISTRIBUTION_TIME_SERIES_COLORS.get(display_label, "#95a5a6")
        if metric == "PowerConsumption" and "fqdd" in df.columns:
            fqdd_list = sub_all["fqdd"].dropna().unique().tolist()
            for index, fqdd in enumerate(fqdd_list):
                sub = sub_all[sub_all["fqdd"] == fqdd].sort_values("timestamp").copy()
                if sub.empty:
                    continue
                sub["power_w"] = convert_power_series_to_watts(sub["value"], unit_metric)
                ax.plot(sub["timestamp"], sub["power_w"], label=f"GPU ({fqdd})", color=TIME_SERIES_GPU_FQDD_COLORS[index % len(TIME_SERIES_GPU_FQDD_COLORS)], alpha=0.8)
        else:
            sub = sub_all.sort_values("timestamp").copy()
            sub["power_w"] = convert_power_series_to_watts(sub["value"], unit_metric)
            ax.plot(sub["timestamp"], sub["power_w"], label=display_label, color=color, alpha=0.8)
    ax.set_xlabel("Time (local)", fontsize=18, weight="bold")
    ax.set_ylabel("Power (W)", fontsize=18, weight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=16, frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M", tz=df["timestamp"].dt.tz))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_pie(pie_segments: Dict[str, float], path: Path, job_id: Optional[str] = None) -> bool:
    if not pie_segments or sum(pie_segments.values()) == 0:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from shared.utils.plot_style import (
            POWER_DISTRIBUTION_RING_COLORS,
            apply_paper_style,
            create_ring_with_smart_labels,
            set_pie_text_color,
        )
    except ImportError:
        return False
    apply_paper_style()
    labels = list(pie_segments.keys())
    values = [pie_segments[label] for label in labels]
    colors = [POWER_DISTRIBUTION_RING_COLORS.get(label, "#95a5a6") for label in labels]
    total_kwh = sum(values)
    fig, ax = plt.subplots(figsize=(8, 8))
    _, _, autotexts = create_ring_with_smart_labels(
        ax,
        values,
        labels,
        colors,
        "Total Energy Consumption (kWh)",
        center_title=f"Job {job_id}" if job_id else "Total",
        center_value=f"{total_kwh:.3f} kWh",
        startangle=90,
    )
    set_pie_text_color(autotexts, colors, values, labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def handle_oob_job(context: SlurmJobContext, out_dir: Optional[Path] = None) -> Path:
    """Run the OOB job export for a Slurm job context."""
    if out_dir is None:
        power_base = Path(os.environ.get("MONSTER_POWER_BASE", "/mnt/SHARED-AREA/power_reports"))
        out_dir = power_base / context.user / context.job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "raw_power.csv"
    ts_path = out_dir / "power_timeseries.pdf"
    pie_path = out_dir / "energy_ring.pdf"

    backend = MonsterDBBackend()
    raw_df = backend.query_job(context)
    raw_df, pie_segments, energy_by_metric, energy_gpu_per_fqdd = summarize_job_power(
        raw_df,
        context.nodes,
        context.start_time,
        context.end_time,
    )
    save_csv(
        raw_df,
        csv_path,
        energy_by_metric=energy_by_metric,
        pie_segments=pie_segments,
        energy_gpu_per_fqdd=energy_gpu_per_fqdd,
        is_h100=any(node.lower().startswith("rpg") for node in context.nodes),
    )
    plot_time_series(raw_df, ts_path)
    plot_pie(pie_segments, pie_path, job_id=context.job_id)
    return out_dir


def parse_cli_test_args() -> Optional[Tuple[SlurmJobContext, str]]:
    parser = argparse.ArgumentParser(description="OOB Slurm job power query")
    parser.add_argument("job_id", nargs="?", help="Job ID for ring chart center.")
    parser.add_argument("--start", type=int, help="Start time (Unix timestamp).")
    parser.add_argument("--end", type=int, help="End time (Unix timestamp).")
    parser.add_argument("--nodelist", type=str, help="Node list (e.g. rpc-97-16).")
    parser.add_argument("--outdir", type=str, default=str(_REPO_ROOT / "output" / "tmp"))
    args = parser.parse_args()
    if args.start is None or args.end is None or args.nodelist is None:
        return None
    nodes = expand_nodelist(args.nodelist)
    if not nodes:
        return None
    context = SlurmJobContext(
        job_id=args.job_id or "test",
        user=os.environ.get("USER", "unknown"),
        nodelist=args.nodelist,
        nodes=nodes,
        start_time=datetime.fromtimestamp(args.start),
        end_time=datetime.fromtimestamp(args.end),
        comment="power:oob",
    )
    return context, args.outdir


def main() -> None:
    cli_result = parse_cli_test_args()
    if cli_result is not None:
        context, out_dir = cli_result
    else:
        context = resolve_job_context_from_env()
        out_dir = os.environ.get("MONSTER_POWER_OUTDIR")
        if not out_dir:
            raise SystemExit("MONSTER_POWER_OUTDIR is required in epilog mode")
    handle_oob_job(context, Path(out_dir))


if __name__ == "__main__":
    main()
