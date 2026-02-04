#!/usr/bin/env python3
"""
Slurm job power query: at job end (EpilogSlurmctld), use job env to query this repo's
DB for node power, save raw CSV, plot time series and energy pie chart.

- Node id starting with rpc -> zen4 DB, ZEN4_METRICS.
- Node id starting with rpg -> h100 DB, H100_METRICS.
- Raw data -> CSV; time series plot; total energy per metric -> pie chart.
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Repo root on path (epilog may run from any cwd)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
for _p in (str(_REPO_ROOT), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

try:
    import hostlist
except ImportError:
    hostlist = None

from analysis.energy import compute_energy_kwh_for_hostname
from constants.metrics import ZEN4_METRICS, H100_METRICS
from database.database import get_raw_database_connection
from queries.compute.idrac import get_compute_metrics_with_joins
from utils.conversions import convert_power_series_to_watts


def expand_nodelist(nodelist: str) -> List[str]:
    """Expand Slurm nodelist to list of hostnames."""
    nodelist = (nodelist or "").strip()
    if not nodelist:
        return []
    if hostlist is not None:
        try:
            return hostlist.expand_hostlist(nodelist)
        except Exception:
            pass
    # Fallback: comma-separated
    return [n.strip() for n in nodelist.split(",") if n.strip()]


def node_db_and_metrics(node_id: str) -> Optional[Tuple[str, str, List[str]]]:
    """Return (database, schema, metrics) for node_id, or None if unsupported."""
    node_id = (node_id or "").strip().lower()
    if node_id.startswith("rpc"):
        return ("zen4", "idrac", list(ZEN4_METRICS))
    if node_id.startswith("rpg"):
        return ("h100", "idrac", list(H100_METRICS))
    return None


def parse_epilog_env() -> Optional[Dict[str, Any]]:
    """If running under EpilogSlurmctld, return start_ts, end_ts, nodelist (expanded)."""
    start_raw = os.environ.get("SLURM_JOB_START_TIME")
    end_raw = os.environ.get("SLURM_JOB_END_TIME")
    nodelist_raw = (os.environ.get("SLURM_JOB_NODELIST") or "").strip()
    if not start_raw or not end_raw or not nodelist_raw:
        return None
    try:
        start_ts = datetime.fromtimestamp(int(start_raw), tz=timezone.utc).replace(tzinfo=None)
        end_ts = datetime.fromtimestamp(int(end_raw), tz=timezone.utc).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None
    nodes = expand_nodelist(nodelist_raw)
    if not nodes:
        return None
    return {
        "start_time": start_ts,
        "end_time": end_ts,
        "nodelist": nodes,
        "job_id": os.environ.get("SLURM_JOB_ID", "job"),
    }


def fetch_raw_power_for_node(
    conn: Any,
    hostname: str,
    metrics: List[str],
    start_str: str,
    end_str: str,
) -> pd.DataFrame:
    """Query raw power for one node using an existing connection."""
    rows = []
    for metric in metrics:
        q = get_compute_metrics_with_joins(
            metric, hostname=hostname, start_time=start_str, end_time=end_str
        )
        df = pd.read_sql_query(q, conn)
        if not df.empty:
            df["metric"] = metric
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _energy_to_pie_segments(
    energy_by_metric: Dict[str, float], is_h100: bool
) -> Dict[str, float]:
    """
    Convert raw metric energies to pie segments.
    Zen4: CPU, Memory, Storage, Fan, PSU loss, Others (6).
    H100: + GPU (4 FQDDs summed) (7).
    Total = In, PSU loss = In - Out, Others = Total - (CPU+Memory+Storage+Fan+PSU_loss [+ GPU]).
    """
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
    out: Dict[str, float] = {
        "CPU": cpu,
        "Memory": memory,
        "Storage": storage,
        "Fan": fan,
        "PSU loss": psu_loss,
        "Others": others,
    }
    if is_h100:
        out["GPU"] = gpu
    # Drop zero segments so pie doesn't show empty slices
    return {k: v for k, v in out.items() if v > 0}


def run_job_power(
    start_time: datetime,
    end_time: datetime,
    nodelist: List[str],
    out_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    For each node, fetch raw power (rpc->ZEN4, rpg->H100), combine into one DataFrame.
    Compute total energy per metric and per-FQDD GPU energy.
    Return (raw_df, pie_segments, energy_by_metric, energy_gpu_per_fqdd).
    """
    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    all_dfs: List[pd.DataFrame] = []
    energy_by_metric: Dict[str, float] = {}
    energy_gpu_per_fqdd: Dict[str, float] = {}

    # Group nodes by (database, schema) to reuse one connection per DB
    by_db: Dict[Tuple[str, str], List[Tuple[str, List[str]]]] = {}
    for node in nodelist:
        info = node_db_and_metrics(node)
        if info is None:
            continue
        database, schema, metrics = info
        key = (database, schema)
        if key not in by_db:
            by_db[key] = []
        by_db[key].append((node, metrics))

    for (database, schema), node_metrics_list in by_db.items():
        conn = get_raw_database_connection(database, schema)
        if conn is None:
            continue
        try:
            for node, metrics in node_metrics_list:
                df = fetch_raw_power_for_node(
                    conn, node, metrics, start_str, end_str
                )
                if df.empty:
                    continue
                all_dfs.append(df)
                for metric in df["metric"].unique():
                    sub = df[df["metric"] == metric].copy()
                    unit_metric = sub["units"].iloc[0] if "units" in sub.columns and len(sub) else "W"
                    # GPU (PowerConsumption): unit mW; per-FQDD energy, then total = sum(per-FQDD) for consistency
                    if metric == "PowerConsumption" and "fqdd" in sub.columns and "timestamp" in sub.columns and "hostname" in sub.columns:
                        node_gpu_total = 0.0
                        for fqdd in sub["fqdd"].dropna().unique():
                            sub_fqdd = sub[sub["fqdd"] == fqdd]
                            e_fqdd = compute_energy_kwh_for_hostname(
                                sub_fqdd, unit_metric, node, start_str, end_str
                            )
                            energy_gpu_per_fqdd[str(fqdd)] = energy_gpu_per_fqdd.get(str(fqdd), 0.0) + e_fqdd
                            node_gpu_total += e_fqdd
                        energy_by_metric[metric] = energy_by_metric.get(metric, 0.0) + node_gpu_total
                        continue
                    elif metric == "PowerConsumption" and "timestamp" in sub.columns and "hostname" in sub.columns:
                        sub = (
                            sub.groupby(["timestamp", "hostname"], as_index=False)["value"]
                            .sum()
                            .assign(units=sub["units"].iloc[0] if "units" in sub.columns else "mW")
                        )
                    e = compute_energy_kwh_for_hostname(
                        sub, unit_metric, node, start_str, end_str
                    )
                    energy_by_metric[metric] = energy_by_metric.get(metric, 0.0) + e
        finally:
            try:
                conn.close()
            except Exception:
                pass

    raw_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    pie_segments = _energy_to_pie_segments(energy_by_metric, is_h100=any(n.lower().startswith("rpg") for n in nodelist))
    return raw_df, pie_segments, energy_by_metric, energy_gpu_per_fqdd


def _energy_summary_rows(
    energy_by_metric: Dict[str, float],
    pie_segments: Dict[str, float],
    energy_gpu_per_fqdd: Dict[str, float],
    is_h100: bool,
    raw_columns: List[str],
) -> pd.DataFrame:
    """Build DataFrame of energy summary rows (same columns as raw for appending)."""
    rows: List[Dict[str, Any]] = []
    # Header row for summary section (metric column marks it)
    rows.append(_summary_row(raw_columns, "ENERGY_SUMMARY_kWh", 0.0, ""))
    # Original metric energies (kWh)
    for name in ["TotalCPUPower", "TotalMemoryPower", "TotalStoragePower", "TotalFanPower", "SystemInputPower", "SystemOutputPower"]:
        if name in energy_by_metric and energy_by_metric[name] is not None:
            rows.append(_summary_row(raw_columns, f"Energy_{name}", energy_by_metric[name], ""))
    if is_h100 and "PowerConsumption" in energy_by_metric:
        rows.append(_summary_row(raw_columns, "Energy_GPU_total", energy_by_metric["PowerConsumption"], ""))
    # Derived: PSU loss, Others
    if "PSU loss" in pie_segments:
        rows.append(_summary_row(raw_columns, "Energy_PSU_loss", pie_segments["PSU loss"], ""))
    if "Others" in pie_segments:
        rows.append(_summary_row(raw_columns, "Energy_Others", pie_segments["Others"], ""))
    # GPU per slot (FQDD)
    for fqdd, e in sorted(energy_gpu_per_fqdd.items()):
        rows.append(_summary_row(raw_columns, f"Energy_GPU_fqdd_{fqdd}", e, str(fqdd)))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=raw_columns)


def _summary_row(raw_columns: List[str], metric: str, value: float, fqdd: str) -> Dict[str, Any]:
    """One summary row with same columns as raw CSV; timestamp/hostname/source empty."""
    row: Dict[str, Any] = {c: "" for c in raw_columns}
    if "timestamp" in row:
        row["timestamp"] = ""
    if "hostname" in row:
        row["hostname"] = ""
    if "source" in row:
        row["source"] = ""
    if "fqdd" in row:
        row["fqdd"] = fqdd
    if "value" in row:
        row["value"] = value
    if "units" in row:
        row["units"] = "kWh"
    if "metric" in row:
        row["metric"] = metric
    return row


def save_csv(
    raw_df: pd.DataFrame,
    path: Path,
    energy_by_metric: Optional[Dict[str, float]] = None,
    pie_segments: Optional[Dict[str, float]] = None,
    energy_gpu_per_fqdd: Optional[Dict[str, float]] = None,
    is_h100: bool = False,
) -> bool:
    """Save raw data to CSV; optionally append energy summary rows below (for pie chart). Returns True if file was written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw_df.empty:
        return False
    raw_df.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    if energy_by_metric is None or pie_segments is None:
        return True
    summary = _energy_summary_rows(
        energy_by_metric,
        pie_segments,
        energy_gpu_per_fqdd or {},
        is_h100,
        list(raw_df.columns),
    )
    if summary.empty:
        return True
    with path.open("a", encoding="utf-8") as f:
        f.write("\n")
    summary.to_csv(path, mode="a", header=False, index=False)
    return True


def plot_time_series(raw_df: pd.DataFrame, path: Path) -> bool:
    """Plot power (W) vs time per metric; save figure. Font/style aligned with pie chart. Returns True if file was written."""
    if raw_df.empty or "timestamp" not in raw_df.columns or "value" not in raw_df.columns:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from utils.plot_style import apply_paper_style
    except ImportError:
        return False
    apply_paper_style()
    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return False
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric in df["metric"].unique():
        sub_all = df[df["metric"] == metric]
        unit_metric = sub_all["units"].iloc[0] if "units" in sub_all.columns and len(sub_all) else "W"
        # GPU (PowerConsumption): plot one curve per FQDD; unit is mW
        if metric == "PowerConsumption" and "fqdd" in df.columns:
            for fqdd in sub_all["fqdd"].dropna().unique():
                sub = sub_all[sub_all["fqdd"] == fqdd].sort_values("timestamp")
                if sub.empty:
                    continue
                sub = sub.copy()
                sub["power_w"] = convert_power_series_to_watts(sub["value"], unit_metric)
                ax.plot(sub["timestamp"], sub["power_w"], label=f"GPU ({fqdd})", alpha=0.8)
        else:
            sub = sub_all.sort_values("timestamp")
            sub = sub.copy()
            sub["power_w"] = convert_power_series_to_watts(sub["value"], unit_metric)
            ax.plot(sub["timestamp"], sub["power_w"], label=metric, alpha=0.8)
    ax.set_xlabel("Time (UTC)", fontsize=13, weight="bold")
    ax.set_ylabel("Power (W)", fontsize=13, weight="bold")
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=14, frameon=True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M", tz=df["timestamp"].dt.tz))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_pie(pie_segments: Dict[str, float], path: Path) -> bool:
    """Plot pie chart: Zen4 = CPU, Memory, Storage, Fan, PSU loss, Others; H100 + GPU. Returns True if file was written."""
    if not pie_segments or sum(pie_segments.values()) == 0:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from utils.plot_style import (
            POWER_DISTRIBUTION_COLORS,
            apply_paper_style,
            create_pie_with_smart_labels,
            set_pie_text_color,
        )
    except ImportError:
        return False
    apply_paper_style()
    # pie_segments keys are display labels (CPU, Memory, Storage, Fan, PSU loss, Others, [GPU])
    display_labels = list(pie_segments.keys())
    values = [pie_segments[k] for k in display_labels]
    colors = [
        POWER_DISTRIBUTION_COLORS.get(lbl, "#95a5a6") for lbl in display_labels
    ]
    total_kwh = sum(values)
    title = f"Total Energy Consumption (kWh)\n{total_kwh:.3f} kWh"
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = create_pie_with_smart_labels(
        ax, values, display_labels, colors, title, startangle=90
    )
    set_pie_text_color(autotexts, colors, values, display_labels)
    # Legend order: GPU, Memory, CPU, Storage, Others, PSU loss, Fan
    label_order = ["GPU", "Memory", "CPU", "Storage", "Others", "PSU loss", "Fan"]
    seen = set()
    legend_labels = []
    for lbl in label_order:
        if lbl in display_labels and lbl not in seen:
            legend_labels.append(lbl)
            seen.add(lbl)
    for lbl in display_labels:
        if lbl not in seen:
            legend_labels.append(lbl)
    legend_elements = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=POWER_DISTRIBUTION_COLORS.get(lbl, "#95a5a6"),
            edgecolor="white",
            linewidth=1.2,
        )
        for lbl in legend_labels
    ]
    fig.legend(
        legend_elements,
        legend_labels,
        loc="lower center",
        ncol=min(len(legend_labels), 4),
        fontsize=11,
        frameon=True,
        framealpha=0.95,
        edgecolor="gray",
        fancybox=True,
    )
    plt.tight_layout(rect=[0, 0.12, 1, 0.98])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def parse_cli_test_args() -> Optional[Tuple[Dict[str, Any], str]]:
    """If --start/--end/--nodelist are given, return (job_info, outdir). Outdir defaults to output/tmp."""
    parser = argparse.ArgumentParser(
        description="Slurm job power query (epilog env or test: --start/--end/--nodelist [--outdir])."
    )
    parser.add_argument("job_id", nargs="?", help="Job ID (default: test).")
    parser.add_argument("--start", type=int, help="Start time (Unix timestamp).")
    parser.add_argument("--end", type=int, help="End time (Unix timestamp).")
    parser.add_argument("--nodelist", type=str, help="Node list (e.g. rpc-97-16).")
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(_REPO_ROOT / "output" / "tmp"),
        help="Output directory for CSV and plots (default: output/tmp).",
    )
    args = parser.parse_args()
    if args.start is None or args.end is None or args.nodelist is None:
        return None
    nodes = expand_nodelist(args.nodelist)
    if not nodes:
        return None
    try:
        start_ts = datetime.fromtimestamp(args.start, tz=timezone.utc).replace(tzinfo=None)
        end_ts = datetime.fromtimestamp(args.end, tz=timezone.utc).replace(tzinfo=None)
    except (TypeError, ValueError):
        return None
    info = {
        "start_time": start_ts,
        "end_time": end_ts,
        "nodelist": nodes,
        "job_id": args.job_id or "test",
    }
    return (info, args.outdir)


def main() -> None:
    cli_result = parse_cli_test_args()
    if cli_result is not None:
        epilog_info, out_dir = cli_result
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    else:
        epilog_info = parse_epilog_env()
        if epilog_info is None:
            parser = argparse.ArgumentParser(
                description="Slurm job power query (epilog or test: --start/--end/--nodelist/--outdir)."
            )
            parser.add_argument("job_id", nargs="?")
            parser.add_argument("--start", type=int)
            parser.add_argument("--end", type=int)
            parser.add_argument("--nodelist", type=str)
            parser.add_argument("--outdir", type=str)
            parser.parse_args()
            print(
                "Run under EpilogSlurmctld (SLURM_JOB_* env) or test mode:\n"
                "  python -m src.services.slurm_power_query --start 1769805600 --end 1769806280 "
                "--nodelist rpc-97-16\n"
                "  (output defaults to output/tmp)",
                file=sys.stderr,
            )
            sys.exit(1)
        out_dir = os.environ.get("MONSTER_POWER_OUTDIR")

    start_time = epilog_info["start_time"]
    end_time = epilog_info["end_time"]
    nodelist = epilog_info["nodelist"]
    job_id = epilog_info["job_id"]

    if not out_dir:
        print("Output directory not set (--outdir or MONSTER_POWER_OUTDIR).", file=sys.stderr)
        sys.exit(1)
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)
    if not out_dir.is_dir():
        print("Output directory not writable: %s" % out_dir, file=sys.stderr)
        sys.exit(1)
    # Write directly under out_dir (POWER_BASE) with job id prefix; no subdir.
    prefix = f"{job_id}_"
    csv_path = out_dir / f"{prefix}raw_power.csv"
    ts_path = out_dir / f"{prefix}timeseries.png"
    pie_path = out_dir / f"{prefix}energy_pie.png"

    raw_df, pie_segments, energy_by_metric, energy_gpu_per_fqdd = run_job_power(start_time, end_time, nodelist, out_dir)

    wrote_csv = save_csv(
        raw_df,
        csv_path,
        energy_by_metric=energy_by_metric,
        pie_segments=pie_segments,
        energy_gpu_per_fqdd=energy_gpu_per_fqdd,
        is_h100=any(n.lower().startswith("rpg") for n in nodelist),
    )
    wrote_ts = plot_time_series(raw_df, ts_path)
    wrote_pie = plot_pie(pie_segments, pie_path)

    out_abs = out_dir.resolve()
    if wrote_csv or wrote_ts or wrote_pie:
        written = [f"{prefix}raw_power.csv"] if wrote_csv else []
        if wrote_ts:
            written.append(f"{prefix}timeseries.png")
        if wrote_pie:
            written.extend([f"{prefix}energy_pie.png", f"{prefix}energy_pie.pdf"])
        print(f"Saved: {out_abs}", file=sys.stderr)
        print(f"  Files: {', '.join(written)}", file=sys.stderr)
        try:
            rel = out_abs.relative_to(_REPO_ROOT)
            print(f"  (in repo: {rel})", file=sys.stderr)
        except (ValueError, AttributeError):
            print(f"  Open folder: open \"{out_abs}\"", file=sys.stderr)
    else:
        print(f"No power data for job {job_id} (time range / nodelist): no CSV or plots written. Output dir: {out_abs}", file=sys.stderr)


if __name__ == "__main__":
    main()
