"""Plots of an in-band (IB) job report, in the style of the OOB report's plots.

The finalizer calls `plot_report` once `raw_power.csv` is written. Plots are a
convenience: a failure becomes a note in the summary and never fails the export.

    power_timeseries.pdf          power over time by component: CPU packages, host DRAM,
                                  and each GPU for one node; each component summed over
                                  the nodes for several
    power_timeseries_by_node.pdf  each node's measured total over time (several nodes only)
    energy_ring.pdf               the job's energy by component (two components or more)

Only the groups that enter job totals are drawn, so `psys` and `core` are left out. Each
series is cut into at most MAX_POINTS time bins of at least MIN_BIN_SECONDS, and a bin's
value is its energy divided by the seconds its intervals cover, so a long job at 100 ms
still plots quickly and the plotted energy matches the CSV. The NVML energy counter of an
H100 advances in 100 ms steps, so a 100 ms bin would show each step as a spike to zero or
to twice the power; `raw_power.csv` keeps the full resolution.
"""

from __future__ import annotations

import csv
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

LOGGER = logging.getLogger("repacss_power.inband.plots")

MAX_POINTS = 1500
MIN_BIN_SECONDS = 1.0
SHORT_SPAN_SECONDS = 2 * 3600  # shorter jobs label time ticks with seconds
COMPONENTS = (("cpu_package", "CPU"), ("host_dram", "Memory"))  # GPUs get one line per device
RING_GROUPS = (("gpu", "GPU"), ("cpu_package", "CPU"), ("host_dram", "Memory"))
TOTAL_COLOR = "#000000"

Series = List[Optional[float]]
Key = Tuple[str, str]  # node, metric


def bin_power(csv_path: Path, start: float, end: float, bins: int) -> Tuple[Dict[Key, Series], Dict[Key, str]]:
    """Average power per time bin for every (node, metric) of `raw_power.csv`.

    A row's interval ends at its timestamp and lasts energy / power seconds; that is the
    part inside the job window, since `energy_j` is prorated and `power_w` is not. A row
    without power (the counter did not advance) lasts until its domain's previous row.
    """
    width = (end - start) / bins
    energy: Dict[Key, List[float]] = defaultdict(lambda: [0.0] * bins)
    seconds: Dict[Key, List[float]] = defaultdict(lambda: [0.0] * bins)
    groups: Dict[Key, str] = {}
    previous: Dict[Key, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["hostname"], row["metric"])
            t = datetime.fromisoformat(row["timestamp"]).timestamp()
            joules = float(row["energy_j"] or 0.0)
            watts = float(row["power_w"] or 0.0)
            duration = joules / watts if watts > 0 else t - previous.get(key, t)
            previous[key] = t
            index = min(bins - 1, max(0, int((t - start) / width)))
            energy[key][index] += joules
            seconds[key][index] += duration
            groups[key] = row["group"]
    power = {
        key: [joules / covered if covered > 0 else None for joules, covered in zip(energy[key], seconds[key])]
        for key in energy
    }
    return power, groups


def _add(series: Sequence[Series]) -> Series:
    """Element-wise sum; a bin is empty unless every series has it."""
    return [None if None in values else sum(values) for values in zip(*series)] if series else []


def _natural(metric: str) -> Tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", metric))


def node_components(power: Mapping[Key, Series], groups: Mapping[Key, str], node: str) -> List[Tuple[str, str, Series]]:
    """(label, group, series) for the node's CPU packages, host DRAM, and each GPU."""
    components = []
    for group, label in COMPONENTS:
        parts = [series for (host, _), series in power.items() if host == node and groups[(host, _)] == group]
        if parts:
            components.append((label, group, _add(parts)))
    gpus = sorted((metric for (host, metric) in power if host == node and groups[(host, metric)] == "gpu"), key=_natural)
    components.extend((f"GPU ({metric})", "gpu", power[(node, metric)]) for metric in gpus)
    return components


def ring_segments(nodes: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    """kWh per component, summed over the nodes that report it."""
    segments = {}
    for group, label in RING_GROUPS:
        joules = sum(result.get(f"{group}_energy_j") or 0.0 for result in nodes.values())
        if joules > 0:
            segments[label] = joules / 3.6e6
    return segments


def _style_time_axis(ax: Any, plt: Any, mdates: Any, tz: Any, span: float, ylabel: str) -> None:
    ax.set_xlabel("Time (local)", fontsize=18, weight="bold")
    ax.set_ylabel(ylabel, fontsize=18, weight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=16, frameon=True)
    fmt = "%H:%M:%S" if span < SHORT_SPAN_SECONDS else "%Y-%m-%d %H:%M"
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt, tz=tz))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=tz))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.grid(True, alpha=0.3)


def plot_report(export_dir: Path, result: Mapping[str, Any], record: Mapping[str, Any]) -> List[str]:
    """Draw the report's plots into `export_dir`; return notes for anything not drawn."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        from shared.utils.plot_style import (
            POWER_DISTRIBUTION_RING_COLORS,
            POWER_DISTRIBUTION_TIME_SERIES_COLORS,
            TIME_SERIES_GPU_FQDD_COLORS,
            apply_paper_style,
            create_ring_with_smart_labels,
            set_pie_text_color,
        )
    except ImportError as exc:
        return [f"Plots were not drawn: {exc}."]

    try:
        apply_paper_style()
        notes: List[str] = []
        start, end = record.get("start_epoch"), record.get("end_epoch")
        nodes = [node for node in record["nodes"] if result["nodes"].get(node, {}).get("domains")]
        if start is None or end is None or end <= start:
            notes.append("Power plots were not drawn: the job window is unknown.")
        elif nodes:
            interval = max((record.get("interval_ms") or 1000) / 1000.0, MIN_BIN_SECONDS)
            bins = max(1, min(MAX_POINTS, round((end - start) / interval)))
            power, groups = bin_power(export_dir / "raw_power.csv", float(start), float(end), bins)
            tz = datetime.now().astimezone().tzinfo
            width = (end - start) / bins
            times = [datetime.fromtimestamp(start + (index + 0.5) * width, tz=tz) for index in range(bins)]
            per_node = {node: node_components(power, groups, node) for node in nodes}

            def draw(path: Path, lines: Sequence[Tuple[str, str, Series]], ylabel: str, figsize: Tuple[int, int]) -> None:
                fig, ax = plt.subplots(figsize=figsize)
                for label, color, series in lines:
                    total = color == TOTAL_COLOR  # dashed, so it stands apart from the dark GPU greens
                    ax.plot(times, [float("nan") if value is None else value for value in series], label=label,
                            color=color, alpha=0.85, linewidth=2.0 if total else 1.5, linestyle="--" if total else "-")
                _style_time_axis(ax, plt, mdates, tz, end - start, ylabel)
                fig.tight_layout()
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)

            if len(nodes) == 1:
                components = per_node[nodes[0]]
                lines = []
                gpu_index = 0
                for label, group, series in components:
                    if group == "gpu":
                        color = TIME_SERIES_GPU_FQDD_COLORS[gpu_index % len(TIME_SERIES_GPU_FQDD_COLORS)]
                        gpu_index += 1
                    else:
                        color = POWER_DISTRIBUTION_TIME_SERIES_COLORS.get(label, "#95a5a6")
                    lines.append((label, color, series))
                if len(lines) > 1:  # on a CPU-only node the total is the CPU line again
                    lines.append(("IB total", TOTAL_COLOR, _add([series for _, _, series in components])))
                draw(export_dir / "power_timeseries.pdf", lines, "Power (W)", (10, 5))
            else:
                lines = []
                for group, label in (*COMPONENTS, ("gpu", "GPU")):
                    parts = [series for node in nodes for _, part_group, series in per_node[node] if part_group == group]
                    if parts:
                        color = POWER_DISTRIBUTION_RING_COLORS["GPU"] if group == "gpu" else POWER_DISTRIBUTION_TIME_SERIES_COLORS[label]
                        lines.append((f"{label} total", color, _add(parts)))
                if len(lines) > 1:
                    lines.append(("IB total", TOTAL_COLOR, _add([series for _, _, series in lines])))
                draw(export_dir / "power_timeseries.pdf", lines, "Total Power (W)", (10, 5))
                by_node = [(node, None, _add([series for _, _, series in per_node[node]])) for node in nodes]
                fig, ax = plt.subplots(figsize=(12, 6))
                for node, _, series in by_node:
                    ax.plot(times, [float("nan") if value is None else value for value in series], label=f"{node} IB total", alpha=0.75, linewidth=1.2)
                _style_time_axis(ax, plt, mdates, tz, end - start, "Power (W)")
                fig.tight_layout()
                fig.savefig(export_dir / "power_timeseries_by_node.pdf", dpi=150, bbox_inches="tight")
                plt.close(fig)

        segments = ring_segments(result["nodes"])
        if len(segments) >= 2:  # a single component would fill the whole ring
            labels = sorted(segments, key=segments.get, reverse=True)
            values = [segments[label] for label in labels]
            colors = [POWER_DISTRIBUTION_RING_COLORS.get(label, "#95a5a6") for label in labels]
            fig, ax = plt.subplots(figsize=(10, 8))
            # Center the largest component under the title, so small ones sit at the bottom.
            wedges, _, autotexts = create_ring_with_smart_labels(
                ax, values, labels, colors, "In-band Energy (kWh)",
                center_title=f"Job {record['job_id']}", center_value=f"{sum(values):.3f} kWh",
                startangle=90 - 180 * values[0] / sum(values),
            )
            set_pie_text_color(autotexts, colors, values, labels)
            ax.legend(
                wedges, [f"{label}: {value:.3f} kWh" for label, value in zip(labels, values)],
                title="Components (RAPL + NVML)", loc="center left", bbox_to_anchor=(1.02, 0.5),
                fontsize=12, title_fontsize=13, frameon=True,
            )
            fig.savefig(export_dir / "energy_ring.pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)
        return notes
    except Exception as exc:  # plots never fail an export
        LOGGER.exception("job %s: plots failed", record.get("job_id"))
        plt.close("all")
        return [f"Plots were not drawn: {exc}."]
