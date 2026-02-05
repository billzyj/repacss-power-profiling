"""
Paper-style plotting: fonts, color palette, and smart pie chart labels.
Compatible with query-derived energy data (e.g. slurm_power_query pie).
"""

from typing import Any, Dict, List, Optional, Tuple

# Display name -> hex (colorblind-friendly, harmonious; matches ring chart style)
POWER_DISTRIBUTION_COLORS: Dict[str, str] = {
    "GPU": "#83af40",
    "CPU": "#077fbb",
    "Memory": "#5cbee3",
    "Storage": "#884595",
    "Others": "#919191",
    "Fan": "#a8273d",
    "PSU loss": "#d86723",
    "Input": "#95a5a6",
    "Output": "#7f8c8d",
}

# Four green-shade colors for GPU FQDDs (time series: one per GPU slot, distinguishable)
GPU_FQDD_COLORS: List[str] = [
    "#2d6a2d",  # dark green
    "#83af40",  # main GPU green
    "#9bc958",  # light green
    "#c5e89c",  # pale green
]

# Metric ID from DB -> display label for pie
METRIC_ID_TO_DISPLAY: Dict[str, str] = {
    "TotalCPUPower": "CPU",
    "TotalMemoryPower": "Memory",
    "TotalStoragePower": "Storage",
    "TotalFanPower": "Fan",
    "PowerConsumption": "GPU",
    "SystemInputPower": "Input",
    "SystemOutputPower": "Output",
}


def apply_paper_style() -> None:
    """Set matplotlib rc for paper figures (font, weight). Prefer Palatino Linotype, fallback to DejaVu Sans etc."""
    import matplotlib
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Palatino Linotype", "DejaVu Sans", "Liberation Sans", "sans-serif"]
    matplotlib.rcParams["font.weight"] = "bold"


def metric_to_display(metric_id: str) -> str:
    """Map DB metric id to display label."""
    return METRIC_ID_TO_DISPLAY.get(metric_id, metric_id)


def create_ring_with_smart_labels(
    ax: Any,
    values: List[float],
    labels: List[str],
    colors: List[str],
    title: str,
    center_title: str,
    center_value: str,
    startangle: float = 90,
) -> Tuple[Any, Any, Any]:
    """
    Ring (donut) chart with smart label placement and center text.
    - Ring: wedgeprops width=0.5; labels at ring center (0.7) or outside with line for small (<3%).
    - Center: center_title (e.g. Job id) and center_value (e.g. total kWh).
    """
    import numpy as np

    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=startangle,
        colors=colors,
        textprops={"fontsize": 12, "weight": "bold"},
        pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=1.5),
    )

    total = sum(values)
    ring_center = 0.7
    for wedge, autotext, value in zip(wedges, autotexts, values):
        pct = (value / total) * 100 if total else 0
        angle = (wedge.theta2 + wedge.theta1) / 2
        angle_rad = np.deg2rad(angle)

        if pct < 3.0 and total > 0:
            line_end = 1.12
            label_distance = 1.25
            x_label = label_distance * np.cos(angle_rad)
            y_label = label_distance * np.sin(angle_rad)
            autotext.set_position((x_label, y_label))
            x_edge = 1.0 * np.cos(angle_rad)
            y_edge = 1.0 * np.sin(angle_rad)
            ax.plot(
                [x_edge, line_end * np.cos(angle_rad)],
                [y_edge, line_end * np.sin(angle_rad)],
                color="gray",
                linewidth=1.0,
                linestyle="-",
                alpha=0.6,
                zorder=0,
            )
        else:
            x = ring_center * np.cos(angle_rad)
            y = ring_center * np.sin(angle_rad)
            autotext.set_position((x, y))

    if center_title or center_value:
        ax.text(
            0, 0,
            f"{center_title}\n{center_value}".strip(),
            ha="center", va="center", fontsize=14, weight="bold",
        )
    ax.set_title(title, fontsize=13, weight="bold", pad=2)
    return wedges, texts, autotexts


def create_pie_with_smart_labels(
    ax: Any,
    values: List[float],
    labels: List[str],
    colors: List[str],
    title: str,
    startangle: float = 90,
) -> Tuple[Any, Any, Any]:
    """Legacy: ring chart with empty center (for callers that do not pass center text)."""
    return create_ring_with_smart_labels(
        ax, values, labels, colors, title,
        center_title="", center_value="",
        startangle=startangle,
    )


def set_pie_text_color(
    autotexts: List[Any],
    colors: List[str],
    values: List[float],
    labels: List[str],
) -> None:
    """White/black by brightness; Storage/Fan/small segments use fixed colors."""
    total = sum(values)
    for autotext, color, value, label in zip(autotexts, colors, values, labels):
        pct = (value / total) * 100 if total else 0
        if pct < 3.0:
            autotext.set_color("black")
        elif label == "Storage":
            autotext.set_color("black")
        elif label == "Fan":
            autotext.set_color("#fff8e7")
        else:
            hex_color = (color or "#95a5a6").lstrip("#")
            if len(hex_color) >= 6:
                rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
                brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
                autotext.set_color("white" if brightness < 128 else "black")
            else:
                autotext.set_color("black")
        autotext.set_weight("bold")
