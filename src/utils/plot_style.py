"""
Paper-style plotting: fonts, color palette, and smart pie chart labels.
Compatible with query-derived energy data (e.g. slurm_power_query pie).
"""

from typing import Any, Dict, List, Optional, Tuple

# Display name -> hex (colorblind-friendly, matches paper figure)
POWER_DISTRIBUTION_COLORS: Dict[str, str] = {
    "CPU": "#3498db",
    "GPU": "#e67e22",
    "Memory": "#2ecc71",
    "Storage": "#9b59b6",
    "Others": "#34495e",
    "Fan": "#e74c3c",
    "PSU loss": "#95a5a6",
    "Input": "#95a5a6",
    "Output": "#7f8c8d",
}

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
    """Set matplotlib rc for paper figures (font, weight)."""
    import matplotlib
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["Palatino Linotype", "DejaVu Sans", "sans-serif"]
    matplotlib.rcParams["font.weight"] = "bold"


def metric_to_display(metric_id: str) -> str:
    """Map DB metric id to display label."""
    return METRIC_ID_TO_DISPLAY.get(metric_id, metric_id)


def create_pie_with_smart_labels(
    ax: Any,
    values: List[float],
    labels: List[str],
    colors: List[str],
    title: str,
    startangle: float = 90,
) -> Tuple[Any, Any, Any]:
    """
    Pie chart with smart label placement:
    - Large segments (>30%): label closer to center
    - Small segments (<3%): label outside with connecting line
    """
    import numpy as np
    import matplotlib.pyplot as plt

    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=startangle,
        colors=colors,
        textprops={"fontsize": 14, "weight": "bold"},
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )

    total = sum(values)
    for wedge, autotext, value in zip(wedges, autotexts, values):
        pct = (value / total) * 100 if total else 0
        angle = (wedge.theta2 + wedge.theta1) / 2
        angle_rad = np.deg2rad(angle)

        if pct < 3.0 and total > 0:
            label_distance = 1.25
            x_label = label_distance * np.cos(angle_rad)
            y_label = label_distance * np.sin(angle_rad)
            autotext.set_position((x_label, y_label))
            x_edge = 1.0 * np.cos(angle_rad)
            y_edge = 1.0 * np.sin(angle_rad)
            ax.plot(
                [x_edge, x_label],
                [y_edge, y_label],
                color="gray",
                linewidth=1.0,
                linestyle="-",
                alpha=0.6,
                zorder=0,
            )
        else:
            if pct > 30:
                pctdistance = 0.6
            elif pct > 10:
                pctdistance = 0.75
            else:
                pctdistance = 0.85
            x = pctdistance * np.cos(angle_rad)
            y = pctdistance * np.sin(angle_rad)
            autotext.set_position((x, y))

    ax.set_title(title, fontsize=13, weight="bold", pad=5)
    return wedges, texts, autotexts


def set_pie_text_color(
    autotexts: List[Any],
    colors: List[str],
    values: List[float],
    labels: List[str],
) -> None:
    """White text on dark segments, black on light; Storage/Fan forced black."""
    for autotext, color, label in zip(autotexts, colors, labels):
        if label in ("Storage", "Fan"):
            autotext.set_color("black")
        else:
            hex_color = (color or "#95a5a6").lstrip("#")
            if len(hex_color) >= 6:
                rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
                brightness = (rgb[0] * 299 + rgb[1] * 587 + rgb[2] * 114) / 1000
                autotext.set_color("white" if brightness < 128 else "black")
            else:
                autotext.set_color("black")
        autotext.set_weight("bold")
