"""Rack-level power comparison helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import pandas as pd

from shared.constants.nodes import get_rack_compute_nodes, get_rack_pdu_nodes

PowerResult = Mapping[str, Tuple[pd.DataFrame, Mapping[str, float]]]


def sum_energy_kwh(results: PowerResult) -> float:
    """Sum all metric energy values in a multi-node power result."""
    total = 0.0
    for _hostname, (_df, energy_by_metric) in results.items():
        for energy_kwh in energy_by_metric.values():
            total += float(energy_kwh)
    return total


def compare_rack_compute_to_pdu(
    rack_number: int,
    compute_results: PowerResult,
    pdu_results: PowerResult,
) -> Dict[str, Any]:
    """Compare combined compute energy with combined PDU energy for one rack."""
    compute_total_kwh = sum_energy_kwh(compute_results)
    pdu_total_kwh = sum_energy_kwh(pdu_results)
    difference_kwh = compute_total_kwh - pdu_total_kwh
    percentage_difference = (difference_kwh / pdu_total_kwh * 100.0) if pdu_total_kwh else None

    return {
        "rack_number": rack_number,
        "compute_nodes": get_rack_compute_nodes(rack_number),
        "pdu_nodes": get_rack_pdu_nodes(rack_number),
        "compute_total_kwh": compute_total_kwh,
        "pdu_total_kwh": pdu_total_kwh,
        "difference_kwh": difference_kwh,
        "percentage_difference": percentage_difference,
        "compute_node_count": len(compute_results),
        "pdu_node_count": len(pdu_results),
    }
