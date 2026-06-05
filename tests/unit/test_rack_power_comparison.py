from __future__ import annotations

import pandas as pd

from shared.analysis.rack import compare_rack_compute_to_pdu
from shared.constants.nodes import RACK_97_COMPUTE_NODES, RACK_97_PDU_NODES


def _result(hostnames, metric: str, energy_kwh: float):
    return {hostname: (pd.DataFrame(), {metric: energy_kwh}) for hostname in hostnames}


def test_rack_97_compute_combined_vs_pdu_combined_difference() -> None:
    compute_results = _result(RACK_97_COMPUTE_NODES, "SystemInputPower", 100.0)
    pdu_results = _result(RACK_97_PDU_NODES, "pdu", 525.0)

    comparison = compare_rack_compute_to_pdu(97, compute_results, pdu_results)

    assert comparison["compute_total_kwh"] == 2000.0
    assert comparison["pdu_total_kwh"] == 2100.0
    assert comparison["difference_kwh"] == -100.0
    assert round(comparison["percentage_difference"], 4) == -4.7619
    assert comparison["compute_node_count"] == 20
    assert comparison["pdu_node_count"] == 4
