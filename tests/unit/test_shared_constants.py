from __future__ import annotations

from src.constants import metrics as src_metrics
from src.constants import nodes as src_nodes
from shared.constants import metrics as shared_metrics
from shared.constants import nodes as shared_nodes


def test_shared_node_constants_match_legacy_src_constants() -> None:
    names = [
        "IRC_NODES",
        "PDU_NODES",
        "RACK_91_COMPUTE_NODES",
        "RACK_91_PDU_NODES",
        "RACK_92_COMPUTE_NODES",
        "RACK_92_PDU_NODES",
        "RACK_93_COMPUTE_NODES",
        "RACK_93_PDU_NODES",
        "RACK_94_COMPUTE_NODES",
        "RACK_94_PDU_NODES",
        "RACK_95_COMPUTE_NODES",
        "RACK_95_PDU_NODES",
        "RACK_96_COMPUTE_NODES",
        "RACK_96_PDU_NODES",
        "RACK_97_COMPUTE_NODES",
        "RACK_97_PDU_NODES",
    ]
    for name in names:
        assert getattr(shared_nodes, name) == getattr(src_nodes, name)


def test_shared_metric_constants_match_legacy_src_constants() -> None:
    names = [
        "IRC_POWER_METRICS",
        "IRC_ALL_METRICS",
        "IRC_SENSING_FREQUENCY",
        "PDU_POWER_METRICS",
        "PDU_SENSING_FREQUENCY",
        "COMPUTE_SENSING_FREQUENCY",
        "EXCLUDED_METRICS",
        "DERIVED_METRICS",
        "H100_METRICS",
        "ZEN4_METRICS",
    ]
    for name in names:
        assert getattr(shared_metrics, name) == getattr(src_metrics, name)


def test_rack_97_shared_config_has_compute_and_pdu_groups() -> None:
    assert len(shared_nodes.RACK_CONFIGS[97]["compute_nodes"]) == 20
    assert len(shared_nodes.RACK_CONFIGS[97]["pdu_nodes"]) == 4
    assert shared_nodes.get_rack_nodes(97) == (
        shared_nodes.RACK_97_COMPUTE_NODES + shared_nodes.RACK_97_PDU_NODES
    )
    assert shared_nodes.RACK_91_PD_NODES == shared_nodes.RACK_91_PDU_NODES
    assert src_nodes.RACK_91_PD_NODES == src_nodes.RACK_91_PDU_NODES
