"""Unit tests for node detection utilities."""

from src.utils.node_detection import get_node_type_and_query_func, get_node_power_metrics
from src.constants.metrics import IRC_POWER_METRICS, PDU_POWER_METRICS


class TestNodeDetection:
    """Test node type detection and classification."""

    def test_h100_node_detection(self):
        """Test detection of H100 compute nodes."""
        node_type, query_func, database, schema = get_node_type_and_query_func("rpg-93-1")

        assert node_type == "compute"
        assert database == "h100"
        assert schema == "idrac"
        assert callable(query_func)

    def test_zen4_node_detection(self):
        """Test detection of ZEN4 compute nodes."""
        node_type, query_func, database, schema = get_node_type_and_query_func("rpc-91-1")

        assert node_type == "compute"
        assert database == "zen4"
        assert schema == "idrac"
        assert callable(query_func)

    def test_irc_node_detection(self):
        """Test detection of IRC infrastructure nodes."""
        node_type, query_func, database, schema = get_node_type_and_query_func("irc-91-5")

        assert node_type == "irc"
        assert database == "infra"
        assert schema == "irc"
        assert callable(query_func)

    def test_pdu_node_detection(self):
        """Test detection of PDU infrastructure nodes."""
        node_type, query_func, database, schema = get_node_type_and_query_func("pdu-91-1")

        assert node_type == "pdu"
        assert database == "infra"
        assert schema == "pdu"
        assert callable(query_func)

    def test_unknown_node_defaults_to_h100(self):
        """Test that unknown nodes default to H100 compute."""
        node_type, query_func, database, schema = get_node_type_and_query_func("unknown-node")

        assert node_type == "compute"
        assert database == "h100"
        assert schema == "idrac"
        assert callable(query_func)

    def test_get_node_power_metrics(self):
        """Infrastructure nodes use static lists; compute metrics are DB-discovered."""
        assert get_node_power_metrics("rpg-93-1") == []
        assert get_node_power_metrics("irc-91-5") == IRC_POWER_METRICS
        assert get_node_power_metrics("pdu-91-1") == PDU_POWER_METRICS
