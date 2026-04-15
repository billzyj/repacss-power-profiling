"""Tests for in-band collector registry helpers."""

from inband.collectors.auto_detect import build_collectors
from inband.collectors.nvidia_smi import NvidiaSMICollector
from inband.collectors.rapl import RAPLCollector
from inband.collectors.rocm_smi import ROCMSMICollector


def test_build_collectors_uses_requested_order():
    collectors = build_collectors(["nvidia_smi", "rapl", "rocm_smi"])
    assert isinstance(collectors[0], NvidiaSMICollector)
    assert isinstance(collectors[1], RAPLCollector)
    assert isinstance(collectors[2], ROCMSMICollector)
