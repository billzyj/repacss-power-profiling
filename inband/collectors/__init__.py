"""Collector registry for in-band sampling."""

from inband.collectors.auto_detect import build_collectors, probe_collectors, select_collectors
from inband.collectors.base import Collector, CollectorHandle, CollectorProbe, CollectorResult
from inband.collectors.nvidia_smi import NvidiaSMICollector
from inband.collectors.rapl import RAPLCollector
from inband.collectors.rocm_smi import ROCMSMICollector

__all__ = [
    "Collector",
    "CollectorHandle",
    "CollectorProbe",
    "CollectorResult",
    "RAPLCollector",
    "NvidiaSMICollector",
    "ROCMSMICollector",
    "build_collectors",
    "probe_collectors",
    "select_collectors",
]
