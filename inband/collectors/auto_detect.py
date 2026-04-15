"""Collector registry and auto-detection helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from inband.collectors.base import Collector, CollectorProbe
from inband.collectors.nvidia_smi import NvidiaSMICollector
from inband.collectors.rapl import RAPLCollector
from inband.collectors.rocm_smi import ROCMSMICollector

COLLECTOR_ORDER = ["rapl", "nvidia_smi", "rocm_smi"]
COLLECTOR_TYPES = {
    "rapl": RAPLCollector,
    "nvidia_smi": NvidiaSMICollector,
    "rocm_smi": ROCMSMICollector,
}


def build_collectors(names: Optional[Iterable[str]] = None) -> List[Collector]:
    """Instantiate collectors in the configured order."""
    requested = list(names) if names is not None else COLLECTOR_ORDER
    collectors: List[Collector] = []
    for name in requested:
        if name not in COLLECTOR_TYPES:
            raise ValueError(f"Unknown collector: {name}")
        collectors.append(COLLECTOR_TYPES[name]())
    return collectors


def probe_collectors(names: Optional[Iterable[str]] = None) -> List[CollectorProbe]:
    """Run availability probes without starting collection."""
    return [collector.probe() for collector in build_collectors(names)]


def select_collectors(names: Optional[Iterable[str]] = None) -> List[Collector]:
    """Return the subset of requested collectors that are currently available."""
    available: List[Collector] = []
    for collector in build_collectors(names):
        if collector.is_available():
            available.append(collector)
    return available
