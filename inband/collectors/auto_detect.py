"""Collector registry and auto-detection helpers."""

from __future__ import annotations

from typing import Iterable, List, Optional

from inband.collectors.base import Collector, CollectorProbe
from inband.collectors.nvidia_smi import NvidiaSMICollector
from inband.collectors.nvml import NVMLCollector
from inband.collectors.rapl import RAPLCollector
from inband.collectors.rocm_smi import ROCMSMICollector

# Collectors used when a request names none. GPU energy comes from NVML; nvidia_smi
# adds telemetry such as temperature and clocks and runs only when requested.
COLLECTOR_ORDER = ["rapl", "nvml", "rocm_smi"]
COLLECTOR_TYPES = {
    "rapl": RAPLCollector,
    "nvml": NVMLCollector,
    "nvidia_smi": NvidiaSMICollector,
    "rocm_smi": ROCMSMICollector,
}


def build_collectors(
    names: Optional[Iterable[str]] = None,
    *,
    partition: str = "",
    node: str = "",
) -> List[Collector]:
    """Instantiate collectors in the configured order.

    RAPL selects its audited profile from the allocated partition and Slurm node name;
    without them it reports itself unavailable instead of guessing.
    """
    requested = list(names) if names is not None else COLLECTOR_ORDER
    collectors: List[Collector] = []
    for name in requested:
        if name not in COLLECTOR_TYPES:
            raise ValueError(f"Unknown collector: {name}")
        if name == "rapl":
            collectors.append(RAPLCollector(partition=partition, node=node))
        else:
            collectors.append(COLLECTOR_TYPES[name]())
    return collectors


def probe_collectors(
    names: Optional[Iterable[str]] = None,
    *,
    partition: str = "",
    node: str = "",
) -> List[CollectorProbe]:
    """Run availability probes without starting collection."""
    return [collector.probe() for collector in build_collectors(names, partition=partition, node=node)]


def select_collectors(
    names: Optional[Iterable[str]] = None,
    *,
    partition: str = "",
    node: str = "",
) -> List[Collector]:
    """Return the subset of requested collectors that are currently available."""
    available: List[Collector] = []
    for collector in build_collectors(names, partition=partition, node=node):
        if collector.is_available():
            available.append(collector)
    return available
