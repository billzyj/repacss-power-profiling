"""NVIDIA GPU collector backed by `nvidia-smi`.

Power_Profiler uses NVML for very high-frequency sampling. For REPACSS P3 we keep the
runtime dependency lighter and use `nvidia-smi`, which is sufficient for the default
1-second interval and easier to deploy on remote cluster nodes.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List

from inband.collectors.base import CollectorProbe, PollingCSVCollector

NVIDIA_FIELDS = [
    "index",
    "uuid",
    "name",
    "power.draw",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "clocks.sm",
    "clocks.mem",
    "memory.used",
]


def _parse_csv_line(line: str) -> Dict[str, Any]:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != len(NVIDIA_FIELDS):
        raise ValueError(f"Unexpected nvidia-smi row: {line}")
    return {
        "gpu_index": parts[0],
        "gpu_uuid": parts[1],
        "gpu_name": parts[2],
        "power_watts": float(parts[3]) if parts[3] not in {"", "N/A", "[N/A]"} else None,
        "temperature_c": float(parts[4]) if parts[4] not in {"", "N/A", "[N/A]"} else None,
        "utilization_gpu_percent": float(parts[5]) if parts[5] not in {"", "N/A", "[N/A]"} else None,
        "utilization_memory_percent": float(parts[6]) if parts[6] not in {"", "N/A", "[N/A]"} else None,
        "sm_clock_mhz": float(parts[7]) if parts[7] not in {"", "N/A", "[N/A]"} else None,
        "mem_clock_mhz": float(parts[8]) if parts[8] not in {"", "N/A", "[N/A]"} else None,
        "memory_used_mb": float(parts[9]) if parts[9] not in {"", "N/A", "[N/A]"} else None,
    }


def query_nvidia_smi(binary: str = "nvidia-smi") -> List[Dict[str, Any]]:
    """Read one snapshot from nvidia-smi."""
    result = subprocess.run(
        [
            binary,
            f"--query-gpu={','.join(NVIDIA_FIELDS)}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    rows = []
    for line in result.stdout.splitlines():
        if line.strip():
            rows.append(_parse_csv_line(line))
    return rows


class NvidiaSMICollector(PollingCSVCollector):
    """In-band collector for NVIDIA GPU power and utilization."""

    name = "nvidia_smi"

    def __init__(self, binary: str = "nvidia-smi"):
        self.binary = binary

    def probe(self) -> CollectorProbe:
        binary_path = shutil.which(self.binary)
        if binary_path is None:
            return CollectorProbe(
                name=self.name,
                available=False,
                reason="nvidia-smi binary was not found in PATH.",
                details={"binary": self.binary},
            )
        try:
            rows = query_nvidia_smi(binary_path)
        except Exception as exc:
            return CollectorProbe(
                name=self.name,
                available=False,
                reason=f"nvidia-smi probe failed: {exc}",
                details={"binary": binary_path},
            )
        return CollectorProbe(
            name=self.name,
            available=bool(rows),
            reason="" if rows else "nvidia-smi reported no GPUs.",
            details={"binary": binary_path, "gpu_count": len(rows)},
        )

    def csv_columns(self) -> List[str]:
        return [
            "timestamp",
            "elapsed_seconds",
            "hostname",
            "gpu_index",
            "gpu_uuid",
            "gpu_name",
            "power_watts",
            "temperature_c",
            "utilization_gpu_percent",
            "utilization_memory_percent",
            "sm_clock_mhz",
            "mem_clock_mhz",
            "memory_used_mb",
        ]

    def prepare_runtime(self) -> Dict[str, Any]:
        binary_path = shutil.which(self.binary) or self.binary
        return {"binary": binary_path}

    def collect_rows(self, runtime: Dict[str, Any], elapsed_seconds: float) -> List[Dict[str, Any]]:
        timestamp = datetime.now(timezone.utc).isoformat()
        hostname = self.hostname()
        rows = []
        for gpu_row in query_nvidia_smi(runtime["binary"]):
            rows.append(
                {
                    "timestamp": timestamp,
                    "elapsed_seconds": round(elapsed_seconds, 6),
                    "hostname": hostname,
                    **gpu_row,
                }
            )
        return rows
