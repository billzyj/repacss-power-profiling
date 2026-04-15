"""ROCm SMI collector.

This is intentionally conservative in P3: we support interface probing and best-effort JSON
sampling so the collector slot can be exercised in tests and future Slurm wiring, while
accepting that ROCm field names vary across versions.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from inband.collectors.base import CollectorProbe, PollingCSVCollector


def _first_value_matching(mapping: Dict[str, Any], fragments: Iterable[str]) -> Optional[Any]:
    lowered = {key.lower(): value for key, value in mapping.items()}
    for fragment in fragments:
        for key, value in lowered.items():
            if fragment in key:
                return value
    return None


def _coerce_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if raw in {"", "N/A"}:
        return None
    filtered = "".join(ch for ch in raw if ch.isdigit() or ch in ".-")
    if not filtered:
        return None
    try:
        return float(filtered)
    except ValueError:
        return None


def query_rocm_smi(binary: str = "rocm-smi") -> List[Dict[str, Any]]:
    """Collect one snapshot from rocm-smi JSON output."""
    result = subprocess.run(
        [binary, "--showpower", "--showtemp", "--showuse", "--json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    payload = json.loads(result.stdout or "{}")
    rows: List[Dict[str, Any]] = []
    for device_name, device_payload in payload.items():
        if not isinstance(device_payload, dict):
            continue
        rows.append(
            {
                "device": device_name,
                "power_watts": _coerce_numeric(
                    _first_value_matching(device_payload, ["average graphics package power", "power", "socket power"])
                ),
                "temperature_c": _coerce_numeric(
                    _first_value_matching(device_payload, ["temperature", "sensor edge", "junction temperature"])
                ),
                "utilization_gpu_percent": _coerce_numeric(
                    _first_value_matching(device_payload, ["gpu use", "gpu utilization"])
                ),
            }
        )
    return rows


class ROCMSMICollector(PollingCSVCollector):
    """Best-effort collector for AMD GPU telemetry through rocm-smi."""

    name = "rocm_smi"

    def __init__(self, binary: str = "rocm-smi"):
        self.binary = binary

    def probe(self) -> CollectorProbe:
        binary_path = shutil.which(self.binary)
        if binary_path is None:
            return CollectorProbe(
                name=self.name,
                available=False,
                reason="rocm-smi binary was not found in PATH.",
                details={"binary": self.binary},
            )
        try:
            rows = query_rocm_smi(binary_path)
        except Exception as exc:
            return CollectorProbe(
                name=self.name,
                available=False,
                reason=f"rocm-smi probe failed: {exc}",
                details={"binary": binary_path},
            )
        return CollectorProbe(
            name=self.name,
            available=bool(rows),
            reason="" if rows else "rocm-smi reported no AMD GPUs.",
            details={"binary": binary_path, "gpu_count": len(rows)},
        )

    def csv_columns(self) -> List[str]:
        return [
            "timestamp",
            "elapsed_seconds",
            "hostname",
            "device",
            "power_watts",
            "temperature_c",
            "utilization_gpu_percent",
        ]

    def prepare_runtime(self) -> Dict[str, Any]:
        binary_path = shutil.which(self.binary) or self.binary
        return {"binary": binary_path}

    def collect_rows(self, runtime: Dict[str, Any], elapsed_seconds: float) -> List[Dict[str, Any]]:
        timestamp = datetime.now(timezone.utc).isoformat()
        hostname = self.hostname()
        rows = []
        for gpu_row in query_rocm_smi(runtime["binary"]):
            rows.append(
                {
                    "timestamp": timestamp,
                    "elapsed_seconds": round(elapsed_seconds, 6),
                    "hostname": hostname,
                    **gpu_row,
                }
            )
        return rows
