"""RAPL-based CPU power collector.

Intel and AMD expose similar energy-counter semantics, but not the same discovery path:

- Intel commonly exposes domains through Linux powercap sysfs under paths such as
  `/sys/class/powercap/intel-rapl:*` with stable `energy_uj` files.
- AMD support is more platform-dependent. Some systems expose powercap-style files,
  while others rely on MSR or HSMP-backed interfaces. Domain naming also differs by
  generation, commonly surfacing package/core or package/L3 counters instead of the
  Intel-style package/dram/psys mix.

This collector therefore probes powercap generically and records the detected vendor in
the output rather than hard-coding an Intel-only path contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from inband.collectors.base import CollectorProbe, PollingCSVCollector

POWER_CAP_ROOT = Path("/sys/class/powercap")
CPUINFO_PATH = Path("/proc/cpuinfo")


@dataclass(frozen=True)
class RaplDomain:
    """One readable RAPL domain discovered from sysfs."""

    name: str
    path: Path
    energy_path: Path
    max_energy_path: Optional[Path]
    max_energy_uj: Optional[int]


def detect_cpu_vendor(cpuinfo_path: Path = CPUINFO_PATH) -> str:
    """Infer CPU vendor from /proc/cpuinfo."""
    if not cpuinfo_path.exists():
        return "unknown"
    content = cpuinfo_path.read_text(encoding="utf-8", errors="ignore").lower()
    if "genuineintel" in content or "intel" in content:
        return "intel"
    if "authenticamd" in content or "amd" in content:
        return "amd"
    return "unknown"


def discover_rapl_domains(powercap_root: Path = POWER_CAP_ROOT) -> List[RaplDomain]:
    """Discover powercap-backed RAPL domains.

    The implementation intentionally does not assume Intel-only naming so AMD systems
    with exported powercap files can reuse the same reader.
    """

    domains: List[RaplDomain] = []
    if not powercap_root.exists():
        return domains

    for energy_path in sorted(powercap_root.rglob("energy_uj")):
        domain_path = energy_path.parent
        name_path = domain_path / "name"
        if not name_path.exists():
            continue
        try:
            domain_name = name_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        max_energy_path = domain_path / "max_energy_range_uj"
        max_energy_uj: Optional[int] = None
        if max_energy_path.exists():
            try:
                max_energy_uj = int(max_energy_path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                max_energy_uj = None
        domains.append(
            RaplDomain(
                name=domain_name,
                path=domain_path,
                energy_path=energy_path,
                max_energy_path=max_energy_path if max_energy_path.exists() else None,
                max_energy_uj=max_energy_uj,
            )
        )
    return domains


class RAPLCollector(PollingCSVCollector):
    """In-band collector that samples RAPL energy counters via sysfs."""

    name = "rapl"

    def __init__(self, powercap_root: Path = POWER_CAP_ROOT, cpuinfo_path: Path = CPUINFO_PATH):
        self.powercap_root = powercap_root
        self.cpuinfo_path = cpuinfo_path

    def probe(self) -> CollectorProbe:
        vendor = detect_cpu_vendor(self.cpuinfo_path)
        domains = discover_rapl_domains(self.powercap_root)
        if not domains:
            reason = "No powercap RAPL domains were discovered."
            if vendor == "amd":
                reason += " AMD platforms may expose RAPL through MSR or HSMP instead of powercap sysfs."
            return CollectorProbe(
                name=self.name,
                available=False,
                reason=reason,
                details={"vendor": vendor, "powercap_root": str(self.powercap_root)},
            )
        return CollectorProbe(
            name=self.name,
            available=True,
            reason="",
            details={
                "vendor": vendor,
                "domain_count": len(domains),
                "domains": [domain.name for domain in domains],
                "powercap_root": str(self.powercap_root),
            },
        )

    def csv_columns(self) -> List[str]:
        return [
            "timestamp",
            "elapsed_seconds",
            "hostname",
            "vendor",
            "domain",
            "domain_path",
            "energy_uj",
            "energy_j",
            "power_watts",
        ]

    def prepare_runtime(self) -> Dict[str, Any]:
        domains = discover_rapl_domains(self.powercap_root)
        return {
            "vendor": detect_cpu_vendor(self.cpuinfo_path),
            "domains": domains,
            "previous_energy_uj": {},
            "previous_timestamp": {},
        }

    def collect_rows(self, runtime: Dict[str, Any], elapsed_seconds: float) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        timestamp = datetime.now(timezone.utc).isoformat()
        vendor = runtime["vendor"]
        hostname = self.hostname()
        for domain in runtime["domains"]:
            try:
                energy_uj = int(domain.energy_path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue

            previous_energy = runtime["previous_energy_uj"].get(domain.name)
            previous_elapsed = runtime["previous_timestamp"].get(domain.name)
            power_watts = None
            if previous_energy is not None and previous_elapsed is not None:
                energy_diff = energy_uj - previous_energy
                if energy_diff < 0 and domain.max_energy_uj:
                    energy_diff += domain.max_energy_uj
                time_diff = elapsed_seconds - previous_elapsed
                if time_diff > 0:
                    power_watts = round((energy_diff / 1_000_000.0) / time_diff, 6)

            rows.append(
                {
                    "timestamp": timestamp,
                    "elapsed_seconds": round(elapsed_seconds, 6),
                    "hostname": hostname,
                    "vendor": vendor,
                    "domain": domain.name,
                    "domain_path": str(domain.path),
                    "energy_uj": energy_uj,
                    "energy_j": round(energy_uj / 1_000_000.0, 6),
                    "power_watts": power_watts,
                }
            )
            runtime["previous_energy_uj"][domain.name] = energy_uj
            runtime["previous_timestamp"][domain.name] = elapsed_seconds
        return rows
