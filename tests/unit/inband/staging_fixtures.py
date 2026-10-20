"""Build staged in-band node directories through the real chunk, seal, and stager code."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from inband.chunks import ChunkWriter, write_seal
from inband.collectors.rapl import COUNTER_COLUMNS
from inband.stager import StageConfig, run_stager
from inband.storage import write_json_file

T0 = datetime(2026, 10, 8, 22, 0, 0, tzinfo=timezone.utc)
Domain = Tuple[str, str, str, float]  # path, name, group, constant watts

H100_DOMAINS: Sequence[Domain] = (
    ("intel-rapl:0", "package-0", "cpu_package", 100.0),
    ("intel-rapl:0/intel-rapl:0:0", "dram", "host_dram", 10.0),
    ("intel-rapl:1", "psys", "platform", 1000.0),
    ("intel-rapl:2", "package-1", "cpu_package", 120.0),
    ("intel-rapl:2/intel-rapl:2:0", "dram", "host_dram", 12.0),
)
ZEN4_DOMAINS: Sequence[Domain] = (
    ("intel-rapl:0", "package-0", "cpu_package", 150.0),
    ("intel-rapl:0/intel-rapl:0:0", "core", "core", 5.0),
    ("intel-rapl:1", "package-1", "cpu_package", 160.0),
    ("intel-rapl:1/intel-rapl:1:0", "core", "core", 6.0),
)


GPU_DOMAINS: Sequence[Domain] = tuple(
    (f"GPU-0000000{index}-test", f"gpu{index}", "gpu", watts) for index, watts in enumerate((60.0, 110.0, 380.0, 390.0))
)


def _write_counter_chunks(
    spool: Path, collector: str, node: str, domains: Sequence[Domain], seconds: int, ambiguous: Iterable[str]
) -> None:
    """Write `seconds` one-second samples at constant power, plus the collector manifest."""
    ambiguous = set(ambiguous)
    clock = {"now": 0.0}
    writer = ChunkWriter(spool / "chunks" / collector, COUNTER_COLUMNS, chunk_seconds=3, clock=lambda: clock["now"])
    def row(second: int, path: str, name: str, group: str, watts: float):
        flagged = path in ambiguous and second == 5
        # RAPL keeps a delta on an ambiguous wrap; NVML writes none after a counter reset.
        no_delta = second == 0 or (flagged and collector == "nvml")
        return {
            "timestamp": (T0 + timedelta(seconds=second)).isoformat(),
            "elapsed_seconds": second,
            "hostname": node,
            "profile": "test",
            "domain_path": path,
            "domain_name": name,
            "domain_group": group,
            "energy_uj": int(watts * 1_000_000 * second),
            "delta_uj": "" if no_delta else int(watts * 1_000_000),
            "interval_seconds": "" if second == 0 else 1.0,
            "power_watts": "" if no_delta else watts,
            "wrapped": False,
            "wrap_ambiguous": flagged,
        }

    for second in range(seconds + 1):
        clock["now"] = float(second)
        writer.write_rows([row(second, *domain) for domain in domains])
    writer.close()
    write_json_file(
        spool / f"{collector}_manifest.json",
        {
            "status": "ready",
            "reason": "",
            "profile": "test",
            "domains": [{"path": path, "name": name, "group": group, "status": "ok"} for path, name, group, _ in domains],
        },
    )


def stage_node(
    staging_job_dir: Path,
    spool_root: Path,
    node: str,
    domains: Sequence[Domain] = H100_DOMAINS,
    seconds: int = 10,
    ambiguous: Iterable[str] = (),
    attempt_id: str = "repacss-777-100",
    gpu_domains: Sequence[Domain] = (),
) -> Path:
    """Publish RAPL samples, and NVML samples when `gpu_domains` is given, for one node."""
    spool = spool_root / node
    _write_counter_chunks(spool, "rapl", node, domains, seconds, ambiguous)
    if gpu_domains:
        _write_counter_chunks(spool, "nvml", node, gpu_domains, seconds, ambiguous)
    write_json_file(spool / "runner_status.json", {"state": "complete"})
    write_seal(spool, job_id="777", storage_key=attempt_id, hostname=node, runner_state="complete")
    assert run_stager(StageConfig(spool_dir=spool, staging_dir=staging_job_dir / node, wait_seconds=0)) == 0
    return staging_job_dir / node
