"""Build staged in-band node directories through the real sample, seal, and stager code."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from inband.spool import SampleWriter, sample_path, write_seal
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


# A RAPL range this small makes every 1 s interval long enough for an unseen wrap.
AMBIGUOUS_RANGE_UJ = 500_000_000
WIDE_RANGE_UJ = 10**15


def _write_counter_samples(
    spool: Path, collector: str, domains: Sequence[Domain], seconds: int, ambiguous: Iterable[str]
) -> None:
    """Write `seconds` one-second raw readings at constant power, plus the collector manifest.

    A domain listed in `ambiguous` gets a wrap-ambiguous RAPL range, or, for NVML, a
    counter reset at 5 s.
    """
    ambiguous = set(ambiguous)
    unit = 1_000_000 if collector == "rapl" else 1_000  # RAPL counts uJ, NVML mJ
    writer = SampleWriter(sample_path(spool, collector), [path for path, *_ in domains])

    def reading(second: int, path: str, watts: float) -> int:
        if collector == "nvml" and path in ambiguous and second >= 5:
            second -= 5  # the driver was reloaded at 5 s
        return int(watts * unit * second)

    for second in range(seconds + 1):
        epoch = (T0 + timedelta(seconds=second)).timestamp()
        writer.write(epoch, [[reading(second, path, watts) for path, _, _, watts in domains]])
    writer.close()
    write_json_file(
        spool / f"{collector}_manifest.json",
        {
            "status": "ready",
            "reason": "",
            "profile": "test",
            "counter_unit": "uJ" if collector == "rapl" else "mJ",
            "domains": [
                {
                    "path": path,
                    "name": name,
                    "group": group,
                    "status": "ok",
                    "max_energy_range_uj": AMBIGUOUS_RANGE_UJ if path in ambiguous else WIDE_RANGE_UJ,
                }
                for path, name, group, _ in domains
            ],
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
    _write_counter_samples(spool, "rapl", domains, seconds, ambiguous)
    if gpu_domains:
        _write_counter_samples(spool, "nvml", gpu_domains, seconds, ambiguous)
    write_json_file(spool / "runner_status.json", {"state": "complete"})
    write_seal(spool, job_id="777", storage_key=attempt_id, hostname=node, runner_state="complete")
    assert run_stager(StageConfig(spool_dir=spool, staging_dir=staging_job_dir / node, wait_seconds=0)) == 0
    return staging_job_dir / node
