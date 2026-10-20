"""Tests for the NVML GPU energy collector."""

from __future__ import annotations

from types import SimpleNamespace

from inband.collectors import nvml
from inband.collectors.nvml import NvmlError, NVMLCollector


class FakeNvml:
    """Stands in for NvmlLibrary; energies are cumulative millijoules per GPU index."""

    def __init__(self, energies_mj, unreadable=()):
        self.energies_mj = list(energies_mj)
        self.unreadable = set(unreadable)

    def driver_version(self):
        return "610.43.02"

    def device_count(self):
        return len(self.energies_mj)

    def handle(self, index):
        return index

    def uuid(self, handle):
        return f"GPU-{handle}"

    def name(self, handle):
        return "NVIDIA H100 NVL"

    def energy_mj(self, handle):
        if handle in self.unreadable:
            raise NvmlError("nvmlDeviceGetTotalEnergyConsumption returned NVML error 3")
        return self.energies_mj[handle]


def _clock(monkeypatch, readings):
    clock = iter(readings)
    monkeypatch.setattr(nvml, "time", SimpleNamespace(monotonic=lambda: next(clock)))


def test_rows_difference_the_energy_counter_per_gpu(monkeypatch):
    library = FakeNvml([1_000, 5_000])
    collector = NVMLCollector(library_factory=lambda: library)
    assert collector.probe().available
    _clock(monkeypatch, [10.0, 10.0, 12.0, 12.0])

    runtime = collector.prepare_runtime()
    first = collector.collect_rows(runtime, 0.0)
    library.energies_mj = [601_000, 1_405_000]  # 600 J and 1400 J later
    second = collector.collect_rows(runtime, 2.0)

    assert [row["delta_uj"] for row in first] == [None, None]
    assert [(row["domain_path"], row["domain_name"], row["domain_group"]) for row in second] == [
        ("GPU-0", "gpu0", "gpu"),
        ("GPU-1", "gpu1", "gpu"),
    ]
    assert [row["delta_uj"] for row in second] == [600_000_000, 1_400_000_000]
    assert [row["power_watts"] for row in second] == [300.0, 700.0]


def test_a_smaller_reading_is_a_counter_reset_without_energy(monkeypatch):
    library = FakeNvml([900_000])
    collector = NVMLCollector(library_factory=lambda: library)
    _clock(monkeypatch, [0.0, 1.0])
    runtime = collector.prepare_runtime()
    collector.collect_rows(runtime, 0.0)
    library.energies_mj = [50]

    (row,) = collector.collect_rows(runtime, 1.0)

    assert row["delta_uj"] is None
    assert row["wrap_ambiguous"] is True


def test_unreadable_gpus_are_recorded_and_skipped():
    collector = NVMLCollector(library_factory=lambda: FakeNvml([1, 2], unreadable={1}))

    manifest = collector.manifest_payload()

    assert manifest["status"] == "partial"
    assert [(domain["path"], domain["status"]) for domain in manifest["domains"]] == [
        ("GPU-0", "ok"),
        ("GPU-1", "unreadable"),
    ]
    assert [device["uuid"] for device in collector.prepare_runtime()["devices"]] == ["GPU-0"]


def test_a_node_without_nvml_reports_the_collector_unavailable():
    def missing_library():
        raise OSError("libnvidia-ml.so.1: cannot open shared object file")

    collector = NVMLCollector(library_factory=missing_library)

    probe = collector.probe()
    assert probe.available is False
    assert "libnvidia-ml.so.1" in probe.reason
    assert collector.manifest_payload()["status"] == "unavailable"
