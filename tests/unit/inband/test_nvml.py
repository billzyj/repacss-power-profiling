"""Tests for the NVML GPU energy collector."""

from __future__ import annotations

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


def test_readings_are_the_raw_millijoule_counters_keyed_by_uuid():
    library = FakeNvml([1_000, 5_000])
    collector = NVMLCollector(library_factory=lambda: library)
    assert collector.probe().available
    assert collector.csv_columns() == ["GPU-0", "GPU-1"]
    assert collector.manifest_payload()["counter_unit"] == "mJ"

    runtime = collector.prepare_runtime()
    assert collector.collect_rows(runtime) == [{"GPU-0": 1_000, "GPU-1": 5_000}]


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
