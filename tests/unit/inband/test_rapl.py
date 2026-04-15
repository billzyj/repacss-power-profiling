"""Tests for RAPL discovery and vendor handling."""

from __future__ import annotations

from pathlib import Path

from inband.collectors.rapl import RAPLCollector, detect_cpu_vendor, discover_rapl_domains


def test_detect_cpu_vendor_from_cpuinfo(tmp_path: Path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("vendor_id\t: GenuineIntel\n", encoding="utf-8")
    assert detect_cpu_vendor(cpuinfo) == "intel"

    cpuinfo.write_text("vendor_id\t: AuthenticAMD\n", encoding="utf-8")
    assert detect_cpu_vendor(cpuinfo) == "amd"


def test_discover_rapl_domains_from_powercap(tmp_path: Path):
    package_dir = tmp_path / "intel-rapl:0"
    package_dir.mkdir(parents=True)
    (package_dir / "name").write_text("package-0\n", encoding="utf-8")
    (package_dir / "energy_uj").write_text("1000000\n", encoding="utf-8")
    (package_dir / "max_energy_range_uj").write_text("262143328850\n", encoding="utf-8")

    dram_dir = package_dir / "intel-rapl:0:0"
    dram_dir.mkdir(parents=True)
    (dram_dir / "name").write_text("dram\n", encoding="utf-8")
    (dram_dir / "energy_uj").write_text("500000\n", encoding="utf-8")

    domains = discover_rapl_domains(tmp_path)
    assert [domain.name for domain in domains] == ["package-0", "dram"]
    assert domains[0].max_energy_uj == 262143328850


def test_amd_probe_reports_platform_specific_reason_when_unavailable(tmp_path: Path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("vendor_id\t: AuthenticAMD\n", encoding="utf-8")
    collector = RAPLCollector(powercap_root=tmp_path / "missing-powercap", cpuinfo_path=cpuinfo)
    probe = collector.probe()
    assert probe.available is False
    assert "HSMP" in probe.reason or "MSR" in probe.reason
