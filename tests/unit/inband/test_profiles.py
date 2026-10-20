"""Tests for audited RAPL profile loading, selection, and startup validation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from inband import profiles
from inband.profiles import (
    DEFAULT_PROFILE_PATH,
    MANIFEST_FAILED,
    MANIFEST_PARTIAL,
    MANIFEST_READY,
    MANIFEST_SKIPPED,
    RaplProfileError,
    domains_signature,
    load_profile_config,
    resolve_manifest,
)


def _write_config(tmp_path: Path, mutate) -> Path:
    raw = json.loads(DEFAULT_PROFILE_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_packaged_profiles_map_partitions_and_index_positions(rapl_config):
    assert rapl_config.partitions["zen4"].profile_id == "zen4_rapl"
    assert rapl_config.partitions["h100"].profile_id == "h100_host_rapl"
    assert rapl_config.partitions["h100-build"].profile_id == "h100_host_rapl"
    assert "rpc-92-1" in rapl_config.partitions["zen4"].verified_nodes

    # The same index means different domains in the two profiles.
    zen4 = {domain.path: domain for domain in rapl_config.profiles["zen4_rapl"].domains}
    h100 = {domain.path: domain for domain in rapl_config.profiles["h100_host_rapl"].domains}
    assert zen4["intel-rapl:1"].name == "package-1"
    assert h100["intel-rapl:1"].name == "psys"
    assert h100["intel-rapl:1"].group == "platform"
    assert h100["intel-rapl:2"].group == "cpu_package"
    assert zen4["intel-rapl:1/intel-rapl:1:0"].group == "core"


def test_edited_domains_fail_the_signature_check(tmp_path: Path):
    def mutate(raw):
        raw["profiles"]["h100_host_rapl"]["domains"][2]["name"] = "package-1"

    with pytest.raises(RaplProfileError, match="signature"):
        load_profile_config(_write_config(tmp_path, mutate))


def test_domain_paths_must_stay_under_the_root(tmp_path: Path):
    def mutate(raw):
        profile = raw["profiles"]["zen4_rapl"]
        profile["domains"][0]["path"] = "../escape"
        profile["signature"] = domains_signature(profile["domains"])

    with pytest.raises(RaplProfileError, match="under the profile root"):
        load_profile_config(_write_config(tmp_path, mutate))


def test_h100_node_resolves_every_domain_through_a_symlinked_root(fake_powercap):
    root = fake_powercap("h100_host_rapl")
    manifest = resolve_manifest(partition="h100", node="rpg-93-1", sysfs_root=root)

    assert manifest.status == MANIFEST_READY
    assert manifest.profile_id == "h100_host_rapl"
    assert [domain.path for domain in manifest.usable_domains()] == [
        "intel-rapl:0",
        "intel-rapl:0/intel-rapl:0:0",
        "intel-rapl:1",
        "intel-rapl:2",
        "intel-rapl:2/intel-rapl:2:0",
    ]
    payload = manifest.to_dict()
    assert payload["config_sha256"]
    assert payload["profile_signature"] == manifest.profile_signature


@pytest.mark.parametrize(
    "partition,node,reason",
    [
        ("", "rpc-91-1", "missing_partition"),
        ("mi210", "rpc-91-1", "no_profile_for_partition"),
        ("zen4", "rpg-93-1", "unverified_node"),
    ],
)
def test_selection_skips_before_reading_sysfs(tmp_path: Path, partition, node, reason):
    manifest = resolve_manifest(partition=partition, node=node, sysfs_root=tmp_path / "absent")
    assert manifest.status == MANIFEST_SKIPPED
    assert manifest.reason == reason
    assert manifest.domains == ()


def test_domains_that_differ_from_the_audit_are_excluded(fake_powercap):
    root = fake_powercap("zen4_rapl")
    (root / "intel-rapl:1" / "name").write_text("psys\n", encoding="utf-8")
    (root / "intel-rapl:0/intel-rapl:0:0" / "max_energy_range_uj").write_text("1\n", encoding="utf-8")
    (root / "intel-rapl:1/intel-rapl:1:0" / "energy_uj").unlink()

    manifest = resolve_manifest(partition="zen4", node="rpc-91-1", sysfs_root=root)

    assert manifest.status == MANIFEST_PARTIAL
    assert {domain.path: domain.status for domain in manifest.domains} == {
        "intel-rapl:0": "ok",
        "intel-rapl:0/intel-rapl:0:0": "range_mismatch",
        "intel-rapl:1": "name_mismatch",
        "intel-rapl:1/intel-rapl:1:0": "missing",
    }


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file modes")
def test_unreadable_counters_fail_the_manifest(fake_powercap, rapl_config):
    root = fake_powercap("zen4_rapl")
    for domain in rapl_config.profiles["zen4_rapl"].domains:
        (root / domain.energy_path).chmod(0o000)

    manifest = resolve_manifest(partition="zen4", node="rpc-91-1", sysfs_root=root)

    assert manifest.status == MANIFEST_FAILED
    assert {domain.status for domain in manifest.domains} == {"permission_denied"}


def test_invalid_config_becomes_a_failed_manifest(tmp_path: Path):
    config = tmp_path / "bad.json"
    config.write_text("{}", encoding="utf-8")
    manifest = resolve_manifest(partition="zen4", node="rpc-91-1", config_path=config)
    assert manifest.status == MANIFEST_FAILED
    assert manifest.reason.startswith("invalid_profile_config")


def test_check_command_samples_counters_and_detects_a_wrap(fake_powercap, monkeypatch, capsys):
    root = fake_powercap("zen4_rapl")

    def advance(_seconds):
        (root / "intel-rapl:0/energy_uj").write_text("201000000\n", encoding="utf-8")
        (root / "intel-rapl:1/energy_uj").write_text("500000\n", encoding="utf-8")
        (root / "intel-rapl:0/intel-rapl:0:0/energy_uj").write_text("6000000\n", encoding="utf-8")
        (root / "intel-rapl:1/intel-rapl:1:0/energy_uj").write_text("6000000\n", encoding="utf-8")

    monkeypatch.setattr(profiles, "time", SimpleNamespace(sleep=advance, monotonic=time.monotonic))
    argv = ["--partition", "zen4", "--node", "rpc-91-1", "--root", str(root), "--sample-seconds", "1"]

    assert profiles.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    samples = {sample["path"]: sample for sample in report["samples"]}
    assert samples["intel-rapl:0"]["delta_uj"] == 200_000_000
    assert samples["intel-rapl:1"]["wrapped"] is True
    assert samples["intel-rapl:1"]["delta_uj"] == 65_532_610_987 - 1_000_000 + 500_000


def test_check_command_explains_an_unverified_node(capsys):
    assert profiles.main(["--partition", "zen4", "--node", "tabe"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["reason"] == "unverified_node"
    assert "run it on a node of that partition" in captured.err


def test_check_command_fails_when_counters_do_not_advance(fake_powercap, monkeypatch, capsys):
    root = fake_powercap("zen4_rapl")
    monkeypatch.setattr(profiles, "time", SimpleNamespace(sleep=lambda _seconds: None, monotonic=time.monotonic))
    argv = ["--partition", "zen4", "--node", "rpc-91-1", "--root", str(root), "--sample-seconds", "1"]

    assert profiles.main(argv) == 1
    report = json.loads(capsys.readouterr().out)
    assert not any(sample["advancing"] for sample in report["samples"])
