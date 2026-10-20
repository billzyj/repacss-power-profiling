"""Shared fixtures for in-band collector tests."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from inband.profiles import DEFAULT_PROFILE_PATH, RaplProfileConfig, load_profile_config


@pytest.fixture(scope="session")
def rapl_config() -> RaplProfileConfig:
    return load_profile_config(DEFAULT_PROFILE_PATH)


@pytest.fixture
def fake_powercap(tmp_path: Path, rapl_config: RaplProfileConfig) -> Callable[[str], Path]:
    """Build a powercap tree for one audited profile, with every counter at 1 J.

    As on a real node, the returned root is a symlink to the device directory.
    """

    def build(profile_id: str) -> Path:
        device_root = tmp_path / profile_id / "devices" / "intel-rapl"
        for domain in rapl_config.profiles[profile_id].domains:
            domain_dir = device_root / domain.path
            domain_dir.mkdir(parents=True, exist_ok=True)
            (domain_dir / "name").write_text(f"{domain.name}\n", encoding="utf-8")
            (domain_dir / "max_energy_range_uj").write_text(f"{domain.max_energy_range_uj}\n", encoding="utf-8")
            (device_root / domain.energy_path).write_text("1000000\n", encoding="utf-8")
        class_root = tmp_path / profile_id / "class" / "intel-rapl"
        class_root.parent.mkdir(parents=True)
        class_root.symlink_to(device_root, target_is_directory=True)
        return class_root

    return build
