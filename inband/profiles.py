"""Audited RAPL profile loading, selection, and startup validation.

`rapl_profiles.json` records the powercap domain structure observed on every audited
node. The runner selects a profile by the allocated Slurm partition, requires the
node to be in that partition's verified list, and checks each configured domain once
at startup. The resulting manifest is the only domain list the RAPL collector samples.

Domains are identified by their path relative to the profile root. Names repeat
across sockets (`dram`, `core`), and index positions differ between profiles
(`intel-rapl:1` is `package-1` on Zen4 but `psys` on H100).

Run as a module for an on-node check that needs only the standard library:

    python3 -m inband.profiles --partition zen4 --sample-seconds 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from shared.errors import InbandCollectorError

DEFAULT_PROFILE_PATH = Path(__file__).resolve().with_name("rapl_profiles.json")
PROFILE_PATH_ENV = "REPACSS_POWER_RAPL_PROFILES"
CONFIG_KIND = "repacss_rapl_profiles"
SUPPORTED_SCHEMA_VERSION = 1

MANIFEST_READY = "ready"
MANIFEST_PARTIAL = "partial"
MANIFEST_FAILED = "failed"
MANIFEST_SKIPPED = "skipped"

DOMAIN_OK = "ok"


class RaplProfileError(InbandCollectorError):
    """Raised when the RAPL profile configuration is malformed or inconsistent."""


@dataclass(frozen=True)
class ProfileDomain:
    """One audited powercap domain, identified by its path relative to the root."""

    path: str
    name: str
    parent: Optional[str]
    energy_path: str
    max_energy_range_uj: int
    group: str


@dataclass(frozen=True)
class RaplProfile:
    profile_id: str
    root: Path
    signature: str
    domains: Tuple[ProfileDomain, ...]


@dataclass(frozen=True)
class PartitionProfile:
    profile_id: str
    verified_nodes: FrozenSet[str]


@dataclass(frozen=True)
class RaplProfileConfig:
    path: Path
    sha256: str
    partitions: Dict[str, PartitionProfile]
    profiles: Dict[str, RaplProfile]


@dataclass(frozen=True)
class DomainCheck:
    """Startup validation result for one configured domain."""

    path: str
    name: str
    group: str
    energy_file: Path
    max_energy_range_uj: int
    status: str
    message: str = ""

    @property
    def usable(self) -> bool:
        return self.status == DOMAIN_OK


@dataclass(frozen=True)
class RaplManifest:
    """The resolved RAPL domain set for one runner, or the reason there is none."""

    status: str
    reason: str
    partition: str
    node: str
    profile_id: str = ""
    profile_signature: str = ""
    config_path: str = ""
    config_sha256: str = ""
    root: str = ""
    kernel: str = field(default_factory=platform.release)
    validated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    domains: Tuple[DomainCheck, ...] = ()

    def usable_domains(self) -> List[DomainCheck]:
        return [domain for domain in self.domains if domain.usable]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "repacss_rapl_manifest",
            "status": self.status,
            "reason": self.reason,
            "partition": self.partition,
            "node": self.node,
            "profile": self.profile_id,
            "profile_signature": self.profile_signature,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "root": self.root,
            "kernel": self.kernel,
            "validated_at": self.validated_at,
            "domains": [
                {
                    "path": domain.path,
                    "name": domain.name,
                    "group": domain.group,
                    "energy_file": str(domain.energy_file),
                    "max_energy_range_uj": domain.max_energy_range_uj,
                    "status": domain.status,
                    "message": domain.message,
                }
                for domain in self.domains
            ],
        }


def default_profile_path(env: Optional[Dict[str, str]] = None) -> Path:
    env = os.environ if env is None else env
    override = (env.get(PROFILE_PATH_ENV) or "").strip()
    return Path(override) if override else DEFAULT_PROFILE_PATH


def domains_signature(raw_domains: List[Dict[str, Any]]) -> str:
    """Reproduce the audit signature: SHA-256 of compact, key-sorted JSON."""
    payload = json.dumps(raw_domains, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_relative_path(raw: Any, field_name: str, profile_id: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise RaplProfileError(f"profile {profile_id}: {field_name} must be a non-empty string")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RaplProfileError(f"profile {profile_id}: {field_name} {raw!r} must stay under the profile root")
    return raw


def _parse_profile(profile_id: str, raw: Dict[str, Any]) -> RaplProfile:
    root = Path(str(raw.get("root") or ""))
    if not root.is_absolute():
        raise RaplProfileError(f"profile {profile_id}: root must be an absolute path")

    raw_domains = raw.get("domains")
    if not isinstance(raw_domains, list) or not raw_domains:
        raise RaplProfileError(f"profile {profile_id}: domains must be a non-empty list")
    signature = str(raw.get("signature") or "")
    if domains_signature(raw_domains) != signature:
        raise RaplProfileError(f"profile {profile_id}: domains do not match the audited signature")

    group_by_path: Dict[str, str] = {}
    for group, paths in (raw.get("domain_groups") or {}).items():
        for path in paths:
            if path in group_by_path:
                raise RaplProfileError(f"profile {profile_id}: domain {path} is in more than one group")
            group_by_path[path] = group

    domains: List[ProfileDomain] = []
    seen_paths = set()
    for item in raw_domains:
        path = _safe_relative_path(item.get("path"), "path", profile_id)
        if path in seen_paths:
            raise RaplProfileError(f"profile {profile_id}: duplicate domain path {path}")
        seen_paths.add(path)
        domains.append(
            ProfileDomain(
                path=path,
                name=str(item.get("name") or ""),
                parent=item.get("parent"),
                energy_path=_safe_relative_path(item.get("energy_path"), "energy_path", profile_id),
                max_energy_range_uj=int(item["max_energy_range_uj"]),
                group=group_by_path.get(path, ""),
            )
        )

    unknown = sorted(set(group_by_path) - seen_paths)
    if unknown:
        raise RaplProfileError(f"profile {profile_id}: domain_groups reference unknown paths {unknown}")
    return RaplProfile(profile_id=profile_id, root=root, signature=signature, domains=tuple(domains))


def load_profile_config(path: Optional[Path] = None) -> RaplProfileConfig:
    """Load and verify the audited profile file. Raises RaplProfileError on any inconsistency."""
    path = path or default_profile_path()
    try:
        content = path.read_bytes()
        raw = json.loads(content.decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise RaplProfileError(f"cannot load RAPL profiles from {path}: {exc}") from exc

    if (
        not isinstance(raw, dict)
        or raw.get("kind") != CONFIG_KIND
        or raw.get("schema_version") != SUPPORTED_SCHEMA_VERSION
    ):
        raise RaplProfileError(f"{path} is not a schema {SUPPORTED_SCHEMA_VERSION} {CONFIG_KIND} file")
    if raw.get("domain_identity") != "path":
        raise RaplProfileError(f"{path}: domain_identity must be 'path'")

    try:
        profiles = {profile_id: _parse_profile(profile_id, item) for profile_id, item in raw["profiles"].items()}
        partitions: Dict[str, PartitionProfile] = {}
        for partition, item in raw["partitions"].items():
            profile_id = item["rapl_profile"]
            if profile_id not in profiles:
                raise RaplProfileError(f"partition {partition} references unknown profile {profile_id}")
            partitions[partition] = PartitionProfile(
                profile_id=profile_id,
                verified_nodes=frozenset(item.get("verified_nodes") or ()),
            )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RaplProfileError(f"{path} is missing or has malformed field: {exc}") from exc

    return RaplProfileConfig(
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        partitions=partitions,
        profiles=profiles,
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def check_domain(root: Path, domain: ProfileDomain) -> DomainCheck:
    """Confirm that one configured domain still matches the audit and is readable."""
    domain_dir = root / domain.path
    energy_file = root / domain.energy_path

    def result(status: str, message: str = "") -> DomainCheck:
        return DomainCheck(
            path=domain.path,
            name=domain.name,
            group=domain.group,
            energy_file=energy_file,
            max_energy_range_uj=domain.max_energy_range_uj,
            status=status,
            message=message,
        )

    try:
        name = _read_text(domain_dir / "name")
        max_range = int(_read_text(domain_dir / "max_energy_range_uj"))
    except FileNotFoundError as exc:
        return result("missing", str(exc))
    except (OSError, ValueError) as exc:
        return result("unreadable", str(exc))
    if name != domain.name:
        return result("name_mismatch", f"expected {domain.name!r}, found {name!r}")
    if max_range != domain.max_energy_range_uj:
        return result("range_mismatch", f"expected {domain.max_energy_range_uj}, found {max_range}")

    try:
        int(_read_text(energy_file))
    except PermissionError as exc:
        return result("permission_denied", str(exc))
    except FileNotFoundError as exc:
        return result("missing", str(exc))
    except (OSError, ValueError) as exc:
        return result("unreadable", str(exc))
    return result(DOMAIN_OK)


def resolve_manifest(
    *,
    partition: str,
    node: str,
    config_path: Optional[Path] = None,
    sysfs_root: Optional[Path] = None,
) -> RaplManifest:
    """Select and validate the profile for this node. Never raises; failures become the manifest status.

    `sysfs_root` replaces the profile root and exists for tests and offline checks.
    """
    partition = (partition or "").strip()
    node = (node or "").strip()
    if not partition:
        return RaplManifest(status=MANIFEST_SKIPPED, reason="missing_partition", partition=partition, node=node)

    try:
        config = load_profile_config(config_path)
    except RaplProfileError as exc:
        return RaplManifest(
            status=MANIFEST_FAILED,
            reason=f"invalid_profile_config: {exc}",
            partition=partition,
            node=node,
            config_path=str(config_path or default_profile_path()),
        )

    provenance = {"config_path": str(config.path), "config_sha256": config.sha256}
    selected = config.partitions.get(partition)
    if selected is None:
        return RaplManifest(
            status=MANIFEST_SKIPPED, reason="no_profile_for_partition", partition=partition, node=node, **provenance
        )
    if node not in selected.verified_nodes:
        return RaplManifest(status=MANIFEST_SKIPPED, reason="unverified_node", partition=partition, node=node, **provenance)

    profile = config.profiles[selected.profile_id]
    root = sysfs_root or profile.root
    checks = tuple(check_domain(root, domain) for domain in profile.domains)
    usable = sum(1 for check in checks if check.usable)
    if usable == len(checks):
        status, reason = MANIFEST_READY, ""
    elif usable:
        status, reason = MANIFEST_PARTIAL, "domain_validation_failed"
    else:
        status, reason = MANIFEST_FAILED, "domain_validation_failed"
    return RaplManifest(
        status=status,
        reason=reason,
        partition=partition,
        node=node,
        profile_id=profile.profile_id,
        profile_signature=profile.signature,
        root=str(root),
        domains=checks,
        **provenance,
    )


def _sample_once(manifest: RaplManifest, seconds: float) -> List[Dict[str, Any]]:
    from inband.collectors.rapl import counter_delta_uj

    domains = manifest.usable_domains()
    first = {}
    for domain in domains:
        first[domain.path] = (int(_read_text(domain.energy_file)), time.monotonic())
    time.sleep(seconds)
    samples = []
    for domain in domains:
        energy_uj = int(_read_text(domain.energy_file))
        now = time.monotonic()
        previous_uj, previous_time = first[domain.path]
        delta_uj, wrapped = counter_delta_uj(previous_uj, energy_uj, domain.max_energy_range_uj)
        interval = now - previous_time
        samples.append(
            {
                "path": domain.path,
                "name": domain.name,
                "group": domain.group,
                "delta_uj": delta_uj,
                "interval_seconds": round(interval, 6),
                "power_watts": round(delta_uj / 1_000_000.0 / interval, 3) if interval > 0 else None,
                "wrapped": wrapped,
                "advancing": delta_uj > 0,
            }
        )
    return samples


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the audited RAPL profile on this node.")
    parser.add_argument("--partition", required=True, help="Slurm partition whose profile to validate.")
    parser.add_argument("--node", default="", help="Slurm node name. Defaults to the short hostname.")
    parser.add_argument("--config", type=Path, default=None, help="Profile file. Defaults to the packaged file.")
    parser.add_argument("--root", type=Path, default=None, help="Replace the profile root, for offline checks.")
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=0.0,
        help="Read each usable counter twice this many seconds apart and report power.",
    )
    args = parser.parse_args(argv)

    node = args.node or socket.gethostname().split(".", 1)[0]
    manifest = resolve_manifest(partition=args.partition, node=node, config_path=args.config, sysfs_root=args.root)
    report = manifest.to_dict()
    healthy = manifest.status == MANIFEST_READY
    if args.sample_seconds > 0 and manifest.usable_domains():
        report["samples"] = _sample_once(manifest, args.sample_seconds)
        healthy = healthy and all(sample["advancing"] for sample in report["samples"])
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if manifest.reason == "unverified_node":
        print(
            f"note: {node!r} is not an audited node of partition {args.partition!r}. "
            "This check reads the local sysfs, so run it on a node of that partition.",
            file=sys.stderr,
        )
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
