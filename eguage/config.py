"""Configuration helpers for the eGauge API connector."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from shared.config.config import config as shared_config


def _normalize_api_prefix(raw: str | None) -> str:
    cleaned = (raw or "/api").strip() or "/api"
    if not cleaned.startswith("/"):
        cleaned = "/" + cleaned
    return cleaned.rstrip("/") or "/api"


def _parse_verify_ssl(raw: str | None) -> bool | str:
    cleaned = (raw or "false").strip()
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if lowered in {"0", "false", "no", "off"}:
        return False
    if lowered in {"1", "true", "yes", "on"}:
        return True
    return cleaned


@dataclass(frozen=True)
class EGaugeAPIConfig:
    """eGauge API connection settings."""

    scheme: str
    host: str
    port: int
    api_prefix: str
    username: str
    password: str
    verify_ssl: bool | str
    timeout: int


@dataclass(frozen=True)
class EGaugeSSHConfig:
    """SSH jump-host configuration for reaching the private eGauge meter."""

    hostname: str
    port: int
    username: str
    private_key_path: str | None
    passphrase: str
    keepalive_interval: int
    local_bind_host: str


@dataclass(frozen=True)
class EGaugeSettings:
    """Bundled eGauge connector settings."""

    api: EGaugeAPIConfig
    ssh: EGaugeSSHConfig

    def validate(self) -> List[str]:
        issues: List[str] = []
        if self.api.scheme not in {"http", "https"}:
            issues.append("REPACSS_EGAUGE_SCHEME must be either 'http' or 'https'")
        if not self.api.host:
            issues.append("REPACSS_EGAUGE_HOST is not configured")
        if self.api.port <= 0:
            issues.append("REPACSS_EGAUGE_PORT must be a positive integer")
        if not self.api.username:
            issues.append("REPACSS_EGAUGE_USERNAME is not configured")
        if not self.api.password:
            issues.append("REPACSS_EGAUGE_PASSWORD is not configured")
        if self.api.timeout <= 0:
            issues.append("REPACSS_EGAUGE_TIMEOUT must be a positive integer")
        if not self.ssh.hostname:
            issues.append("REPACSS_EGAUGE_SSH_HOSTNAME is not configured")
        if not self.ssh.username:
            issues.append("REPACSS_EGAUGE_SSH_USERNAME is not configured")
        if self.ssh.port <= 0:
            issues.append("REPACSS_EGAUGE_SSH_PORT must be a positive integer")
        if self.ssh.private_key_path and not Path(self.ssh.private_key_path).exists():
            issues.append(f"SSH private key file not found: {self.ssh.private_key_path}")
        return issues


def get_eguage_settings() -> EGaugeSettings:
    """Load eGauge connector settings from the shared root `.env` file."""

    base_ssh = shared_config.get_ssh_config()
    common_ssh_host = os.getenv("REPACSS_SSH_HOSTNAME", "").strip() or base_ssh.hostname
    common_ssh_port = int(os.getenv("REPACSS_SSH_PORT", "").strip() or base_ssh.port)
    common_ssh_user = os.getenv("REPACSS_SSH_USERNAME", "").strip() or base_ssh.username
    common_ssh_key = os.getenv("REPACSS_SSH_KEY_PATH", "").strip() or base_ssh.private_key_path
    common_ssh_passphrase = os.getenv("REPACSS_SSH_PASSPHRASE", "").strip() or base_ssh.passphrase
    common_ssh_keepalive = int(os.getenv("REPACSS_SSH_KEEPALIVE", "").strip() or base_ssh.keepalive_interval)

    api = EGaugeAPIConfig(
        scheme=(os.getenv("REPACSS_EGAUGE_SCHEME", "https").strip() or "https").lower(),
        host=os.getenv("REPACSS_EGAUGE_HOST", "192.168.4.81").strip() or "192.168.4.81",
        port=int(os.getenv("REPACSS_EGAUGE_PORT", "443").strip() or 443),
        api_prefix=_normalize_api_prefix(os.getenv("REPACSS_EGAUGE_API_PREFIX", "/api")),
        username=os.getenv("REPACSS_EGAUGE_USERNAME", "").strip(),
        password=os.getenv("REPACSS_EGAUGE_PASSWORD", "").strip(),
        verify_ssl=_parse_verify_ssl(os.getenv("REPACSS_EGAUGE_VERIFY_SSL", "false")),
        timeout=int(os.getenv("REPACSS_EGAUGE_TIMEOUT", "30").strip() or 30),
    )

    ssh_key = os.getenv("REPACSS_EGAUGE_SSH_KEY_PATH", "").strip() or common_ssh_key
    ssh = EGaugeSSHConfig(
        hostname=os.getenv("REPACSS_EGAUGE_SSH_HOSTNAME", "").strip() or common_ssh_host,
        port=int(os.getenv("REPACSS_EGAUGE_SSH_PORT", "").strip() or common_ssh_port),
        username=os.getenv("REPACSS_EGAUGE_SSH_USERNAME", "").strip() or common_ssh_user,
        private_key_path=ssh_key or None,
        passphrase=os.getenv("REPACSS_EGAUGE_SSH_PASSPHRASE", "").strip() or common_ssh_passphrase,
        keepalive_interval=int(os.getenv("REPACSS_EGAUGE_SSH_KEEPALIVE", "").strip() or common_ssh_keepalive),
        local_bind_host=os.getenv("REPACSS_EGAUGE_LOCAL_BIND_HOST", "127.0.0.1").strip() or "127.0.0.1",
    )

    return EGaugeSettings(api=api, ssh=ssh)
