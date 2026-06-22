"""Connection access policy helpers for private REPACSS services."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Callable


VALID_ACCESS_MODES = {"auto", "direct", "tunnel"}
DEFAULT_PROBE_TIMEOUT_SECONDS = 1.0


@dataclass(frozen=True)
class AccessDecision:
    """Resolved direct-vs-tunnel decision for one service target."""

    source: str
    mode: str
    target_host: str
    target_port: int
    use_tunnel: bool
    reason: str


ProbeFunc = Callable[[str, int, float], bool]


def normalize_access_mode(raw_mode: str | None, default: str = "auto") -> str:
    """Normalize an access mode string and reject unsupported values."""

    mode = (raw_mode or default or "auto").strip().lower()
    if mode not in VALID_ACCESS_MODES:
        allowed = ", ".join(sorted(VALID_ACCESS_MODES))
        raise ValueError(f"Unsupported REPACSS access mode {mode!r}; expected one of: {allowed}")
    return mode


def get_configured_access_mode(source: str, default: str = "auto") -> str:
    """Return the source-specific access mode with a common fallback."""

    source_key = source.upper()
    raw_mode = os.getenv(f"REPACSS_{source_key}_ACCESS_MODE") or os.getenv("REPACSS_ACCESS_MODE")
    return normalize_access_mode(raw_mode, default=default)


def get_probe_timeout(source: str) -> float:
    """Return the source-specific internal-network probe timeout."""

    source_key = source.upper()
    raw_timeout = (
        os.getenv(f"REPACSS_{source_key}_PROBE_TIMEOUT")
        or os.getenv("REPACSS_INTERNAL_PROBE_TIMEOUT")
        or str(DEFAULT_PROBE_TIMEOUT_SECONDS)
    )
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise ValueError(f"Invalid REPACSS probe timeout {raw_timeout!r}") from exc
    if timeout <= 0:
        raise ValueError("REPACSS probe timeout must be positive")
    return timeout


def can_open_tcp_connection(host: str, port: int, timeout: float) -> bool:
    """Return whether host:port accepts a TCP connection within timeout."""

    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def resolve_access_decision(
    *,
    source: str,
    target_host: str,
    target_port: int,
    access_mode: str | None = None,
    probe_timeout: float | None = None,
    probe: ProbeFunc | None = None,
) -> AccessDecision:
    """Resolve whether a private REPACSS target should be reached directly or through SSH."""

    default_mode = get_configured_access_mode(source) if access_mode is None else "auto"
    mode = normalize_access_mode(access_mode, default=default_mode)
    port = int(target_port)

    if mode == "direct":
        return AccessDecision(
            source=source,
            mode=mode,
            target_host=target_host,
            target_port=port,
            use_tunnel=False,
            reason="direct mode is configured",
        )
    if mode == "tunnel":
        return AccessDecision(
            source=source,
            mode=mode,
            target_host=target_host,
            target_port=port,
            use_tunnel=True,
            reason="tunnel mode is configured",
        )

    timeout = probe_timeout if probe_timeout is not None else get_probe_timeout(source)
    probe_func = probe or can_open_tcp_connection
    reachable = probe_func(target_host, port, timeout)
    if reachable:
        return AccessDecision(
            source=source,
            mode=mode,
            target_host=target_host,
            target_port=port,
            use_tunnel=False,
            reason=f"direct probe to {target_host}:{port} succeeded",
        )
    return AccessDecision(
        source=source,
        mode=mode,
        target_host=target_host,
        target_port=port,
        use_tunnel=True,
        reason=f"direct probe to {target_host}:{port} failed",
    )
