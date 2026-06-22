from __future__ import annotations

import pytest

from shared.connection_policy import (
    get_configured_access_mode,
    normalize_access_mode,
    resolve_access_decision,
)


def test_normalize_access_mode_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        normalize_access_mode("vpn")


def test_source_specific_access_mode_overrides_common_mode(monkeypatch) -> None:
    monkeypatch.setenv("REPACSS_ACCESS_MODE", "tunnel")
    monkeypatch.setenv("REPACSS_DB_ACCESS_MODE", "direct")

    assert get_configured_access_mode("db") == "direct"
    assert get_configured_access_mode("eguage") == "tunnel"


def test_auto_access_mode_uses_direct_when_probe_succeeds() -> None:
    decision = resolve_access_decision(
        source="db",
        target_host="db.internal",
        target_port=5432,
        access_mode="auto",
        probe=lambda _host, _port, _timeout: True,
    )

    assert decision.use_tunnel is False
    assert decision.reason == "direct probe to db.internal:5432 succeeded"


def test_auto_access_mode_uses_tunnel_when_probe_fails() -> None:
    decision = resolve_access_decision(
        source="eguage",
        target_host="192.168.4.81",
        target_port=443,
        access_mode="auto",
        probe=lambda _host, _port, _timeout: False,
    )

    assert decision.use_tunnel is True
    assert decision.reason == "direct probe to 192.168.4.81:443 failed"


def test_explicit_tunnel_mode_does_not_probe() -> None:
    called = False

    def probe(_host: str, _port: int, _timeout: float) -> bool:
        nonlocal called
        called = True
        return True

    decision = resolve_access_decision(
        source="db",
        target_host="db.internal",
        target_port=5432,
        access_mode="tunnel",
        probe=probe,
    )

    assert decision.use_tunnel is True
    assert called is False
