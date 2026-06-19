from __future__ import annotations

from eguage.config import _normalize_api_prefix, _parse_verify_ssl, get_eguage_settings


def test_parse_verify_ssl_supports_bool_and_path_values() -> None:
    assert _parse_verify_ssl("false") is False
    assert _parse_verify_ssl("TRUE") is True
    assert _parse_verify_ssl("/tmp/ca.pem") == "/tmp/ca.pem"


def test_normalize_api_prefix_adds_leading_slash() -> None:
    assert _normalize_api_prefix("api") == "/api"
    assert _normalize_api_prefix("/api/") == "/api"


def test_eguage_settings_fall_back_to_common_ssh_settings(monkeypatch) -> None:
    monkeypatch.setenv("REPACSS_EGAUGE_USERNAME", "eguage-user")
    monkeypatch.setenv("REPACSS_EGAUGE_PASSWORD", "eguage-pass")
    monkeypatch.delenv("REPACSS_EGAUGE_SSH_HOSTNAME", raising=False)
    monkeypatch.delenv("REPACSS_EGAUGE_SSH_USERNAME", raising=False)
    monkeypatch.setenv("REPACSS_SSH_HOSTNAME", "narumuu.ttu.edu")
    monkeypatch.setenv("REPACSS_SSH_USERNAME", "jump-user")
    monkeypatch.setenv("REPACSS_SSH_PORT", "22")
    monkeypatch.setenv("REPACSS_SSH_KEEPALIVE", "90")

    settings = get_eguage_settings()

    assert settings.api.host == "192.168.4.81"
    assert settings.api.api_prefix == "/api"
    assert settings.ssh.hostname == "narumuu.ttu.edu"
    assert settings.ssh.username == "jump-user"
    assert settings.ssh.keepalive_interval == 90
