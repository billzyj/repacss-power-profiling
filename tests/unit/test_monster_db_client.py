from __future__ import annotations

from unittest.mock import MagicMock, patch

from oob.backends.monster_db.client import REPACSSPowerClient
from shared.config.config import DatabaseConfig, SSHConfig


def _db_config() -> DatabaseConfig:
    return DatabaseConfig(
        host="db.internal",
        port=5432,
        database="h100",
        username="monster",
        password="secret",
        ssl_mode="prefer",
        schema="idrac",
    )


def _ssh_config() -> SSHConfig:
    return SSHConfig(
        hostname="narumuu.ttu.edu",
        port=22,
        username="jump-user",
        private_key_path=None,
        passphrase="",
        keepalive_interval=60,
    )


@patch("oob.backends.monster_db.client.subprocess.Popen")
@patch("oob.backends.monster_db.client.psycopg2.connect")
def test_monster_db_client_uses_direct_connection_when_configured(
    mock_connect: MagicMock,
    mock_popen: MagicMock,
    monkeypatch,
) -> None:
    monkeypatch.setenv("REPACSS_DB_ACCESS_MODE", "direct")
    client = REPACSSPowerClient(_db_config(), _ssh_config())

    client.connect()

    mock_popen.assert_not_called()
    mock_connect.assert_called_once_with(
        host="db.internal",
        port=5432,
        database="h100",
        user="monster",
        password="secret",
        sslmode="prefer",
    )
