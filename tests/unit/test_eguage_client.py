from __future__ import annotations

from eguage.client import EGaugeClient, build_register_params
from eguage.config import EGaugeAPIConfig, EGaugeSettings, EGaugeSSHConfig


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]):
        self.responses = list(responses)
        self.calls = []
        self.headers = {}
        self.verify = True

    def request(self, method, url, timeout, **kwargs):
        self.calls.append({"method": method, "url": url, "timeout": timeout, "kwargs": kwargs})
        return self.responses.pop(0)

    def close(self):
        return None


class _FakeTunnel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.local_port = 18443

    def start(self):
        return self.local_port

    def stop(self):
        return None


def _settings() -> EGaugeSettings:
    return EGaugeSettings(
        api=EGaugeAPIConfig(
            scheme="https",
            host="192.168.4.81",
            port=443,
            api_prefix="/api",
            username="owner",
            password="secret",
            verify_ssl=False,
            timeout=15,
        ),
        ssh=EGaugeSSHConfig(
            hostname="narumuu.ttu.edu",
            port=22,
            username="jump-user",
            private_key_path=None,
            passphrase="",
            keepalive_interval=60,
            local_bind_host="127.0.0.1",
        ),
        access_mode="tunnel",
        probe_timeout=0.1,
    )


def test_build_register_params_uses_presence_flags() -> None:
    params = build_register_params(
        view="=generation",
        register_range="all",
        time_range="now-300:60:now",
        include_rate=True,
        include_raw=True,
        include_delta=True,
        virtual="value",
        max_rows=5,
    )

    assert params == {
        "view": "=generation",
        "reg": "all",
        "time": "now-300:60:now",
        "rate": "",
        "raw": "",
        "delta": "",
        "virtual": "value",
        "max-rows": 5,
    }


def test_client_login_with_password_sets_bearer_header(monkeypatch) -> None:
    session = _FakeSession(
        [
            _FakeResponse(401, {"rlm": "eGauge Administration", "nnc": "server-nonce"}),
            _FakeResponse(200, {"jwt": "token-123", "rights": ["api"]}),
        ]
    )
    monkeypatch.setattr("eguage.client.SSHLocalForward", _FakeTunnel)

    client = EGaugeClient(settings=_settings(), session=session)
    client.connect()
    reply = client.login_with_password()

    assert reply["jwt"] == "token-123"
    assert session.headers["Authorization"] == "Bearer token-123"
    assert session.calls[0]["url"] == "https://127.0.0.1:18443/api/auth/unauthorized"
    assert session.calls[1]["kwargs"]["json"] == {
        "usr": "owner",
        "pwd": "secret",
        "rlm": "eGauge Administration",
    }


def test_client_connects_direct_when_access_mode_is_direct() -> None:
    session = _FakeSession([])
    settings = _settings()
    direct_settings = EGaugeSettings(
        api=settings.api,
        ssh=settings.ssh,
        access_mode="direct",
        probe_timeout=settings.probe_timeout,
    )

    client = EGaugeClient(settings=direct_settings, session=session)
    client.connect()

    assert client.tunnel is None
    assert client.base_url == "https://192.168.4.81:443/api"
