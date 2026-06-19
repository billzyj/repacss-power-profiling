"""eGauge API client with SSH jump-host support."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable

import requests

from shared.errors import EGaugeError

from .config import EGaugeSettings, get_eguage_settings
from .tunnel import SSHLocalForward


logger = logging.getLogger(__name__)


def build_register_params(
    *,
    view: str | None = None,
    register_range: str | None = None,
    time_range: str | None = None,
    include_rate: bool = True,
    include_raw: bool = False,
    include_delta: bool = False,
    virtual: str | None = "formula",
    max_rows: int | None = None,
) -> Dict[str, Any]:
    """Build `/register` query parameters using eGauge's presence-flag style."""

    params: Dict[str, Any] = {}
    if view:
        params["view"] = view
    if register_range:
        params["reg"] = register_range
    if time_range:
        params["time"] = time_range
    if include_rate:
        params["rate"] = ""
    if include_raw:
        params["raw"] = ""
    if include_delta:
        params["delta"] = ""
    if virtual:
        params["virtual"] = virtual
    if max_rows is not None:
        params["max-rows"] = max_rows
    return params


class EGaugeClient:
    """Minimal eGauge Web API client for REPACSS workflows."""

    def __init__(self, settings: EGaugeSettings | None = None, session: requests.Session | None = None):
        self.settings = settings or get_eguage_settings()
        self.session = session
        self.tunnel: SSHLocalForward | None = None
        self.base_url: str | None = None
        self.jwt: str | None = None

    @classmethod
    def from_env(cls) -> "EGaugeClient":
        """Create a client from the shared root `.env` file."""

        return cls(settings=get_eguage_settings())

    def connect(self) -> None:
        """Open the SSH tunnel and prepare the HTTP session."""

        issues = self.settings.validate()
        if issues:
            raise EGaugeError("Invalid eGauge configuration:\n- " + "\n- ".join(issues))

        self.tunnel = SSHLocalForward(
            ssh_host=self.settings.ssh.hostname,
            ssh_port=self.settings.ssh.port,
            ssh_user=self.settings.ssh.username,
            target_host=self.settings.api.host,
            target_port=self.settings.api.port,
            keepalive_interval=self.settings.ssh.keepalive_interval,
            local_bind_host=self.settings.ssh.local_bind_host,
            private_key_path=self.settings.ssh.private_key_path,
        )
        local_port = self.tunnel.start()
        self.base_url = (
            f"{self.settings.api.scheme}://{self.settings.ssh.local_bind_host}:{local_port}"
            f"{self.settings.api.api_prefix}"
        )

        if self.session is None:
            self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self.session.verify = self.settings.api.verify_ssl

        if self.settings.api.verify_ssl is False:
            requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]

    def disconnect(self) -> None:
        """Close the HTTP session and SSH tunnel."""

        if self.session is not None:
            self.session.close()
            self.session = None
        if self.tunnel is not None:
            self.tunnel.stop()
            self.tunnel = None
        self.base_url = None
        self.jwt = None

    def __enter__(self) -> "EGaugeClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    def _url(self, path: str) -> str:
        if not self.base_url:
            raise EGaugeError("Client is not connected.")
        normalized = path if path.startswith("/") else "/" + path
        return f"{self.base_url}{normalized}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        expected_status: Iterable[int] = (200,),
        **kwargs: Any,
    ) -> requests.Response:
        if self.session is None:
            raise EGaugeError("Client is not connected.")
        try:
            response = self.session.request(
                method=method,
                url=self._url(path),
                timeout=self.settings.api.timeout,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise EGaugeError(f"eGauge request failed: {exc}") from exc

        if response.status_code not in set(expected_status):
            error_text = response.text.strip()
            try:
                payload = response.json()
                error_text = json.dumps(payload, indent=2)
            except ValueError:
                pass
            raise EGaugeError(f"eGauge API {method} {path} failed with HTTP {response.status_code}: {error_text}")
        return response

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        expected_status: Iterable[int] = (200,),
        **kwargs: Any,
    ) -> Dict[str, Any]:
        response = self._request(method, path, expected_status=expected_status, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            raise EGaugeError(f"eGauge API {method} {path} returned non-JSON data.") from exc

    def fetch_auth_challenge(self) -> Dict[str, Any]:
        """Fetch the auth realm and nonce from `/auth/unauthorized`."""

        response = self._request("GET", "/auth/unauthorized", expected_status=(200, 401))
        try:
            payload = response.json()
        except ValueError as exc:
            raise EGaugeError("eGauge auth challenge did not return JSON.") from exc

        if response.status_code == 200:
            return payload
        if "rlm" not in payload:
            raise EGaugeError("eGauge auth challenge did not include the authentication realm.")
        return payload

    def login_with_password(self) -> Dict[str, Any]:
        """Authenticate with username/password and store the returned JWT."""

        if self.settings.api.scheme != "https":
            raise EGaugeError("Password login requires HTTPS according to the eGauge Web API documentation.")

        challenge = self.fetch_auth_challenge()
        realm = challenge.get("rlm")
        if not realm:
            raise EGaugeError("eGauge auth challenge did not provide a realm.")

        payload = {
            "usr": self.settings.api.username,
            "pwd": self.settings.api.password,
            "rlm": realm,
        }
        reply = self._request_json("POST", "/auth/login", expected_status=(200,), json=payload)
        token = reply.get("jwt")
        if not token:
            raise EGaugeError("eGauge login succeeded without returning a JWT token.")

        self.jwt = token
        assert self.session is not None
        self.session.headers["Authorization"] = f"Bearer {token}"
        logger.info("Authenticated to eGauge API as %s", self.settings.api.username)
        return reply

    def ensure_authenticated(self) -> None:
        """Log in if we do not yet have a JWT token."""

        if not self.jwt:
            self.login_with_password()

    def get_token_rights(self) -> Dict[str, Any]:
        """Return the rights associated with the active token."""

        self.ensure_authenticated()
        try:
            return self._request_json("GET", "/auth/rights", expected_status=(200,))
        except EGaugeError as exc:
            if "HTTP 401" not in str(exc):
                raise
            self.login_with_password()
            return self._request_json("GET", "/auth/rights", expected_status=(200,))

    def get_registers(
        self,
        *,
        view: str | None = None,
        register_range: str | None = None,
        time_range: str | None = None,
        include_rate: bool = True,
        include_raw: bool = False,
        include_delta: bool = False,
        virtual: str | None = "formula",
        max_rows: int | None = None,
    ) -> Dict[str, Any]:
        """Fetch register metadata, current rates, and optional historical rows."""

        self.ensure_authenticated()
        params = build_register_params(
            view=view,
            register_range=register_range,
            time_range=time_range,
            include_rate=include_rate,
            include_raw=include_raw,
            include_delta=include_delta,
            virtual=virtual,
            max_rows=max_rows,
        )
        return self._request_json("GET", "/register", expected_status=(200,), params=params)
