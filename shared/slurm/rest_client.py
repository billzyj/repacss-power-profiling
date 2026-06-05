"""Small Slurm REST API client used by offline OOB job resolution."""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any, Dict

import requests
from requests.adapters import HTTPAdapter

from shared.config import SlurmRESTConfig
from shared.errors import SlurmResolutionError


class SlurmRESTClient:
    """Fetch Slurm metadata through slurmrestd using short-lived JWT tokens."""

    def __init__(self, config: SlurmRESTConfig):
        self.config = config
        self._token = ""
        self._token_expires_at = 0.0

    def get_job(self, job_id: str, *, from_database: bool = True) -> Dict[str, Any]:
        path = self.config.db_job_path if from_database else self.config.job_path
        payload = self._get_json(f"{path}{job_id}")
        jobs = payload.get("jobs") or []
        return jobs[0] if jobs else {}

    def _get_json(self, path: str) -> Dict[str, Any]:
        url = f"http://{self.config.host}:{self.config.port}{path}"
        headers = {
            "X-SLURM-USER-NAME": self.config.user,
            "X-SLURM-USER-TOKEN": self._read_token(),
        }
        adapter = HTTPAdapter(max_retries=3)
        try:
            with requests.Session() as session:
                session.mount("http://", adapter)
                response = session.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                return response.json()
        except requests.RequestException as exc:
            raise SlurmResolutionError(f"Slurm REST request failed for {url}: {exc}") from exc
        except ValueError as exc:
            raise SlurmResolutionError(f"Slurm REST response was not valid JSON for {url}") from exc

    def _read_token(self) -> str:
        now = time.time()
        if self._token and now < self._token_expires_at:
            return self._token
        return self._refresh_token()

    def _refresh_token(self) -> str:
        if not self.config.headnode:
            raise SlurmResolutionError("Slurm REST headnode is required to request a token")
        command = ["ssh", self.config.headnode, "scontrol token lifespan=3600"]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SlurmResolutionError(f"Could not obtain Slurm token through {self.config.headnode}: {exc}") from exc

        token = _parse_token(result.stdout)
        self._token = token
        self._token_expires_at = time.time() + 3500
        return token


def _parse_token(raw_output: str) -> str:
    for line in raw_output.splitlines():
        if "=" in line:
            _key, value = line.split("=", 1)
            token = value.strip()
            if token:
                return token
    try:
        decoded = json.loads(raw_output)
    except json.JSONDecodeError:
        decoded = {}
    if isinstance(decoded, dict) and decoded.get("token"):
        return str(decoded["token"])
    raise SlurmResolutionError("Could not parse token from scontrol output")
