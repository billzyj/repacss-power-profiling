from __future__ import annotations

from types import SimpleNamespace

from shared.config import SlurmRESTConfig
from shared.slurm.rest_client import SlurmRESTClient


def _config() -> SlurmRESTConfig:
    return SlurmRESTConfig(
        host="nuetu",
        port=6820,
        user="yongzhao",
        headnode="repacss",
        jobs_path="/slurm/v0.0.42/jobs/",
        nodes_path="/slurm/v0.0.42/nodes/",
        job_path="/slurm/v0.0.42/job/",
        db_job_path="/slurmdb/v0.0.42/job/",
        openapi_path="/openapi/v3",
    )


def test_rest_client_requests_token_through_headnode(monkeypatch) -> None:
    client = SlurmRESTClient(_config())
    calls = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        return SimpleNamespace(stdout="SLURM_JWT=abc123\n")

    monkeypatch.setattr("shared.slurm.rest_client.subprocess.run", fake_run)

    assert client._read_token() == "abc123"
    assert calls == [["ssh", "repacss", "scontrol token lifespan=3600"]]
