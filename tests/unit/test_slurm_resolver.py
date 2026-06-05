from __future__ import annotations

from datetime import datetime, timezone

from shared.slurm.resolver import parse_job_start_epoch, resolve_job_context_from_rest


def test_parse_job_start_epoch_accepts_naive_iso_timestamp() -> None:
    raw = "2026-04-14T08:00:00"
    expected = int(datetime(2026, 4, 14, 8, 0, 0).timestamp())
    assert parse_job_start_epoch(raw) == expected


def test_parse_job_start_epoch_accepts_utc_z_suffix() -> None:
    raw = "2026-04-14T08:00:00Z"
    expected = int(datetime(2026, 4, 14, 8, 0, 0, tzinfo=timezone.utc).timestamp())
    assert parse_job_start_epoch(raw) == expected


def test_resolve_job_context_from_rest_maps_slurmdb_payload(monkeypatch) -> None:
    class _FakeClient:
        def get_job(self, job_id: str, *, from_database: bool = True):
            assert job_id == "96597"
            assert from_database is True
            return {
                "job_id": 96597,
                "user": {"name": "alice"},
                "nodes": "rpg-93-1,rpg-93-2",
                "time": {
                    "start": "2026-04-13T00:00:00",
                    "end": "2026-04-13T01:00:00",
                },
                "comment": "power:oob",
                "cluster": "repacss",
            }

    monkeypatch.setattr("shared.slurm.resolver.config.validate_slurm_rest_config", lambda: [])

    context = resolve_job_context_from_rest("96597", client=_FakeClient())

    assert context.job_id == "96597"
    assert context.user == "alice"
    assert context.nodes == ["rpg-93-1", "rpg-93-2"]
    assert context.start_time == datetime(2026, 4, 13, 0, 0, 0)
    assert context.end_time == datetime(2026, 4, 13, 1, 0, 0)
    assert context.comment == "power:oob"
    assert context.cluster_name == "repacss"
