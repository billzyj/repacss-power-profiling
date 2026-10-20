"""Unit tests for in-band staging and state helpers."""

from pathlib import Path

from inband.storage import (
    build_storage_key,
    job_state_file_path,
    load_env_file,
    read_json_file,
    resolve_storage_key_for_job,
    spool_node_path,
    unit_env_file_path,
    write_env_file,
    write_json_file,
    write_systemd_env_file,
)


def test_build_storage_key_uses_cluster_job_and_epoch():
    assert build_storage_key("12345", 1775808000, cluster_name="repacss") == "repacss-12345-1775808000"


def test_env_file_round_trip(tmp_path: Path):
    state_file = job_state_file_path("12345", state_dir=tmp_path)
    write_env_file(
        state_file,
        {
            "JOB_ID": "12345",
            "STORAGE_KEY": "repacss-12345-1775808000",
            "NODE_OUTPUT_DIR": str(tmp_path / "node path"),
        },
    )
    payload = load_env_file(state_file)
    assert payload["JOB_ID"] == "12345"
    assert payload["STORAGE_KEY"] == "repacss-12345-1775808000"
    assert payload["NODE_OUTPUT_DIR"] == str(tmp_path / "node path")


def test_unit_env_file_uses_storage_key_name(tmp_path: Path):
    env_path = unit_env_file_path("repacss-12345-1775808000", state_dir=tmp_path)
    assert env_path == tmp_path / "repacss-12345-1775808000.env"


def test_spool_node_path_is_keyed_by_attempt_and_node(tmp_path: Path):
    assert spool_node_path("repacss-12345-1775808000", "rpg-93-1", spool_root=tmp_path) == (
        tmp_path / "repacss-12345-1775808000" / "rpg-93-1"
    )


def test_write_json_file_replaces_atomically(tmp_path: Path):
    path = tmp_path / "status" / "runner_status.json"
    write_json_file(path, {"state": "running"})
    write_json_file(path, {"state": "complete"})
    assert read_json_file(path) == {"state": "complete"}
    assert sorted(item.name for item in path.parent.iterdir()) == ["runner_status.json"]


def test_resolve_storage_key_for_job_returns_latest_epoch(tmp_path: Path):
    (tmp_path / "repacss-12345-100").mkdir()
    (tmp_path / "repacss-12345-200").mkdir()
    (tmp_path / "repacss-99999-300").mkdir()
    resolved = resolve_storage_key_for_job("12345", ib_store_root=tmp_path, cluster_name="repacss")
    assert resolved == "repacss-12345-200"


def test_write_systemd_env_file_uses_environmentfile_safe_quoting(tmp_path: Path):
    env_path = unit_env_file_path("repacss-12345-1775808000", state_dir=tmp_path)
    write_systemd_env_file(
        env_path,
        {
            "PLAIN": "/tmp/no-spaces",
            "WITH_SPACE": str(tmp_path / "path with spaces"),
        },
    )
    content = env_path.read_text(encoding="utf-8")
    assert "PLAIN=/tmp/no-spaces" in content
    assert 'WITH_SPACE="' in content
    assert "path with spaces" in content

