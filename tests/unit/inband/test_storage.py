"""Unit tests for in-band staging and state helpers."""

from pathlib import Path

from inband.storage import (
    build_storage_key,
    epoch_cache_path,
    ensure_node_staging,
    job_state_file_path,
    load_env_file,
    node_root_path,
    resolve_storage_key_for_job,
    unit_env_file_path,
    write_env_file,
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


def test_ensure_node_staging_creates_expected_tree(tmp_path: Path):
    root = ensure_node_staging("repacss-12345-1775808000", "rpg-93-1", ib_store_root=tmp_path)
    assert root == node_root_path("repacss-12345-1775808000", "rpg-93-1", ib_store_root=tmp_path)
    assert root.exists()


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


def test_epoch_cache_path_uses_hidden_state_dir(tmp_path: Path):
    path = epoch_cache_path("12345", "repacss", ib_store_root=tmp_path)
    assert path == tmp_path / ".state" / "repacss-12345.epoch"
