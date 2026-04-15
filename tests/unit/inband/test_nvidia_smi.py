"""Tests for NVIDIA SMI parsing helpers."""

from inband.collectors.nvidia_smi import _parse_csv_line


def test_parse_nvidia_smi_csv_line():
    row = _parse_csv_line("0, GPU-123, H100, 285.50, 61, 98, 40, 1410, 1593, 12345")
    assert row["gpu_index"] == "0"
    assert row["gpu_uuid"] == "GPU-123"
    assert row["power_watts"] == 285.50
    assert row["temperature_c"] == 61.0
    assert row["utilization_gpu_percent"] == 98.0
    assert row["memory_used_mb"] == 12345.0


def test_parse_nvidia_smi_csv_line_with_na_values():
    row = _parse_csv_line("0, GPU-123, H100, N/A, N/A, N/A, N/A, N/A, N/A, N/A")
    assert row["power_watts"] is None
    assert row["temperature_c"] is None
