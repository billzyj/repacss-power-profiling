"""Tests for ROCm SMI parsing helpers."""

from inband.collectors.rocm_smi import _coerce_numeric, _first_value_matching


def test_first_value_matching_is_case_insensitive():
    payload = {
        "Average Graphics Package Power (W)": "185.0",
        "GPU use (%)": "97",
    }
    assert _first_value_matching(payload, ["average graphics package power"]) == "185.0"
    assert _first_value_matching(payload, ["gpu use"]) == "97"


def test_coerce_numeric_strips_units():
    assert _coerce_numeric("185.5W") == 185.5
    assert _coerce_numeric("97 %") == 97.0
    assert _coerce_numeric("N/A") is None
