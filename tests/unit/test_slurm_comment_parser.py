"""Unit tests for shared Slurm comment parsing."""

import pytest

from shared.errors import CommentParseError
from shared.slurm.comments import parse_power_comment


def test_parse_oob_comment_defaults():
    parsed = parse_power_comment("power:oob")
    assert parsed.enabled is True
    assert parsed.mode == "oob"
    assert parsed.backend == "db"
    assert parsed.includes_oob is True
    assert parsed.includes_inband is False


def test_parse_both_comment_with_options():
    parsed = parse_power_comment("power:both;backend=api;collectors=rapl,nvidia_smi;interval_ms=500")
    assert parsed.mode == "both"
    assert parsed.backend == "api"
    assert parsed.collectors == ["rapl", "nvidia_smi"]
    assert parsed.interval_ms == 500


def test_legacy_keyword_maps_to_oob():
    parsed = parse_power_comment("power_profiling")
    assert parsed.enabled is True
    assert parsed.mode == "oob"
    assert parsed.warnings


def test_structural_error_raises():
    with pytest.raises(CommentParseError):
        parse_power_comment("power:foo")


def test_semantic_mismatch_becomes_warning():
    parsed = parse_power_comment("power:inband;backend=api")
    assert parsed.mode == "inband"
    assert parsed.backend == "db"
    assert any("ignored backend" in warning for warning in parsed.warnings)
