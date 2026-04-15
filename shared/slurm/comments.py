"""Parse Slurm job comments into structured REPACSS power workflow settings."""

from __future__ import annotations

from typing import List

from shared.errors import CommentParseError
from shared.models import ParsedPowerComment


LEGACY_KEYWORD = "power_profiling"
VALID_MODES = {"oob", "inband", "both"}
VALID_BACKENDS = {"db", "api"}


def extract_power_token(raw_comment: str) -> str:
    """Extract the REPACSS power token from a potentially mixed Slurm comment.

    Slurm comments may contain multiple workflow namespaces separated by whitespace,
    for example ``ECHO=1;R=0.80 power:both;collectors=rapl``. The power parser only
    consumes the first whitespace-delimited token that starts with ``power:``.
    """

    raw = (raw_comment or "").strip()
    if not raw:
        return ""

    if raw == LEGACY_KEYWORD:
        return raw

    for token in raw.split():
        token = token.strip()
        if token == LEGACY_KEYWORD or token.startswith("power:"):
            return token
    return ""


def parse_power_comment(raw_comment: str) -> ParsedPowerComment:
    """Parse a Slurm comment using the refactor-plan grammar."""
    raw = (raw_comment or "").strip()
    token = extract_power_token(raw)
    if not token:
        return ParsedPowerComment(mode=None, raw=raw, enabled=False)

    if token == LEGACY_KEYWORD:
        return ParsedPowerComment(
            mode="oob",
            raw=raw,
            enabled=True,
            warnings=["legacy comment keyword accepted as power:oob"],
        )

    parts = [part.strip() for part in token.split(";") if part.strip()]
    if not parts:
        return ParsedPowerComment(mode=None, raw=raw, enabled=False)

    head = parts[0]
    if not head.startswith("power:"):
        return ParsedPowerComment(mode=None, raw=raw, enabled=False)

    mode = head.split(":", 1)[1].strip().lower()
    if mode not in VALID_MODES:
        raise CommentParseError(f"Unsupported power mode: {mode}")

    backend = "db"
    collectors: List[str] = []
    interval_ms = 1000
    warnings: List[str] = []

    for token in parts[1:]:
        if "=" not in token:
            raise CommentParseError(f"Malformed comment token: {token}")
        key, value = token.split("=", 1)
        key = key.strip().lower()
        value = value.strip()

        if key == "backend":
            if mode not in {"oob", "both"}:
                warnings.append(f"ignored backend={value} for mode={mode}")
                continue
            value = value.lower()
            if value not in VALID_BACKENDS:
                raise CommentParseError(f"Unsupported backend: {value}")
            backend = value
        elif key == "collectors":
            if mode not in {"inband", "both"}:
                warnings.append(f"ignored collectors={value} for mode={mode}")
                continue
            collectors = [item.strip() for item in value.split(",") if item.strip()]
        elif key == "interval_ms":
            if mode not in {"inband", "both"}:
                warnings.append(f"ignored interval_ms={value} for mode={mode}")
                continue
            try:
                interval_ms = int(value)
            except ValueError as exc:
                raise CommentParseError(f"Invalid interval_ms: {value}") from exc
            if interval_ms <= 0:
                raise CommentParseError("interval_ms must be positive")
        else:
            warnings.append(f"ignored unknown key {key}")

    return ParsedPowerComment(
        mode=mode,
        backend=backend,
        collectors=collectors,
        interval_ms=interval_ms,
        raw=raw,
        warnings=warnings,
        enabled=True,
    )
