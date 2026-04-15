"""Headnode dispatcher for Slurm-triggered power workflows."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from shared.errors import CommentParseError, OOBBackendError, SlurmResolutionError
from shared.slurm.comments import parse_power_comment
from shared.slurm.resolver import resolve_job_context_from_env


LOGGER = logging.getLogger("repacss_power.slurm.dispatcher")


def configure_logging() -> None:
    logfile = os.environ.get("MONSTER_POWER_LOGFILE")
    level = logging.DEBUG if os.environ.get("MONSTER_POWER_DEBUG") == "1" else logging.INFO
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=level, filename=logfile, format="%(asctime)s %(levelname)s %(message)s")
    else:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def dispatch_from_env() -> int:
    """Dispatch the current Slurm job based on the parsed comment."""
    try:
        context = resolve_job_context_from_env()
    except SlurmResolutionError as exc:
        LOGGER.info("skip power workflow: %s", exc)
        return 0

    try:
        parsed = parse_power_comment(context.comment)
    except CommentParseError as exc:
        LOGGER.error("comment parse error for job %s: %s", context.job_id, exc)
        return 0

    if not parsed.enabled:
        LOGGER.info("skip job %s: no REPACSS power workflow comment", context.job_id)
        return 0

    for warning in parsed.warnings:
        LOGGER.warning("job %s comment warning: %s", context.job_id, warning)

    if not parsed.includes_oob:
        LOGGER.info("job %s requested mode=%s; OOB dispatcher leaves non-OOB modes for later phases", context.job_id, parsed.mode)
        return 0

    if parsed.backend != "db":
        LOGGER.error("job %s requested backend=%s; only monster-db is implemented in P0-P2", context.job_id, parsed.backend)
        return 0

    from oob.slurm.epilog_handler import handle_oob_job

    try:
        handle_oob_job(context)
    except OOBBackendError as exc:
        LOGGER.error("oob backend error for job %s: %s", context.job_id, exc)
        return 1
    except Exception as exc:  # pragma: no cover - defensive wrapper
        LOGGER.exception("unexpected OOB dispatcher failure for job %s: %s", context.job_id, exc)
        return 1
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="REPACSS Slurm dispatcher")
    parser.parse_args(argv)
    configure_logging()
    return dispatch_from_env()


if __name__ == "__main__":
    raise SystemExit(main())

