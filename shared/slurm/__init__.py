"""Shared Slurm helpers.

The helpers below are imported on first use. Importing a light submodule, as the root
delivery service does, then does not load `resolver` and the configuration, including
database credentials, that it reads from `.env`.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "expand_nodelist": "resolver",
    "fetch_job_comment": "resolver",
    "parse_power_comment": "comments",
    "resolve_job_context": "resolver",
    "resolve_job_context_from_env": "resolver",
    "resolve_job_context_from_rest": "resolver",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
