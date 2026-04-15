"""Shared models for Slurm/OOB orchestration."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass(frozen=True)
class ParsedPowerComment:
    """Parsed Slurm comment for REPACSS power workflows."""

    mode: Optional[str]
    backend: str = "db"
    collectors: List[str] = field(default_factory=list)
    interval_ms: int = 1000
    raw: str = ""
    warnings: List[str] = field(default_factory=list)
    enabled: bool = False

    @property
    def includes_oob(self) -> bool:
        return self.mode in {"oob", "both"}

    @property
    def includes_inband(self) -> bool:
        return self.mode in {"inband", "both"}


@dataclass(frozen=True)
class SlurmJobContext:
    """Normalized Slurm job context used by handlers and backends."""

    job_id: str
    user: str
    nodelist: str
    nodes: List[str]
    start_time: datetime
    end_time: datetime
    comment: str = ""
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    cluster_name: Optional[str] = None

