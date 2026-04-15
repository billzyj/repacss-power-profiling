"""Base interface for OOB telemetry backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from shared.models import SlurmJobContext


class OOBBackend(ABC):
    """Backend-neutral interface for OOB data retrieval."""

    name: str

    @abstractmethod
    def query_job(self, context: SlurmJobContext) -> pd.DataFrame:
        """Query all OOB rows needed for a single job."""

    @abstractmethod
    def query_metrics(self, hostname: str, start_time=None, end_time=None, limit: int = 100) -> pd.DataFrame:
        """Query generic OOB metrics for a hostname/time range."""

