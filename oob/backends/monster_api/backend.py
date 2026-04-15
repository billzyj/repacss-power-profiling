"""Monster API backend placeholder for later phases."""

from oob.backends.base import OOBBackend


class MonsterAPIBackend(OOBBackend):
    """Placeholder backend to reserve the planned package path."""

    name = "monster-api"

    def query_job(self, context):
        raise NotImplementedError("monster-api backend is planned for P8")

    def query_metrics(self, hostname, start_time=None, end_time=None, limit: int = 100):
        raise NotImplementedError("monster-api backend is planned for P8")

