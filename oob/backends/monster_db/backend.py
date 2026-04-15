"""Monster DB backend implementation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from oob.backends.monster_db.database import get_raw_database_connection
from shared.constants.metrics import H100_METRICS, ZEN4_METRICS
from shared.errors import OOBBackendError
from shared.models import SlurmJobContext
from src.queries.compute.idrac import get_compute_metrics_with_joins

from oob.backends.base import OOBBackend
from oob.query_manager import QueryManager


def _format_local_ts_with_offset(dt: datetime) -> str:
    ts = dt.timestamp()
    local = datetime.fromtimestamp(ts).astimezone()
    offset = local.strftime("%z")
    return dt.strftime("%Y-%m-%d %H:%M:%S") + (offset[:3] if len(offset) >= 3 else "")


def _node_db_and_metrics(node_id: str) -> Optional[Tuple[str, str, List[str]]]:
    node_id = (node_id or "").strip().lower()
    if node_id.startswith("rpc"):
        return ("zen4", "idrac", list(ZEN4_METRICS))
    if node_id.startswith("rpg"):
        return ("h100", "idrac", list(H100_METRICS))
    return None


def _fetch_raw_power_for_node(conn: Any, hostname: str, metrics: List[str], start_str: str, end_str: str) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        query = get_compute_metrics_with_joins(metric, hostname=hostname, start_time=start_str, end_time=end_str)
        df = pd.read_sql_query(query, conn)
        if not df.empty:
            df["metric"] = metric
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


class MonsterDBBackend(OOBBackend):
    """OOB backend backed by direct Monster DB access."""

    name = "monster-db"

    def __init__(self, database: str = "h100", schema: str | None = None):
        self.database = database
        self.schema = schema
        self.query_manager = QueryManager(database, schema)

    def query_metrics(self, hostname: str, start_time=None, end_time=None, limit: int = 100) -> pd.DataFrame:
        return self.query_manager.get_power_metrics(hostname, start_time=start_time, end_time=end_time, limit=limit)

    def query_job(self, context: SlurmJobContext) -> pd.DataFrame:
        start_str = _format_local_ts_with_offset(context.start_time)
        end_str = _format_local_ts_with_offset(context.end_time)
        all_dfs: List[pd.DataFrame] = []
        by_db: Dict[Tuple[str, str], List[Tuple[str, List[str]]]] = {}

        for node in context.nodes:
            info = _node_db_and_metrics(node)
            if info is None:
                continue
            database, schema, metrics = info
            by_db.setdefault((database, schema), []).append((node, metrics))

        try:
            for (database, schema), node_metrics_list in by_db.items():
                conn = get_raw_database_connection(database, schema)
                if conn is None:
                    raise OOBBackendError(f"Could not open raw database connection for {database}.{schema}")
                try:
                    for node, metrics in node_metrics_list:
                        df = _fetch_raw_power_for_node(conn, node, metrics, start_str, end_str)
                        if not df.empty:
                            all_dfs.append(df)
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass
        except OOBBackendError:
            raise
        except Exception as exc:
            raise OOBBackendError(f"Monster DB job query failed for job {context.job_id}: {exc}") from exc

        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
