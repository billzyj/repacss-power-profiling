#!/usr/bin/env python3
"""OOB query management system."""

from datetime import datetime
import logging
from typing import Any, Dict, List

import pandas as pd

from oob.backends.monster_db.connection_pool import get_pooled_connection
from shared.errors import OOBBackendError
from src.queries.compute.idrac import get_compute_metrics_with_joins
from src.queries.infra.irc_pdu import get_irc_metrics_with_joins, get_pdu_metrics_with_joins


logger = logging.getLogger(__name__)


class QueryManager:
    """Manage direct DB-backed OOB queries."""

    def __init__(self, database: str, schema: str = None):
        self.database = database
        self.schema = schema
        self._query_cache: Dict[str, str] = {}
        self._result_cache: Dict[str, pd.DataFrame] = {}

    def get_power_metrics(
        self,
        hostname: str,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        try:
            node_type, query_func, db, schema = self._get_node_type_and_query_func(hostname)
            metrics = self._get_metrics_for_node_type(node_type, db, schema)
            all_data = []
            for metric in metrics:
                try:
                    if node_type in ["pdu"]:
                        query = query_func(hostname, start_time, end_time)
                    else:
                        query = query_func(metric, hostname, start_time, end_time)
                    with get_pooled_connection(db, schema) as client:
                        df = pd.read_sql_query(query, client.db_connection)
                    if not df.empty:
                        df["metric"] = metric
                        all_data.append(df)
                except Exception as exc:
                    logger.warning("Error querying metric %s for %s: %s", metric, hostname, exc)
                    continue
            if all_data:
                return pd.concat(all_data, ignore_index=True)
            return pd.DataFrame()
        except Exception as exc:
            logger.error("Error getting power metrics for %s: %s", hostname, exc)
            raise OOBBackendError(f"Failed to query power metrics for {hostname}: {exc}") from exc

    def get_metrics_definition(self, database: str = None, schema: str = None) -> pd.DataFrame:
        db = database or self.database
        schema = schema or "public"
        query = """
        SELECT
            metric_id,
            metric_name,
            description,
            metric_data_type,
            units,
            accuracy,
            sensing_interval
        FROM public.metrics_definition
        ORDER BY metric_name
        """
        try:
            with get_pooled_connection(db, schema) as client:
                return pd.read_sql_query(query, client.db_connection)
        except Exception as exc:
            logger.error("Error getting metrics definition: %s", exc)
            return pd.DataFrame()

    def get_power_metrics_definition(self, database: str = None, schema: str = None) -> pd.DataFrame:
        db = database or self.database
        schema = schema or "public"
        query = """
        SELECT
            metric_id,
            metric_name,
            description,
            metric_data_type,
            units,
            accuracy,
            sensing_interval
        FROM public.metrics_definition
        WHERE units IN ('mW', 'W', 'kW') OR metric_name LIKE '%Power%'
        ORDER BY metric_name
        """
        try:
            with get_pooled_connection(db, schema) as client:
                return pd.read_sql_query(query, client.db_connection)
        except Exception as exc:
            logger.error("Error getting power metrics definition: %s", exc)
            return pd.DataFrame()

    def get_database_info(self, database: str = None) -> Dict[str, Any]:
        db = database or self.database
        try:
            with get_pooled_connection(db, "public") as client:
                version_result = pd.read_sql_query("SELECT version()", client.db_connection)
                tables_result = pd.read_sql_query(
                    """
                    SELECT COUNT(*) as table_count
                    FROM information_schema.tables
                    WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                    """,
                    client.db_connection,
                )
                schemas_result = pd.read_sql_query(
                    """
                    SELECT schema_name
                    FROM information_schema.schemata
                    WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                    ORDER BY schema_name
                    """,
                    client.db_connection,
                )
            return {
                "database": db,
                "version": version_result.iloc[0]["version"] if not version_result.empty else "Unknown",
                "table_count": tables_result.iloc[0]["table_count"] if not tables_result.empty else 0,
                "schemas": schemas_result["schema_name"].tolist() if not schemas_result.empty else [],
            }
        except Exception as exc:
            logger.error("Error getting database info: %s", exc)
            return {"database": db, "error": str(exc)}

    @staticmethod
    def _get_node_type_and_query_func(hostname: str):
        mapping = {
            "pdu": ("pdu", get_pdu_metrics_with_joins, "infra", "pdu"),
            "irc": ("irc", get_irc_metrics_with_joins, "infra", "irc"),
            "rpg": ("h100", get_compute_metrics_with_joins, "h100", "idrac"),
            "rpc": ("zen4", get_compute_metrics_with_joins, "zen4", "idrac"),
        }
        for prefix, value in mapping.items():
            if hostname.startswith(prefix):
                return value
        raise ValueError(f"Invalid hostname: {hostname}")

    def _get_metrics_for_node_type(self, node_type: str, database: str, schema: str) -> List[str]:
        if node_type == "pdu":
            return ["pdu"]
        if node_type == "irc":
            return [
                "CompressorPower",
                "CondenserFanPower",
                "CoolDemand",
                "CoolOutput",
                "TotalAirSideCoolingDemand",
                "TotalSensibleCoolingPower",
            ]
        return self._get_compute_power_metrics(database, schema)

    @staticmethod
    def _get_compute_power_metrics(database: str, schema: str) -> List[str]:
        try:
            with get_pooled_connection(database, "public") as client:
                df = pd.read_sql_query(
                    """
                    SELECT metric_id
                    FROM public.metrics_definition
                    WHERE units IN ('mW', 'W', 'kW')
                    ORDER BY metric_id
                    """,
                    client.db_connection,
                )
            return df["metric_id"].tolist()
        except Exception as exc:
            logger.error("Error getting compute power metrics: %s", exc)
            return []
