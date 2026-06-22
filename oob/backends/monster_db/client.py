#!/usr/bin/env python3
"""Monster DB SSH client for OOB queries."""

import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import psycopg2

from shared.connection_policy import AccessDecision, resolve_access_decision
from shared.config.config import DatabaseConfig, SSHConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class REPACSSPowerClient:
    """Client for querying TimescaleDB through an SSH tunnel."""

    def __init__(self, db_config: DatabaseConfig, ssh_config: SSHConfig, schema: str = "idrac"):
        self.db_config = db_config
        self.ssh_config = ssh_config
        self.schema = schema
        self.tunnel = None
        self.db_connection = None
        self.access_decision: Optional[AccessDecision] = None

    def connect(self) -> None:
        try:
            self.access_decision = resolve_access_decision(
                source="db",
                target_host=self.db_config.host,
                target_port=self.db_config.port,
            )
            if not self.access_decision.use_tunnel:
                logger.info("Using direct database access: %s", self.access_decision.reason)
                self.db_connection = psycopg2.connect(
                    host=self.db_config.host,
                    port=self.db_config.port,
                    database=self.db_config.database,
                    user=self.db_config.username,
                    password=self.db_config.password,
                    sslmode=self.db_config.ssl_mode,
                )
                return

            logger.info("Using SSH tunnel for database access: %s", self.access_decision.reason)
            logger.info("Establishing SSH tunnel to %s:%s", self.ssh_config.hostname, self.ssh_config.port)

            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                local_port = sock.getsockname()[1]

            ssh_cmd = [
                "ssh",
                "-N",
                "-L",
                f"127.0.0.1:{local_port}:{self.db_config.host}:{self.db_config.port}",
                "-p",
                str(self.ssh_config.port),
                "-o",
                "ExitOnForwardFailure=yes",
                "-o",
                "BatchMode=yes",
                "-o",
                f"ServerAliveInterval={self.ssh_config.keepalive_interval}",
            ]
            if self.ssh_config.private_key_path:
                ssh_cmd += ["-o", "IdentitiesOnly=yes", "-i", self.ssh_config.private_key_path]

            destination = (
                f"{self.ssh_config.username}@{self.ssh_config.hostname}"
                if self.ssh_config.username
                else self.ssh_config.hostname
            )
            ssh_cmd.append(destination)

            self.tunnel = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(2)

            if self.tunnel.poll() is not None:
                _, stderr = self.tunnel.communicate()
                raise Exception(f"SSH tunnel failed: {stderr.decode()}")

            self.db_connection = psycopg2.connect(
                host="127.0.0.1",
                port=local_port,
                database=self.db_config.database,
                user=self.db_config.username,
                password=self.db_config.password,
                sslmode=self.db_config.ssl_mode,
            )
        except Exception as exc:
            logger.error("Failed to connect: %s", exc)
            self.disconnect()
            raise

    def disconnect(self) -> None:
        if self.db_connection:
            self.db_connection.close()
            self.db_connection = None
        if self.tunnel:
            self.tunnel.terminate()
            self.tunnel.wait()
            self.tunnel = None
        self.access_decision = None

    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[tuple]:
        if not self.db_connection:
            raise ConnectionError("Not connected to database")
        with self.db_connection.cursor() as cursor:
            cursor.execute(query, params)
            if query.strip().upper().startswith("SELECT"):
                return cursor.fetchall()
            self.db_connection.commit()
            return []

    def get_power_metrics(
        self,
        node_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(hours=1)

        query = """
        SELECT
            timestamp,
            node_id,
            power_consumption_watts,
            power_limit_watts,
            temperature_celsius,
            cpu_utilization_percent,
            memory_utilization_percent
        FROM idrac_power_metrics
        WHERE timestamp BETWEEN %s AND %s
        """
        params = [start_time, end_time]
        if node_id:
            query += " AND node_id = %s"
            params.append(node_id)
        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)
        results = self.execute_query(query, tuple(params))
        columns = [
            "timestamp",
            "node_id",
            "power_consumption_watts",
            "power_limit_watts",
            "temperature_celsius",
            "cpu_utilization_percent",
            "memory_utilization_percent",
        ]
        return [dict(zip(columns, row)) for row in results]
