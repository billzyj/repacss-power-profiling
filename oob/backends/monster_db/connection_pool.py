#!/usr/bin/env python3
"""Connection pooling for the Monster DB backend."""

import logging
import threading
import time
from contextlib import contextmanager
from queue import Empty, Queue
from typing import Dict, Optional

from shared.config import config

from .client import REPACSSPowerClient


logger = logging.getLogger(__name__)


class ConnectionPool:
    """Manage pooled Monster DB connections."""

    def __init__(self, max_connections: int = 10, max_idle_time: int = 300):
        self.max_connections = max_connections
        self.max_idle_time = max_idle_time
        self._pools: Dict[str, Queue] = {}
        self._connection_info: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._cleanup_thread = None
        self._shutdown = False
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        self._cleanup_thread = threading.Thread(target=self._cleanup_idle_connections, daemon=True)
        self._cleanup_thread.start()

    def _cleanup_idle_connections(self):
        while not self._shutdown:
            time.sleep(60)
            self._cleanup_old_connections()

    def _cleanup_old_connections(self):
        current_time = time.time()
        with self._lock:
            for pool_key, pool in self._pools.items():
                survivors = []
                while not pool.empty():
                    try:
                        connection_info = pool.get_nowait()
                    except Empty:
                        break
                    if current_time - connection_info["last_used"] < self.max_idle_time:
                        survivors.append(connection_info)
                    else:
                        try:
                            connection_info["client"].disconnect()
                        except Exception as exc:
                            logger.warning("Error closing idle connection for %s: %s", pool_key, exc)
                for connection_info in survivors:
                    pool.put(connection_info)

    @staticmethod
    def _get_pool_key(database: str, schema: str) -> str:
        return f"{database}_{schema}"

    @staticmethod
    def _create_connection(database: str, schema: str) -> REPACSSPowerClient:
        db_config = config.get_database_config(database, schema)
        ssh_config = config.get_ssh_config()
        client = REPACSSPowerClient(db_config, ssh_config, schema=db_config.schema)
        client.connect()
        return client

    def get_connection(self, database: str, schema: str = None) -> Optional[REPACSSPowerClient]:
        if schema is None:
            schema = config.get_default_schema(database)
        pool_key = self._get_pool_key(database, schema)
        with self._lock:
            if pool_key not in self._pools:
                self._pools[pool_key] = Queue()
                self._connection_info[pool_key] = {
                    "created": time.time(),
                    "total_connections": 0,
                    "active_connections": 0,
                }

            pool = self._pools[pool_key]
            info = self._connection_info[pool_key]
            try:
                conn_info = pool.get_nowait()
                conn_info["last_used"] = time.time()
                info["active_connections"] += 1
                return conn_info["client"]
            except Empty:
                if info["total_connections"] >= self.max_connections:
                    logger.warning("Connection pool full for %s", pool_key)
                    return None
                client = self._create_connection(database, schema)
                info["total_connections"] += 1
                info["active_connections"] += 1
                return client

    def return_connection(self, client: REPACSSPowerClient, database: str, schema: str = None):
        if schema is None:
            schema = config.get_default_schema(database)
        pool_key = self._get_pool_key(database, schema)
        with self._lock:
            if pool_key not in self._pools:
                return
            pool = self._pools[pool_key]
            info = self._connection_info[pool_key]
            try:
                with client.db_connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                pool.put({"client": client, "last_used": time.time(), "database": database, "schema": schema})
                info["active_connections"] -= 1
            except Exception:
                try:
                    client.disconnect()
                finally:
                    info["total_connections"] -= 1
                    info["active_connections"] -= 1

    def shutdown(self):
        self._shutdown = True
        with self._lock:
            for pool in self._pools.values():
                while not pool.empty():
                    try:
                        conn_info = pool.get_nowait()
                    except Empty:
                        break
                    try:
                        conn_info["client"].disconnect()
                    except Exception:
                        pass
            self._pools.clear()
            self._connection_info.clear()


_connection_pool: Optional[ConnectionPool] = None


def get_connection_pool() -> ConnectionPool:
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = ConnectionPool()
    return _connection_pool


@contextmanager
def get_pooled_connection(database: str, schema: str = None):
    pool = get_connection_pool()
    client = pool.get_connection(database, schema)
    if client is None:
        raise ConnectionError(f"Failed to get connection for {database}/{schema}")
    try:
        yield client
    finally:
        pool.return_connection(client, database, schema)

