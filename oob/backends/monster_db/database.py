#!/usr/bin/env python3
"""Connection manager utilities for the Monster DB backend."""

from typing import Dict, List, Optional

from shared.config import config

from .client import REPACSSPowerClient


class DatabaseConnectionManager:
    """Manage on-demand Monster DB clients."""

    def __init__(self):
        self.clients: Dict[str, REPACSSPowerClient] = {}
        self.connected_databases: List[str] = []

    def create_client_for_database(self, database_name: str, schema: str = None) -> REPACSSPowerClient:
        db_config = config.get_database_config(database_name, schema)
        ssh_config = config.get_ssh_config()
        return REPACSSPowerClient(db_config, ssh_config, schema=db_config.schema)

    def connect_to_database(self, database_name: str, schema: str = None) -> Optional[REPACSSPowerClient]:
        try:
            client = self.create_client_for_database(database_name, schema)
            client.connect()
            key = f"{database_name}_{schema}" if schema else database_name
            self.clients[key] = client
            self.connected_databases.append(database_name)
            return client
        except Exception:
            return None

    def disconnect_all(self):
        for client in list(self.clients.values()):
            try:
                client.disconnect()
            except Exception:
                pass
        self.clients.clear()
        self.connected_databases.clear()

    def get_client(self, database_name: str, schema: str = None) -> Optional[REPACSSPowerClient]:
        key = f"{database_name}_{schema}" if schema else database_name
        return self.clients.get(key)


_connection_manager: Optional[DatabaseConnectionManager] = None


def get_connection_manager() -> DatabaseConnectionManager:
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = DatabaseConnectionManager()
    return _connection_manager


def connect_to_database(database_name: str, schema: str = None) -> Optional[REPACSSPowerClient]:
    return get_connection_manager().connect_to_database(database_name, schema)


def get_raw_database_connection(database_name: str, schema: str = None):
    client = get_connection_manager().connect_to_database(database_name, schema)
    if client and client.db_connection:
        return client.db_connection
    return None


def disconnect_all():
    get_connection_manager().disconnect_all()

