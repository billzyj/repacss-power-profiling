#!/usr/bin/env python3
"""
REPACSS Power Measurement Client
Connects to TimescaleDB through SSH tunnel to query power metrics from iDRAC
"""

import psycopg2
from sshtunnel import SSHTunnelForwarder
import paramiko
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: str
    schema: str


@dataclass
class SSHConfig:
    """SSH tunnel configuration"""
    hostname: str
    port: int
    username: str
    private_key_path: str
    passphrase: str
    keepalive_interval: int


class REPACSSPowerClient:
    """Client for connecting to REPACSS TimescaleDB and querying power metrics"""
    
    def __init__(self, db_config: DatabaseConfig, ssh_config: SSHConfig, schema: str = "idrac"):
        self.db_config = db_config
        self.ssh_config = ssh_config
        self.schema = schema  # Schema to query from (default: idrac)
        self.tunnel = None
        self.db_connection = None
        
    def connect(self) -> None:
        """Connect to the database through SSH tunnel"""
        try:
            logger.info(f"Establishing SSH tunnel to {self.ssh_config.hostname}:{self.ssh_config.port}")
            # Let SSHTunnelForwarder handle key loading automatically
            # Don't pre-load the key, let SSHTunnelForwarder handle it
            ssh_key = self.ssh_config.private_key_path
            
            # Use subprocess to create SSH tunnel (bypasses DSS issues)
            import subprocess
            import socket
            import time
            
            # Find an available local port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                local_port = s.getsockname()[1]
            
            # Create SSH tunnel using subprocess
            ssh_cmd = [
                'ssh',
                '-N',
                '-L',
                f'{local_port}:{self.db_config.host}:{self.db_config.port}',
                '-p',
                str(self.ssh_config.port),
                # Fail fast if port-forward can't be established
                '-o',
                'ExitOnForwardFailure=yes',
                # Keep the tunnel alive
                '-o',
                f'ServerAliveInterval={self.ssh_config.keepalive_interval}',
            ]

            # Optional explicit identity file; if not set, rely on ssh-agent/default keys/ssh config.
            if self.ssh_config.private_key_path:
                ssh_cmd += ['-i', self.ssh_config.private_key_path]

            destination = (
                f'{self.ssh_config.username}@{self.ssh_config.hostname}'
                if self.ssh_config.username
                else self.ssh_config.hostname
            )
            ssh_cmd.append(destination)
            
            # Start the SSH tunnel process
            self.tunnel = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a moment for the tunnel to establish
            time.sleep(2)
            
            # Check if the process is still running
            if self.tunnel.poll() is not None:
                stdout, stderr = self.tunnel.communicate()
                raise Exception(f"SSH tunnel failed: {stderr.decode()}")
            
            logger.info(f"SSH tunnel established: localhost:{local_port} -> {self.db_config.host}:{self.db_config.port}")
            self.db_connection = psycopg2.connect(
                host='localhost',
                port=local_port,
                database=self.db_config.database,
                user=self.db_config.username,
                password=self.db_config.password,
                sslmode=self.db_config.ssl_mode
            )
            with self.db_connection.cursor() as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()
                logger.info(f"Connected to database: {version[0]}")
                
        except Exception as e:
            error_msg = str(e)
            if "DSSKey" in error_msg:
                logger.error(f"SSH key format issue: {error_msg}")
                logger.error("DSS keys are deprecated. Please convert your SSH key to RSA or Ed25519 format:")
                logger.error("  ssh-keygen -p -m RFC4716 -f /path/to/your/key")
                logger.error("  or generate a new key: ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519")
            else:
                logger.error(f"Failed to connect: {e}")
            self.disconnect()
            raise
    
    def disconnect(self) -> None:
        """Disconnect from database and SSH tunnel"""
        if self.db_connection:
            self.db_connection.close()
            self.db_connection = None
            logger.info("Database connection closed")
        
        if self.tunnel:
            if hasattr(self.tunnel, 'stop'):
                self.tunnel.stop()
            elif hasattr(self.tunnel, 'terminate'):
                # Subprocess
                self.tunnel.terminate()
                self.tunnel.wait()
            else:
                # SSH client
                self.tunnel.close()
            self.tunnel = None
            logger.info("SSH tunnel closed")
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[tuple]:
        """Execute a query and return results"""
        if not self.db_connection:
            raise ConnectionError("Not connected to database")
        
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(query, params)
                if query.strip().upper().startswith('SELECT'):
                    return cursor.fetchall()
                else:
                    self.db_connection.commit()
                    return []
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            self.db_connection.rollback()
            raise
    
    def get_power_metrics(self, node_id: Optional[str] = None, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Get power metrics from iDRAC"""
        
        # Default time range if not provided
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
        
        # Convert to list of dictionaries
        columns = ['timestamp', 'node_id', 'power_consumption_watts', 'power_limit_watts', 
                  'temperature_celsius', 'cpu_utilization_percent', 'memory_utilization_percent']
        
        return [dict(zip(columns, row)) for row in results]
    
    def get_node_summary(self, node_id: str) -> Dict[str, Any]:
        """Get power summary for a specific node"""
        query = """
        SELECT 
            node_id,
            AVG(power_consumption_watts) as avg_power_watts,
            MAX(power_consumption_watts) as max_power_watts,
            MIN(power_consumption_watts) as min_power_watts,
            AVG(temperature_celsius) as avg_temperature_celsius,
            MAX(temperature_celsius) as max_temperature_celsius,
            AVG(cpu_utilization_percent) as avg_cpu_utilization,
            AVG(memory_utilization_percent) as avg_memory_utilization,
            COUNT(*) as data_points
        FROM idrac_power_metrics
        WHERE node_id = %s
        AND timestamp >= NOW() - INTERVAL '24 hours'
        GROUP BY node_id
        """
        
        results = self.execute_query(query, (node_id,))
        if results:
            columns = ['node_id', 'avg_power_watts', 'max_power_watts', 'min_power_watts',
                      'avg_temperature_celsius', 'max_temperature_celsius', 'avg_cpu_utilization',
                      'avg_memory_utilization', 'data_points']
            return dict(zip(columns, results[0]))
        return {}
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get power summary for entire cluster"""
        query = """
        SELECT 
            COUNT(DISTINCT node_id) as total_nodes,
            AVG(power_consumption_watts) as cluster_avg_power_watts,
            SUM(power_consumption_watts) as cluster_total_power_watts,
            MAX(power_consumption_watts) as cluster_max_power_watts,
            AVG(temperature_celsius) as cluster_avg_temperature_celsius,
            MAX(temperature_celsius) as cluster_max_temperature_celsius,
            AVG(cpu_utilization_percent) as cluster_avg_cpu_utilization,
            AVG(memory_utilization_percent) as cluster_avg_memory_utilization
        FROM idrac_power_metrics
        WHERE timestamp >= NOW() - INTERVAL '1 hour'
        """
        
        results = self.execute_query(query)
        if results:
            columns = ['total_nodes', 'cluster_avg_power_watts', 'cluster_total_power_watts',
                      'cluster_max_power_watts', 'cluster_avg_temperature_celsius', 
                      'cluster_max_temperature_celsius', 'cluster_avg_cpu_utilization',
                      'cluster_avg_memory_utilization']
            return dict(zip(columns, results[0]))
        return {} 

    def get_computepower_metrics(self, node_id: Optional[str] = None, 
                                start_time: Optional[datetime] = None,
                                end_time: Optional[datetime] = None,
                                limit: int = 100) -> List[Dict[str, Any]]:
        """Get compute power metrics from the specified schema"""
        
        # Default time range if not provided
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(hours=1)
        
        query = f"""
        SELECT 
            timestamp,
            nodeid,
            source,
            fqdd,
            value
        FROM {self.schema}.computepower
        WHERE timestamp BETWEEN %s AND %s
        """
        params = [start_time, end_time]
        
        if node_id:
            query += " AND nodeid = %s"
            params.append(node_id)
        
        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)
        
        results = self.execute_query(query, tuple(params))
        
        # Convert to list of dictionaries
        columns = ['timestamp', 'nodeid', 'source', 'fqdd', 'value']
        
        return [dict(zip(columns, row)) for row in results]
    
    def get_boardtemperature_metrics(self, node_id: Optional[str] = None, 
                                   start_time: Optional[datetime] = None,
                                   end_time: Optional[datetime] = None,
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """Get board temperature metrics from the specified schema"""
        
        # Default time range if not provided
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(hours=1)
        
        query = f"""
        SELECT 
            timestamp,
            nodeid,
            source,
            fqdd,
            value
        FROM {self.schema}.boardtemperature
        WHERE timestamp BETWEEN %s AND %s
        """
        params = [start_time, end_time]
        
        if node_id:
            query += " AND nodeid = %s"
            params.append(node_id)
        
        query += " ORDER BY timestamp DESC LIMIT %s"
        params.append(limit)
        
        results = self.execute_query(query, tuple(params))
        
        # Convert to list of dictionaries
        columns = ['timestamp', 'nodeid', 'source', 'fqdd', 'value']
        
        return [dict(zip(columns, row)) for row in results]
    
    def get_computepower_summary(self, node_id: Optional[str] = None) -> Dict[str, Any]:
        """Get compute power summary for a specific node or all nodes"""
        query = f"""
        SELECT 
            AVG(value) as avg_power,
            MAX(value) as max_power,
            MIN(value) as min_power,
            COUNT(*) as data_points
        FROM {self.schema}.computepower
        WHERE timestamp >= NOW() - INTERVAL '24 hours'
        """
        params = []
        
        if node_id:
            query += " AND nodeid = %s"
            params.append(node_id)
        
        results = self.execute_query(query, tuple(params) if params else None)
        if results:
            columns = ['avg_power', 'max_power', 'min_power', 'data_points']
            return dict(zip(columns, results[0]))
        return {}
    
    def get_boardtemperature_summary(self, node_id: Optional[str] = None) -> Dict[str, Any]:
        """Get board temperature summary for a specific node or all nodes"""
        query = f"""
        SELECT 
            AVG(value) as avg_temp,
            MAX(value) as max_temp,
            MIN(value) as min_temp,
            COUNT(*) as data_points
        FROM {self.schema}.boardtemperature
        WHERE timestamp >= NOW() - INTERVAL '24 hours'
        """
        params = []
        
        if node_id:
            query += " AND nodeid = %s"
            params.append(node_id)
        
        results = self.execute_query(query, tuple(params) if params else None)
        if results:
            columns = ['avg_temp', 'max_temp', 'min_temp', 'data_points']
            return dict(zip(columns, results[0]))
        return {}
    
    def get_idrac_cluster_summary(self) -> Dict[str, Any]:
        """Get iDRAC metrics summary for entire cluster"""
        query = f"""
        SELECT 
            COUNT(DISTINCT comp.nodeid) as total_nodes,
            AVG(comp.value) as cluster_avg_power,
            MAX(comp.value) as cluster_max_power,
            AVG(temp.value) as cluster_avg_temp,
            MAX(temp.value) as cluster_max_temp,
            COUNT(comp.value) as power_data_points,
            COUNT(temp.value) as temp_data_points
        FROM {self.schema}.computepower comp
        FULL OUTER JOIN {self.schema}.boardtemperature temp 
            ON comp.nodeid = temp.nodeid 
            AND comp.timestamp = temp.timestamp
        WHERE comp.timestamp >= NOW() - INTERVAL '1 hour'
           OR temp.timestamp >= NOW() - INTERVAL '1 hour'
        """
        
        results = self.execute_query(query)
        if results:
            columns = ['total_nodes', 'cluster_avg_power', 'cluster_max_power',
                      'cluster_avg_temp', 'cluster_max_temp',
                      'power_data_points', 'temp_data_points']
            return dict(zip(columns, results[0]))
        return {}
    
    def get_available_idrac_metrics(self) -> List[str]:
        """Get list of available iDRAC metrics"""
        query = f"""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = '{self.schema}'
        ORDER BY table_name
        """
        
        results = self.execute_query(query)
        return [row[0] for row in results] if results else [] 