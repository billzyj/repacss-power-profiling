#!/usr/bin/env python3
"""
Energy calculation module for REPACSS Power Measurement
Handles energy consumption calculations with proper boundary handling
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conversions import convert_power_series_to_watts
import pandas as pd
from database.database import get_raw_database_connection
from queries.compute.idrac import get_compute_metrics_with_joins
from queries.infra.irc_pdu import get_irc_metrics_with_joins, get_pdu_metrics_with_joins


def compute_energy_kwh_for_hostname(df, unit: str, hostname: str, start_time: str = None, end_time: str = None) -> float:
    """
    Compute energy consumption (kWh) for a given hostname over the DataFrame's time range.
    
    Handles edge cases where query results don't include exact start/end times by
    using the first/last power values to estimate energy for missing boundary periods.

    The DataFrame is expected to contain columns: 'timestamp', 'hostname', 'value'.
    Power values will be converted to Watts before integration based on the unit.

    Integration uses trapezoidal rule over irregular sampling.

    Args:
        df: Query result DataFrame with at least 'timestamp', 'hostname', 'value'
        unit: Unit from metrics_definition table (e.g., 'mW', 'kW', 'W')
        hostname: Hostname to filter by
        start_time: Original query start time (optional, for boundary estimation)
        end_time: Original query end time (optional, for boundary estimation)

    Returns:
        Energy in kWh as a float (0.0 if insufficient data)
    """
    df = df.copy()
    if "value" not in df.columns and "power_watts" in df.columns:
        df["value"] = df["power_watts"]
    if "hostname" not in df.columns:
        df["hostname"] = hostname

    required_cols = {"timestamp", "hostname", "value"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    host_df = df[df["hostname"] == hostname].copy()
    if host_df.empty:
        return 0.0

    # Parse and sort by timestamp
    host_df["timestamp"] = pd.to_datetime(host_df["timestamp"], utc=True, errors="coerce")
    host_df = host_df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if len(host_df) < 1:
        return 0.0

    # Convert power to Watts
    host_df["power_w"] = convert_power_series_to_watts(host_df["value"], unit)

    # Handle edge cases for boundary energy estimation
    total_energy_joules = 0.0
    
    # Parse original query times if provided
    query_start = None
    query_end = None
    if start_time:
        query_start = pd.to_datetime(start_time, utc=True)
    if end_time:
        query_end = pd.to_datetime(end_time, utc=True)
    
    # Add boundary rows if needed
    if query_start and host_df["timestamp"].min() > query_start:
        # Add start boundary row
        boundary_start = pd.DataFrame({
            "timestamp": [query_start],
            "hostname": [hostname],
            "value": [host_df["value"].iloc[0]],
            "power_w": [host_df["power_w"].iloc[0]]
        })
        host_df = pd.concat([boundary_start, host_df], ignore_index=True)
        host_df = host_df.sort_values("timestamp").reset_index(drop=True)
    
    if query_end and host_df["timestamp"].max() < query_end:
        # Add end boundary row
        boundary_end = pd.DataFrame({
            "timestamp": [query_end],
            "hostname": [hostname],
            "value": [host_df["value"].iloc[-1]],
            "power_w": [host_df["power_w"].iloc[-1]]
        })
        host_df = pd.concat([host_df, boundary_end], ignore_index=True)
        host_df = host_df.sort_values("timestamp").reset_index(drop=True)
    
    # Calculate energy using trapezoidal rule
    for i in range(1, len(host_df)):
        time_diff = (host_df["timestamp"].iloc[i] - host_df["timestamp"].iloc[i-1]).total_seconds()
        avg_power = (host_df["power_w"].iloc[i] + host_df["power_w"].iloc[i-1]) / 2.0
        energy_joules = avg_power * time_diff
        total_energy_joules += energy_joules
    
    # Convert from Joules to kWh
    return total_energy_joules / 3_600_000.0


class EnergyCalculator:
    """Handles energy consumption calculations for power analysis"""
    
    def __init__(self, database: str):
        self.database = database
        
    def calculate_energy_for_hostname(self, df: pd.DataFrame, unit: str, hostname: str, 
                                    start_time: str = None, end_time: str = None) -> float:
        """
        Compute energy consumption (kWh) for a given hostname over the DataFrame's time range.
        
        Handles edge cases where query results don't include exact start/end times by
        using the first/last power values to estimate energy for missing boundary periods.

        Args:
            df: Query result DataFrame with at least 'timestamp', 'hostname', 'value'
            unit: Unit from metrics_definition table (e.g., 'mW', 'kW', 'W')
            hostname: Hostname to filter by
            start_time: Original query start time (optional, for boundary estimation)
            end_time: Original query end time (optional, for boundary estimation)

        Returns:
            Energy in kWh as a float (0.0 if insufficient data)
        """
        df = df.copy()
        if "value" not in df.columns and "power_watts" in df.columns:
            df["value"] = df["power_watts"]
        if "hostname" not in df.columns:
            df["hostname"] = hostname

        required_cols = {"timestamp", "hostname", "value"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

        host_df = df[df["hostname"] == hostname].copy()
        if host_df.empty:
            return 0.0

        # Parse and sort by timestamp
        host_df["timestamp"] = pd.to_datetime(host_df["timestamp"], utc=True, errors="coerce")
        host_df = host_df.dropna(subset=["timestamp"]).sort_values("timestamp")
        if len(host_df) < 1:
            return 0.0

        # Convert power to Watts
        host_df["power_w"] = convert_power_series_to_watts(host_df["value"], unit)

        # Handle edge cases for boundary energy estimation
        total_energy_joules = 0.0
        
        # Parse original query times if provided
        query_start = None
        query_end = None
        if start_time:
            query_start = pd.to_datetime(start_time, utc=True)
        if end_time:
            query_end = pd.to_datetime(end_time, utc=True)
        
        # Get actual data boundaries
        data_start = host_df["timestamp"].iloc[0]
        data_end = host_df["timestamp"].iloc[-1]
        first_power = host_df["power_w"].iloc[0]
        last_power = host_df["power_w"].iloc[-1]
        
        # Estimate energy for missing start period (query_start to data_start)
        if query_start and data_start > query_start:
            start_duration = (data_start - query_start).total_seconds()
            start_energy = first_power * start_duration
            total_energy_joules += start_energy
        
        # Calculate energy for the actual data period using trapezoidal rule
        if len(host_df) >= 2:
            time_seconds = host_df["timestamp"].astype("int64") // 10**9
            dt = time_seconds.diff().astype("float64")  # seconds between samples
            avg_power = (host_df["power_w"] + host_df["power_w"].shift(1)) / 2.0
            data_energy = (avg_power * dt).sum(skipna=True)
            total_energy_joules += data_energy
        else:
            # Single data point - use it for the entire period if we have query boundaries
            if query_start and query_end:
                total_duration = (query_end - query_start).total_seconds()
                total_energy_joules = first_power * total_duration
            else:
                # No query boundaries, can't estimate energy accurately
                return 0.0
        
        # Estimate energy for missing end period (data_end to query_end)
        if query_end and data_end < query_end:
            end_duration = (query_end - data_end).total_seconds()
            end_energy = last_power * end_duration
            total_energy_joules += end_energy

        # Convert Joules to kWh (1 kWh = 3.6e6 J)
        energy_kwh = float(total_energy_joules) / 3_600_000.0
        return max(energy_kwh, 0.0)
    
    def calculate_energy(self, hostname: str, start_time: datetime, end_time: datetime, 
                        metrics: List[str] = None) -> Dict[str, Any]:
        """
        Calculate energy consumption for a specific hostname and time range.
        
        Args:
            hostname: Node hostname
            start_time: Start timestamp
            end_time: End timestamp
            metrics: List of metrics to analyze (if None, uses defaults per node type)
        
        Returns:
            Dictionary with energy results
        """
        from utils.node_detection import get_node_type_and_query_func
        from constants.metrics import PDU_POWER_METRICS, IRC_POWER_METRICS
        
        if metrics is None:
            # Default metrics based on node type
            node_type, _, database, schema = get_node_type_and_query_func(hostname)
            if node_type == 'pdu':
                metrics = PDU_POWER_METRICS
            elif node_type == 'irc':
                metrics = IRC_POWER_METRICS
            else:  # compute nodes - get from database
                metrics = self._get_compute_power_metrics(database, schema)
        
        # Get queries and execute
        queries, database, schema = self._get_power_metrics_with_joins(metrics, hostname, start_time, end_time)
        
        # Connect to database
        db_connection = get_raw_database_connection(database, schema)
        
        try:
            # Create SQLAlchemy engine to avoid pandas warnings
            from sqlalchemy import create_engine
            engine = create_engine(f"postgresql://{db_connection.info.user}:{db_connection.info.password}@{db_connection.info.host}:{db_connection.info.port}/{db_connection.info.dbname}")
            
            # Execute queries and calculate energy
            energy_results = {}
            for metric, query in queries.items():
                df = pd.read_sql_query(query, engine)
                if not df.empty:
                    # Get unit from the data
                    unit = df['units'].iloc[0] if 'units' in df.columns else 'W'
                    
                    # Calculate total energy for this metric
                    total_energy_kwh = self.calculate_energy_for_hostname(
                        df, unit, hostname, 
                        start_time.strftime('%Y-%m-%d %H:%M:%S'),
                        end_time.strftime('%Y-%m-%d %H:%M:%S')
                    )
                    energy_results[metric] = total_energy_kwh
            
            return energy_results
            
        finally:
            if db_connection:
                db_connection.close()
    
    def _get_compute_power_metrics(self, database: str, schema: str) -> List[str]:
        """Get power metrics for compute nodes from the database."""
        from queries.compute.public import POWER_METRICS_QUERY_UNIT_IN_MW_W_KW
        
        db_connection = get_raw_database_connection(database, schema)
        try:
            # Create SQLAlchemy engine to avoid pandas warnings
            from sqlalchemy import create_engine
            engine = create_engine(f"postgresql://{db_connection.info.user}:{db_connection.info.password}@{db_connection.info.host}:{db_connection.info.port}/{db_connection.info.dbname}")
            df = pd.read_sql_query(POWER_METRICS_QUERY_UNIT_IN_MW_W_KW, engine)
            return df['metric_id'].tolist()
        finally:
            if db_connection:
                db_connection.close()
    
    def _get_power_metrics_with_joins(self, metrics: List[str], hostname: str, 
                                    start_time: datetime, end_time: datetime):
        """Generate queries for multiple metrics for a given hostname."""
        from utils.node_detection import get_node_type_and_query_func
        
        node_type, query_func, database, schema = get_node_type_and_query_func(hostname)
        
        queries = {}
        for metric in metrics:
            if node_type in ['pdu']:
                # PDU doesn't use metric_id parameter
                query = query_func(hostname, start_time, end_time)
            else:
                # IRC and compute nodes use metric_id
                query = query_func(metric, hostname, start_time, end_time)
            
            queries[metric] = query
        
        return queries, database, schema
    
    def display_energy_summary(self, energy_results: Dict[str, Any]):
        """Display energy calculation summary."""
        print("⚡ Energy Consumption Summary")
        print("=" * 50)
        
        total_energy = 0.0
        for metric, energy_kwh in energy_results.items():
            print(f"{metric:30} {energy_kwh:10.4f} kWh")
            total_energy += energy_kwh
        
        print("-" * 50)
        print(f"{'Total Energy':30} {total_energy:10.4f} kWh")
        print(f"{'Total Energy':30} {total_energy * 1000:10.2f} Wh")
