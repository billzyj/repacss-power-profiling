#!/usr/bin/env python3
"""
Power analysis module for REPACSS Power Measurement
Handles power consumption analysis and data processing
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys
import os
from sqlalchemy import create_engine

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conversions import convert_power_series_to_watts
from utils.data_processing import process_power_data
from database.database import get_raw_database_connection
from queries.compute.idrac import get_compute_metrics_with_joins
from queries.infra.irc_pdu import get_irc_metrics_with_joins, get_pdu_metrics_with_joins


class PowerAnalyzer:
    """Handles power consumption analysis for different node types"""
    
    def __init__(self, database: str):
        self.database = database
        
    def analyze_power(self, hostname: str, start_time: datetime, end_time: datetime, 
                     metrics: List[str] = None) -> Dict[str, Any]:
        """
        Single node power analysis for a given hostname and time range.
        
        Args:
            hostname: Node hostname
            start_time: Start timestamp
            end_time: End timestamp  
            metrics: List of metrics to analyze (if None, uses all available)
        
        Returns:
            Dictionary with power analysis results
        """
        if metrics is None:
            # Default metrics based on node type
            node_type, _, database, schema = self._get_node_type_and_query_func(hostname)
            if node_type == 'pdu':
                metrics = ['pdu']  # PDU_POWER_METRICS
            elif node_type == 'irc':
                from constants.metrics import IRC_ALL_METRICS
                metrics = IRC_ALL_METRICS
            else:  # compute nodes - get from database
                metrics = self._get_compute_power_metrics(database, schema)
        
        queries, database, schema = self._get_power_metrics_with_joins(metrics, hostname, start_time, end_time)
        
        # Connect to database
        db_connection = get_raw_database_connection(database, schema)
        
        try:
            # Create SQLAlchemy engine to avoid pandas warnings
            engine = create_engine(f"postgresql://{db_connection.info.user}:{db_connection.info.password}@{db_connection.info.host}:{db_connection.info.port}/{db_connection.info.dbname}")
            # Execute queries and combine results
            all_data = []
            for metric, query in queries.items():
                df = pd.read_sql_query(query, engine)
                if not df.empty:
                    df['metric'] = metric
                    all_data.append(df)
            
            if not all_data:
                return {}
            
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Process power data with energy calculations
            processed_df = process_power_data(combined_df, hostname, start_time, end_time)
            
            return {
                'data': processed_df,
                'summary': self._calculate_power_summary(processed_df),
                'hostname': hostname,
                'start_time': start_time,
                'end_time': end_time
            }
            
        finally:
            if db_connection:
                db_connection.close()
    
    def multi_node_analysis(self, hostnames: List[str], start_time: datetime, end_time: datetime, 
                          metrics: List[str] = None) -> Dict[str, Any]:
        """
        Multi-node power analysis with connection pooling for efficiency.
        
        Args:
            hostnames: List of node hostnames
            start_time: Start timestamp
            end_time: End timestamp
            metrics: List of metrics to analyze (if None, uses defaults per node type)
        
        Returns:
            Dictionary with hostname as key and analysis results as value
        """
        # Group hostnames by database/schema to reuse connections
        db_groups = {}
        for hostname in hostnames:
            _, _, database, schema = self._get_node_type_and_query_func(hostname)
            key = (database, schema)
            if key not in db_groups:
                db_groups[key] = []
            db_groups[key].append(hostname)
        
        results = {}
        
        # Process each database group
        for (database, schema), group_hostnames in db_groups.items():
            db_connection = get_raw_database_connection(database, schema)
            
            try:
                for hostname in group_hostnames:
                    if metrics is None:
                        # Default metrics based on node type
                        node_type, _, host_database, host_schema = self._get_node_type_and_query_func(hostname)
                        if node_type == 'pdu':
                            host_metrics = ['pdu']
                        elif node_type == 'irc':
                            host_metrics = ['CompressorPower', 'CondenserFanPower', 'CoolDemand', 'CoolOutput', 'TotalAirSideCoolingDemand', 'TotalSensibleCoolingPower']
                        else:  # compute nodes - get from database
                            host_metrics = self._get_compute_power_metrics(host_database, host_schema)
                    else:
                        host_metrics = metrics
                    
                    queries, _, _ = self._get_power_metrics_with_joins(host_metrics, hostname, start_time, end_time)
                    
                    # Execute queries and combine results
                    all_data = []
                    for metric, query in queries.items():
                        df = pd.read_sql_query(query, engine)
                        if not df.empty:
                            df['metric'] = metric
                            all_data.append(df)
                    
                    if all_data:
                        combined_df = pd.concat(all_data, ignore_index=True)
                        processed_df = process_power_data(combined_df, hostname, start_time, end_time)
                        
                        results[hostname] = {
                            'data': processed_df,
                            'summary': self._calculate_power_summary(processed_df),
                            'hostname': hostname,
                            'start_time': start_time,
                            'end_time': end_time
                        }
                    else:
                        results[hostname] = {
                            'data': pd.DataFrame(),
                            'summary': {},
                            'hostname': hostname,
                            'start_time': start_time,
                            'end_time': end_time
                        }
                        
            finally:
                if db_connection:
                    db_connection.close()
        
        return results
    
    def _get_node_type_and_query_func(self, hostname: str):
        """Determine node type and return appropriate query function."""
        hostname_mapping = {
            'pdu': ('pdu', get_pdu_metrics_with_joins, 'infra', 'pdu'),
            'irc': ('irc', get_irc_metrics_with_joins, 'infra', 'irc'),
            'rpg': ('h100', get_compute_metrics_with_joins, 'h100', 'idrac'),
            'rpc': ('zen4', get_compute_metrics_with_joins, 'zen4', 'idrac')
        }
        
        for prefix, (node_type, query_func, database, schema) in hostname_mapping.items():
            if hostname.startswith(prefix):
                return node_type, query_func, database, schema
        
        raise ValueError(f"Invalid hostname: {hostname}")
    
    def _get_compute_power_metrics(self, database: str, schema: str) -> List[str]:
        """Get power metrics for compute nodes from the database."""
        from queries.compute.public import POWER_METRICS_QUERY_UNIT_IN_MW_W_KW
        
        db_connection = get_raw_database_connection(database, schema)
        try:
            # Create SQLAlchemy engine to avoid pandas warnings
            engine = create_engine(f"postgresql://{db_connection.info.user}:{db_connection.info.password}@{db_connection.info.host}:{db_connection.info.port}/{db_connection.info.dbname}")
            df = pd.read_sql_query(POWER_METRICS_QUERY_UNIT_IN_MW_W_KW, engine)
            return df['metric_id'].tolist()
        finally:
            if db_connection:
                db_connection.close()
    
    def _get_power_metrics_with_joins(self, metrics: List[str], hostname: str, 
                                     start_time: datetime, end_time: datetime):
        """Generate queries for multiple metrics for a given hostname."""
        node_type, query_func, database, schema = self._get_node_type_and_query_func(hostname)
        
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
    
    def _calculate_power_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate power summary statistics."""
        if df.empty:
            return {}
        
        summary = {
            'total_records': len(df),
            'time_range_hours': 0,
            'metrics': df['metric'].unique().tolist() if 'metric' in df.columns else []
        }
        
        if 'timestamp' in df.columns and len(df) > 1:
            time_range = df['timestamp'].max() - df['timestamp'].min()
            summary['time_range_hours'] = time_range.total_seconds() / 3600
        
        if 'power_w' in df.columns:
            summary.update({
                'avg_power_w': df['power_w'].mean(),
                'max_power_w': df['power_w'].max(),
                'min_power_w': df['power_w'].min(),
                'total_energy_kwh': df['cumulative_energy_kwh'].iloc[-1] if 'cumulative_energy_kwh' in df.columns else 0
            })
        
        return summary
    
    def display_summary(self, results: Dict[str, Any]):
        """Display power analysis summary."""
        if not results:
            print("❌ No data found")
            return
        
        print("📊 Power Analysis Summary")
        print("=" * 50)
        
        if 'data' in results:
            # Single node analysis
            df = results['data']
            summary = results['summary']
            
            print(f"🖥️  Hostname: {results['hostname']}")
            print(f"📅 Time Range: {results['start_time']} to {results['end_time']}")
            print(f"📊 Records: {summary.get('total_records', 0)}")
            print(f"⏱️  Duration: {summary.get('time_range_hours', 0):.2f} hours")
            
            if 'power_w' in df.columns:
                print(f"⚡ Average Power: {summary.get('avg_power_w', 0):.2f} W")
                print(f"⚡ Max Power: {summary.get('max_power_w', 0):.2f} W")
                print(f"⚡ Min Power: {summary.get('min_power_w', 0):.2f} W")
                print(f"🔋 Total Energy: {summary.get('total_energy_kwh', 0):.4f} kWh")
        else:
            # Multi-node analysis
            for hostname, node_results in results.items():
                if isinstance(node_results, dict) and 'summary' in node_results:
                    summary = node_results['summary']
                    print(f"\n🖥️  {hostname}:")
                    print(f"  📊 Records: {summary.get('total_records', 0)}")
                    if 'power_w' in summary:
                        print(f"  ⚡ Avg Power: {summary.get('avg_power_w', 0):.2f} W")
                        print(f"  🔋 Total Energy: {summary.get('total_energy_kwh', 0):.4f} kWh")
    
    def export_to_excel(self, results: Dict[str, Any], output_path: str):
        """Export results to Excel file."""
        from reporting.excel import ExcelReporter
        
        reporter = ExcelReporter()
        reporter.export_analysis_results(results, output_path)
    
    def export_to_csv(self, results: Dict[str, Any], output_path: str):
        """Export results to CSV file."""
        if 'data' in results:
            # Single node analysis
            results['data'].to_csv(output_path, index=False)
        else:
            # Multi-node analysis - combine all data
            all_data = []
            for hostname, node_results in results.items():
                if isinstance(node_results, dict) and 'data' in node_results:
                    df = node_results['data'].copy()
                    df['hostname'] = hostname
                    all_data.append(df)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df.to_csv(output_path, index=False)
    
    def export_to_json(self, results: Dict[str, Any], output_path: str):
        """Export results to JSON file."""
        import json
        
        # Convert datetime objects to strings for JSON serialization
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        # Convert DataFrame to dict for JSON serialization
        def convert_dataframe(df):
            if df.empty:
                return {}
            return df.to_dict('records')
        
        # Prepare results for JSON export
        json_results = {}
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                json_results[key] = convert_dataframe(value)
            elif isinstance(value, dict) and 'data' in value:
                json_results[key] = {
                    'summary': value['summary'],
                    'data': convert_dataframe(value['data']),
                    'hostname': value['hostname'],
                    'start_time': convert_datetime(value['start_time']),
                    'end_time': convert_datetime(value['end_time'])
                }
            else:
                json_results[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=convert_datetime)
