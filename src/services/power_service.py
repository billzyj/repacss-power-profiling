#!/usr/bin/env python3
"""
Power analysis service layer
Handles high-level power analysis operations without direct database coupling
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
import os
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.power import PowerAnalyzer
from analysis.energy import EnergyCalculator
from queries.manager import QueryManager
from database.connection_pool import close_all_pools

# Configure logger
logger = logging.getLogger(__name__)


class PowerAnalysisService:
    """High-level service for power analysis operations"""
    
    def __init__(self, database: str):
        self.database = database
        self.query_manager = QueryManager(database)
        self.power_analyzer = PowerAnalyzer(database)
        self.energy_calculator = EnergyCalculator(database)
    
    def get_system_overview(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get system-wide power overview for the specified database.
        
        Args:
            hours: Number of hours to analyze
        
        Returns:
            System overview with key metrics
        """
        end_time = datetime.now()
        start_time = end_time - pd.Timedelta(hours=hours)
        
        try:
            # Get available nodes
            nodes = self._get_available_nodes()
            
            # Get system metrics
            system_metrics = self._get_system_metrics(start_time, end_time)
            
            return {
                'database': self.database,
                'time_range': {
                    'start': start_time,
                    'end': end_time,
                    'hours': hours
                },
                'nodes': nodes,
                'metrics': system_metrics,
                'summary': self._calculate_system_summary(system_metrics)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_node_power(self, hostname: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Analyze power consumption for a specific node.
        
        Args:
            hostname: Node hostname
            start_time: Start timestamp
            end_time: End timestamp
        
        Returns:
            Node power analysis results
        """
        try:
            # Run power analysis
            power_results = self.power_analyzer.analyze_power(hostname, start_time, end_time)
            
            # Calculate energy consumption
            energy_results = self.energy_calculator.calculate_energy(hostname, start_time, end_time)
            
            return {
                'hostname': hostname,
                'time_range': {
                    'start': start_time,
                    'end': end_time
                },
                'power_analysis': power_results,
                'energy_consumption': energy_results,
                'summary': self._create_node_summary(power_results, energy_results)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_rack_power(self, rack_number: int, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Analyze power consumption for an entire rack.
        
        Args:
            rack_number: Rack number (91-97)
            start_time: Start timestamp
            end_time: End timestamp
        
        Returns:
            Rack power analysis results
        """
        try:
            # Get rack nodes
            rack_nodes = self._get_rack_nodes(rack_number)
            
            # Analyze each node
            node_results = {}
            total_energy = 0.0
            
            for node in rack_nodes:
                node_analysis = self.analyze_node_power(node, start_time, end_time)
                if 'error' not in node_analysis:
                    node_results[node] = node_analysis
                    if 'energy_consumption' in node_analysis:
                        total_energy += sum(node_analysis['energy_consumption'].values())
            
            return {
                'rack_number': rack_number,
                'time_range': {
                    'start': start_time,
                    'end': end_time
                },
                'nodes': list(rack_nodes),
                'node_analyses': node_results,
                'total_energy_kwh': total_energy,
                'summary': self._create_rack_summary(node_results, total_energy)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_available_metrics(self) -> List[str]:
        """Get list of available power metrics for the database"""
        try:
            metrics_df = self.query_manager.get_power_metrics_definition()
            return metrics_df['metric_id'].tolist() if not metrics_df.empty else []
        except Exception as e:
            return []
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics"""
        try:
            return self.query_manager.get_database_info()
        except Exception as e:
            return {'error': str(e)}
    
    def _get_available_nodes(self) -> List[str]:
        """Get list of available nodes in the database"""
        # This would query the database for available nodes
        # For now, return empty list - would need to implement based on database schema
        return []
    
    def _get_system_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get system-wide metrics"""
        try:
            # Get power metrics for all nodes
            all_metrics = self.query_manager.get_power_metrics(
                hostname=None, start_time=start_time, end_time=end_time, limit=1000
            )
            
            return {
                'total_records': len(all_metrics),
                'metrics_available': len(all_metrics['metric'].unique()) if not all_metrics.empty else 0,
                'time_range_hours': (end_time - start_time).total_seconds() / 3600
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_system_summary(self, system_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate system summary statistics"""
        if 'error' in system_metrics:
            return system_metrics
        
        return {
            'status': 'healthy' if system_metrics.get('total_records', 0) > 0 else 'no_data',
            'total_records': system_metrics.get('total_records', 0),
            'metrics_count': system_metrics.get('metrics_available', 0),
            'time_range_hours': system_metrics.get('time_range_hours', 0)
        }
    
    def _create_node_summary(self, power_results: Dict[str, Any], energy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary for node analysis"""
        summary = {
            'status': 'success' if power_results else 'no_data',
            'has_power_data': bool(power_results),
            'has_energy_data': bool(energy_results)
        }
        
        if power_results and 'summary' in power_results:
            power_summary = power_results['summary']
            summary.update({
                'total_records': power_summary.get('total_records', 0),
                'avg_power_w': power_summary.get('avg_power_w', 0),
                'max_power_w': power_summary.get('max_power_w', 0),
                'total_energy_kwh': power_summary.get('total_energy_kwh', 0)
            })
        
        if energy_results:
            summary['total_energy_kwh'] = sum(energy_results.values())
        
        return summary
    
    def _create_rack_summary(self, node_results: Dict[str, Any], total_energy: float) -> Dict[str, Any]:
        """Create summary for rack analysis"""
        return {
            'status': 'success' if node_results else 'no_data',
            'nodes_analyzed': len(node_results),
            'total_energy_kwh': total_energy,
            'avg_energy_per_node': total_energy / len(node_results) if node_results else 0
        }
    
    def _get_rack_nodes(self, rack_number: int) -> List[str]:
        """Get nodes for a specific rack"""
        # Import rack configurations
        from constants.nodes import (
            RACK_91_COMPUTE_NODES, RACK_91_PD_NODES,
            RACK_92_COMPUTE_NODES, RACK_92_PD_NODES,
            RACK_93_COMPUTE_NODES, RACK_93_PD_NODES,
            RACK_94_COMPUTE_NODES, RACK_94_PD_NODES,
            RACK_95_COMPUTE_NODES, RACK_95_PD_NODES,
            RACK_96_COMPUTE_NODES, RACK_96_PD_NODES,
            RACK_97_COMPUTE_NODES, RACK_97_PDU_NODES
        )
        
        rack_configs = {
            91: RACK_91_COMPUTE_NODES + RACK_91_PD_NODES,
            92: RACK_92_COMPUTE_NODES + RACK_92_PD_NODES,
            93: RACK_93_COMPUTE_NODES + RACK_93_PD_NODES,
            94: RACK_94_COMPUTE_NODES + RACK_94_PD_NODES,
            95: RACK_95_COMPUTE_NODES + RACK_95_PD_NODES,
            96: RACK_96_COMPUTE_NODES + RACK_96_PD_NODES,
            97: RACK_97_COMPUTE_NODES + RACK_97_PDU_NODES
        }
        
        return rack_configs.get(rack_number, [])
    
    def calculate_pue(self, start_time: datetime, end_time: datetime, max_gap_minutes: int = None) -> Dict[str, Any]:
        """
        Calculate PUE (Power Usage Effectiveness) for the entire cluster.
        
        PUE = (Total IRC Power Consumption + Total PDU Power Consumption) / Total PDU Power Consumption
        
        Optimized to use batch queries with connection reuse for efficiency.

        Data quality rule:
        - If the gap between consecutive readings exceeds 4x the sensing frequency for that node type,
          that interval is treated as 0 power (skip trapezoidal integration across the gap).
        - IRC: 4 * 120s = 480s (8 minutes)
        - PDU: 4 * 60s = 240s (4 minutes)
        - Compute: 4 * 5s = 20s
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            max_gap_minutes: [Deprecated] Now automatically calculated based on node type (4x sensing frequency)
        
        Returns:
            Dictionary with PUE calculation results
        """
        from constants.nodes import IRC_NODES, PDU_NODES
        from constants.metrics import IRC_POWER_METRICS, PDU_POWER_METRICS, IRC_SENSING_FREQUENCY, PDU_SENSING_FREQUENCY
        from database.database import get_raw_database_connection
        from queries.infra.irc_pdu import get_irc_metrics_with_joins, get_pdu_metrics_with_joins
        from utils.conversions import convert_power_series_to_watts
        from sqlalchemy import create_engine
        import pandas as pd

        def _energy_kwh_for_host_df(host_df: "pd.DataFrame", unit: str, hostname: str) -> float:
            """Trapezoidal energy (kWh) for a hostname, skipping intervals with large gaps (> 4x sensing frequency)."""
            if host_df is None or host_df.empty:
                return 0.0

            # Get max gap threshold based on node type (4x sensing frequency)
            from utils.node_detection import get_node_type_and_query_func
            node_type, _, _, _ = get_node_type_and_query_func(hostname)
            if node_type == 'irc':
                max_gap_seconds = 4.0 * IRC_SENSING_FREQUENCY  # 4 * 120 = 480 seconds (8 minutes)
            elif node_type == 'pdu':
                max_gap_seconds = 4.0 * PDU_SENSING_FREQUENCY  # 4 * 60 = 240 seconds (4 minutes)
            else:  # compute
                from constants.metrics import COMPUTE_SENSING_FREQUENCY
                max_gap_seconds = 4.0 * COMPUTE_SENSING_FREQUENCY  # 4 * 5 = 20 seconds

            df = host_df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
            if len(df) < 2:
                return 0.0

            df["power_w"] = convert_power_series_to_watts(df["value"], unit).astype("float64")
            ts = df["timestamp"].to_list()
            pw = df["power_w"].to_list()

            total_j = 0.0
            for i in range(1, len(ts)):
                t0 = ts[i - 1]
                t1 = ts[i]
                if t1 <= t0:
                    continue
                dt = (t1 - t0).total_seconds()
                # Data quality: don't bridge large gaps (> 4x sensing frequency)
                if dt > max_gap_seconds:
                    # Treat missing interval as 0 power (skip)
                    continue
                total_j += ((float(pw[i - 1]) + float(pw[i])) / 2.0) * dt

            return float(total_j) / 3_600_000.0
        
        try:
            # Convert datetime to string for SQL queries
            start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
            end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Calculate total IRC and PDU power consumption (energy) - single connection for both
            total_irc_energy_kwh = 0.0
            total_pdu_energy_kwh = 0.0
            irc_node_results = {}
            pdu_node_results = {}
            
            # Use a single connection to infra database (schema doesn't matter since we use schema.table format)
            db_connection = get_raw_database_connection('infra', 'public')
            try:
                engine = create_engine(
                    f"postgresql://{db_connection.info.user}:{db_connection.info.password}@"
                    f"{db_connection.info.host}:{db_connection.info.port}/{db_connection.info.dbname}"
                )
                
                # Query all IRC metrics for all IRC nodes at once
                # Only CompressorPower (kW) and CondenserFanPower (W) represent actual power consumption
                logger.info(f"Querying IRC metrics: {IRC_POWER_METRICS}")
                for metric in IRC_POWER_METRICS:
                    try:
                        # Query without hostname filter to get all nodes at once
                        query = get_irc_metrics_with_joins(metric, hostname=None, start_time=start_time_str, end_time=end_time_str)
                        logger.debug(f"Executing query for {metric}: {query[:200]}...")
                        df = pd.read_sql_query(query, engine)
                        
                        logger.info(f"IRC metric {metric}: Retrieved {len(df)} rows")
                        if not df.empty:
                            # Get unit from the data (should be 'kW' for CompressorPower, 'W' for CondenserFanPower)
                            unit = df['units'].iloc[0] if 'units' in df.columns else 'W'
                            logger.info(f"IRC metric {metric}: Unit = {unit}")
                            
                            # Check available hostnames in the data
                            available_hostnames = df['hostname'].unique() if 'hostname' in df.columns else []
                            logger.info(f"IRC metric {metric}: Found hostnames: {list(available_hostnames)}")
                            
                            # Verify unit matches expected unit for this metric
                            if metric == 'CompressorPower' and unit.lower() != 'kw':
                                logger.warning(f"CompressorPower expected unit 'kW' but got '{unit}'")
                            elif metric == 'CondenserFanPower' and unit.lower() != 'w':
                                logger.warning(f"CondenserFanPower expected unit 'W' but got '{unit}'")
                            
                            # Calculate energy for each IRC node
                            for irc_node in IRC_NODES:
                                if irc_node not in irc_node_results:
                                    irc_node_results[irc_node] = {'energy_kwh': 0.0, 'metrics': {}}

                                node_df = df[df["hostname"] == irc_node].copy() if "hostname" in df.columns else None
                                node_energy = _energy_kwh_for_host_df(node_df, unit, irc_node)
                                logger.debug(f"IRC node {irc_node} metric {metric}: {node_energy:.6f} kWh")
                                irc_node_results[irc_node]['metrics'][metric] = node_energy
                                irc_node_results[irc_node]['energy_kwh'] += node_energy
                        else:
                            # No data for this metric - log but continue
                            logger.warning(f"No data found for IRC metric {metric} in time range {start_time_str} to {end_time_str}")
                    except Exception as e:
                        # Log error but continue with other metrics
                        logger.error(f"Error querying IRC metric {metric}: {e}", exc_info=True)
                        for irc_node in IRC_NODES:
                            if irc_node not in irc_node_results:
                                irc_node_results[irc_node] = {'energy_kwh': 0.0, 'metrics': {}, 'error': str(e)}
                
                # Query PDU metrics for all PDU nodes at once (using the same connection)
                try:
                    # Query without hostname filter to get all nodes at once
                    query = get_pdu_metrics_with_joins(hostname=None, start_time=start_time_str, end_time=end_time_str)
                    df = pd.read_sql_query(query, engine)
                    
                    if not df.empty:
                        # Get unit (PDU always uses 'W')
                        unit = 'W'
                        
                        # Calculate energy for each PDU node
                        for pdu_node in PDU_NODES:
                            if pdu_node not in pdu_node_results:
                                pdu_node_results[pdu_node] = {'energy_kwh': 0.0, 'metrics': {}}

                            node_df = df[df["hostname"] == pdu_node].copy() if "hostname" in df.columns else None
                            node_energy = _energy_kwh_for_host_df(node_df, unit, pdu_node)
                            pdu_node_results[pdu_node]['metrics']['pdu'] = node_energy
                            pdu_node_results[pdu_node]['energy_kwh'] = node_energy
                            total_pdu_energy_kwh += node_energy
                except Exception as e:
                    # Log error for all PDU nodes
                    for pdu_node in PDU_NODES:
                        pdu_node_results[pdu_node] = {
                            'energy_kwh': 0.0,
                            'error': str(e)
                        }
            finally:
                if db_connection:
                    db_connection.close()
            
            # Calculate total IRC energy from node results
            for irc_node, node_data in irc_node_results.items():
                if 'error' not in node_data:
                    total_irc_energy_kwh += node_data['energy_kwh']
            
            # Calculate PUE with special handling for edge cases:
            # - 0/0 (both IRC and PDU are 0): PUE = 0 (entire cluster down)
            # - 0/PDU (IRC is 0, PDU > 0): PUE = 1 (IRC down, PDU operational)
            # - IRC/PDU (both > 0): PUE = (IRC + PDU) / PDU (normal case)
            if total_pdu_energy_kwh == 0:
                if total_irc_energy_kwh == 0:
                    pue = 0.0  # Entire cluster down
                else:
                    # Edge case: IRC has data but PDU doesn't (shouldn't happen normally)
                    pue = None
            elif total_irc_energy_kwh == 0:
                pue = 1.0  # IRC down, PDU operational
            else:
                pue = (total_irc_energy_kwh + total_pdu_energy_kwh) / total_pdu_energy_kwh  # Normal case
            
            # Calculate time duration in hours
            duration_hours = (end_time - start_time).total_seconds() / 3600.0
            
            # Calculate average power (energy / time)
            avg_irc_power_kw = total_irc_energy_kwh / duration_hours if duration_hours > 0 else 0.0
            avg_pdu_power_kw = total_pdu_energy_kwh / duration_hours if duration_hours > 0 else 0.0
            avg_total_power_kw = (total_irc_energy_kwh + total_pdu_energy_kwh) / duration_hours if duration_hours > 0 else 0.0
            
            return {
                'time_range': {
                    'start': start_time,
                    'end': end_time,
                    'duration_hours': duration_hours
                },
                'irc': {
                    'total_energy_kwh': total_irc_energy_kwh,
                    'avg_power_kw': avg_irc_power_kw,
                    'nodes': irc_node_results,
                    'node_count': len(IRC_NODES),
                    'successful_nodes': len([n for n in irc_node_results.values() if 'error' not in n])
                },
                'pdu': {
                    'total_energy_kwh': total_pdu_energy_kwh,
                    'avg_power_kw': avg_pdu_power_kw,
                    'nodes': pdu_node_results,
                    'node_count': len(PDU_NODES),
                    'successful_nodes': len([n for n in pdu_node_results.values() if 'error' not in n])
                },
                'pue': {
                    'value': pue,
                    'total_energy_kwh': total_irc_energy_kwh + total_pdu_energy_kwh,
                    'avg_power_kw': avg_total_power_kw,
                    'formula': 'PUE = (IRC Energy + PDU Energy) / PDU Energy'
                },
                'summary': {
                    'status': 'success' if pue is not None else 'error',
                    'pue': pue,
                    'irc_energy_kwh': total_irc_energy_kwh,
                    'pdu_energy_kwh': total_pdu_energy_kwh,
                    'total_energy_kwh': total_irc_energy_kwh + total_pdu_energy_kwh
                }
            }
            
        except Exception as e:
            return {'error': str(e)}

    def calculate_pue_daily(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Calculate daily PUE for a time range and return a table suitable for Excel export.

        This is optimized to avoid per-day SQL by querying the full time range once:
        - IRC: 2 metrics (CompressorPower, CondenserFanPower)
        - PDU: 1 metric (pdu)

        Data quality rule:
        - If the gap between consecutive readings exceeds 4x the sensing frequency for that node type,
          that interval is treated as 0 power (skip trapezoidal integration across the gap).
        - IRC: 4 * 120s = 480s (8 minutes)
        - PDU: 4 * 60s = 240s (4 minutes)
        - Compute: 4 * 5s = 20s

        Output columns:
        - date (YYYY-MM-DD, UTC)
        - irc_energy_kwh
        - pdu_energy_kwh
        - pue
        """
        from constants.nodes import IRC_NODES, PDU_NODES
        from constants.metrics import IRC_POWER_METRICS
        from database.database import get_raw_database_connection
        from queries.infra.irc_pdu import get_irc_metrics_with_joins, get_pdu_metrics_with_joins
        from utils.conversions import convert_power_series_to_watts
        from sqlalchemy import create_engine
        import pandas as pd

        from constants.metrics import IRC_SENSING_FREQUENCY, PDU_SENSING_FREQUENCY, COMPUTE_SENSING_FREQUENCY
        
        def _get_max_gap_seconds_for_node(hostname: str) -> float:
            """Get max gap threshold (4x sensing frequency) for a node type"""
            from utils.node_detection import get_node_type_and_query_func
            node_type, _, _, _ = get_node_type_and_query_func(hostname)
            if node_type == 'irc':
                return 4.0 * IRC_SENSING_FREQUENCY  # 4 * 120 = 480 seconds (8 minutes)
            elif node_type == 'pdu':
                return 4.0 * PDU_SENSING_FREQUENCY  # 4 * 60 = 240 seconds (4 minutes)
            else:  # compute
                return 4.0 * COMPUTE_SENSING_FREQUENCY  # 4 * 5 = 20 seconds

        def _daily_energy_kwh_for_host_df(
            host_df: "pd.DataFrame",
            unit: str,
            start_ts: "pd.Timestamp",
            end_ts: "pd.Timestamp",
            max_gap_seconds: float,
        ) -> dict:
            """
            Compute daily energy (kWh) for a single hostname time series using trapezoidal integration,
            splitting intervals that cross midnight (UTC) via linear interpolation.
            """
            if host_df is None or host_df.empty:
                return {}

            df = host_df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
            if df.empty:
                return {}

            df["power_w"] = convert_power_series_to_watts(df["value"], unit)

            # Clip to [start_ts, end_ts] - only use actual data points, don't extrapolate
            df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].copy()
            if df.empty:
                return {}

            # Only use actual data points - don't add boundary points that would "borrow" power from adjacent days
            # This ensures that days with no data return 0 energy, not energy borrowed from neighboring days
            df = df[["timestamp", "power_w"]].copy()
            df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
            
            # Need at least 2 points to calculate energy (trapezoidal rule)
            if len(df) < 2:
                return {}

            out_j = {}
            ts = df["timestamp"].to_list()
            pw = df["power_w"].astype("float64").to_list()

            for i in range(1, len(ts)):
                t0 = ts[i - 1]
                t1 = ts[i]
                if t1 <= t0:
                    continue
                # Data quality: don't bridge large gaps
                if (t1 - t0).total_seconds() > max_gap_seconds:
                    continue
                p0 = float(pw[i - 1])
                p1 = float(pw[i])

                # Split at UTC midnights
                cur_t = t0
                cur_p = p0
                next_midnight = (t0.normalize() + pd.Timedelta(days=1))
                while next_midnight < t1:
                    frac = (next_midnight - t0).total_seconds() / (t1 - t0).total_seconds()
                    pb = p0 + (p1 - p0) * frac
                    dt = (next_midnight - cur_t).total_seconds()
                    e_j = ((cur_p + pb) / 2.0) * dt
                    key = cur_t.date().isoformat()
                    out_j[key] = out_j.get(key, 0.0) + e_j
                    cur_t = next_midnight
                    cur_p = pb
                    next_midnight = next_midnight + pd.Timedelta(days=1)

                # Last segment
                dt = (t1 - cur_t).total_seconds()
                e_j = ((cur_p + p1) / 2.0) * dt
                key = cur_t.date().isoformat()
                out_j[key] = out_j.get(key, 0.0) + e_j

            # Joules -> kWh
            return {k: float(v) / 3_600_000.0 for k, v in out_j.items()}

        try:
            start_ts = pd.Timestamp(start_time, tz="UTC")
            end_ts = pd.Timestamp(end_time, tz="UTC")
            if end_ts <= start_ts:
                return {"error": "end_time must be after start_time"}

            start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
            end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

            # Build the full date index (UTC) we will report
            all_dates = pd.date_range(start_ts.normalize(), end_ts.normalize(), freq="D", tz="UTC")
            date_keys = [d.date().isoformat() for d in all_dates]

            irc_daily_total = {k: 0.0 for k in date_keys}
            pdu_daily_total = {k: 0.0 for k in date_keys}

            db_connection = get_raw_database_connection("infra", "public")
            try:
                engine = create_engine(
                    f"postgresql://{db_connection.info.user}:{db_connection.info.password}@"
                    f"{db_connection.info.host}:{db_connection.info.port}/{db_connection.info.dbname}"
                )

                # IRC: query per metric (2 queries) and accumulate per day over IRC nodes
                for metric in IRC_POWER_METRICS:
                    query = get_irc_metrics_with_joins(metric, hostname=None, start_time=start_time_str, end_time=end_time_str)
                    df = pd.read_sql_query(query, engine)
                    if df.empty:
                        continue

                    unit = df["units"].iloc[0] if "units" in df.columns and len(df["units"]) else "W"
                    grouped = {h: g for h, g in df.groupby("hostname")} if "hostname" in df.columns else {}
                    for host in IRC_NODES:
                        host_df = grouped.get(host)
                        max_gap_sec = _get_max_gap_seconds_for_node(host)
                        daily = _daily_energy_kwh_for_host_df(host_df, unit, start_ts, end_ts, max_gap_sec)
                        for k, v in daily.items():
                            if k in irc_daily_total:
                                irc_daily_total[k] += v

                # PDU: single query
                query = get_pdu_metrics_with_joins(hostname=None, start_time=start_time_str, end_time=end_time_str)
                df = pd.read_sql_query(query, engine)
                if not df.empty:
                    unit = "W"  # defined by query
                    grouped = {h: g for h, g in df.groupby("hostname")} if "hostname" in df.columns else {}
                    for host in PDU_NODES:
                        host_df = grouped.get(host)
                        max_gap_sec = _get_max_gap_seconds_for_node(host)
                        daily = _daily_energy_kwh_for_host_df(host_df, unit, start_ts, end_ts, max_gap_sec)
                        for k, v in daily.items():
                            if k in pdu_daily_total:
                                pdu_daily_total[k] += v
            finally:
                if db_connection:
                    db_connection.close()

            # Assemble output table
            rows = []
            for k in date_keys:
                irc_kwh = float(irc_daily_total.get(k, 0.0))
                pdu_kwh = float(pdu_daily_total.get(k, 0.0))
                
                # Calculate PUE with special handling for edge cases:
                # - 0/0 (both IRC and PDU are 0): PUE = 0 (entire cluster down)
                # - 0/PDU (IRC is 0, PDU > 0): PUE = 1 (IRC down, PDU operational)
                # - IRC/PDU (both > 0): PUE = (IRC + PDU) / PDU (normal case)
                if pdu_kwh == 0:
                    if irc_kwh == 0:
                        pue = 0.0  # Entire cluster down
                    else:
                        # Edge case: IRC has data but PDU doesn't (shouldn't happen normally)
                        # Set to None to indicate invalid state
                        pue = None
                elif irc_kwh == 0:
                    pue = 1.0  # IRC down, PDU operational
                else:
                    pue = (irc_kwh + pdu_kwh) / pdu_kwh  # Normal case
                
                rows.append(
                    {
                        "date": k,
                        "irc_energy_kwh": irc_kwh,
                        "pdu_energy_kwh": pdu_kwh,
                        "pue": pue,
                    }
                )

            out_df = pd.DataFrame(rows)
            return {
                "time_range": {"start": start_time, "end": end_time},
                "data": out_df,
                "summary": {
                    "days": len(out_df),
                    "start_date": date_keys[0] if date_keys else None,
                    "end_date": date_keys[-1] if date_keys else None,
                },
            }
        except Exception as e:
            return {"error": str(e)}
    
    def __del__(self):
        """Cleanup connections when service is destroyed"""
        try:
            close_all_pools()
        except:
            pass
