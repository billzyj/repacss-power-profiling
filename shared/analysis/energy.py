"""Shared energy math used by OOB and future IB paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from shared.utils.conversions import convert_power_series_to_watts


def compute_energy_kwh_for_hostname(
    df: pd.DataFrame,
    unit: str,
    hostname: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> float:
    """Compute energy in kWh for one hostname from timestamped power samples."""
    required_cols = {"timestamp", "value"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    host_df = df.copy()
    if "hostname" in host_df.columns:
        host_df = host_df[host_df["hostname"] == hostname].copy()
    elif len(host_df) > 0:
        host_df["hostname"] = hostname

    if host_df.empty:
        return 0.0

    host_df["timestamp"] = pd.to_datetime(host_df["timestamp"], utc=True, errors="coerce")
    host_df = host_df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if host_df.empty:
        return 0.0

    host_df["power_w"] = convert_power_series_to_watts(host_df["value"], unit)

    query_start = pd.to_datetime(start_time, utc=True) if start_time else None
    query_end = pd.to_datetime(end_time, utc=True) if end_time else None

    if query_start is not None and host_df["timestamp"].min() > query_start:
        boundary_start = pd.DataFrame(
            {
                "timestamp": [query_start],
                "hostname": [hostname],
                "value": [host_df["value"].iloc[0]],
                "power_w": [host_df["power_w"].iloc[0]],
            }
        )
        host_df = pd.concat([boundary_start, host_df], ignore_index=True)
        host_df = host_df.sort_values("timestamp").reset_index(drop=True)

    if query_end is not None and host_df["timestamp"].max() < query_end:
        boundary_end = pd.DataFrame(
            {
                "timestamp": [query_end],
                "hostname": [hostname],
                "value": [host_df["value"].iloc[-1]],
                "power_w": [host_df["power_w"].iloc[-1]],
            }
        )
        host_df = pd.concat([host_df, boundary_end], ignore_index=True)
        host_df = host_df.sort_values("timestamp").reset_index(drop=True)

    total_energy_joules = 0.0
    for index in range(1, len(host_df)):
        time_diff = (host_df["timestamp"].iloc[index] - host_df["timestamp"].iloc[index - 1]).total_seconds()
        avg_power = (host_df["power_w"].iloc[index] + host_df["power_w"].iloc[index - 1]) / 2.0
        total_energy_joules += avg_power * time_diff

    return total_energy_joules / 3_600_000.0


@dataclass
class EnergyCalculator:
    """Pure energy calculator wrapper retained for compatibility."""

    database: Optional[str] = None

    def calculate_energy_for_hostname(
        self,
        df: pd.DataFrame,
        unit: str,
        hostname: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> float:
        return compute_energy_kwh_for_hostname(df, unit, hostname, start_time, end_time)

