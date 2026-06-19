#!/usr/bin/env python3
"""Export and plot monthly TotalFanPower distributions for REPACSS racks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import os
from pathlib import Path
import sys
import tempfile
import warnings

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oob.backends.monster_db.connection_pool import get_pooled_connection
from shared.constants.nodes import RACK_CONFIGS
from src.queries.compute.idrac import get_compute_metrics_with_joins


METRIC_ID = "TotalFanPower"
SYSTEM_INPUT_METRIC_ID = "SystemInputPower"
DEFAULT_RACKS = tuple(sorted(RACK_CONFIGS))
POSITIONAL_DEFAULT_SAMPLE_SECONDS = 300
POSITIONAL_GRANULARITIES = ("hour", "day")


@dataclass(frozen=True)
class QueryTarget:
    rack: int
    hostname: str
    database: str
    schema: str = "idrac"


@dataclass(frozen=True)
class QueryGroup:
    rack: int
    database: str
    schema: str
    hostnames: tuple[str, ...]


@dataclass(frozen=True)
class PositionalQueryGroup:
    rack: int
    database: str
    schema: str
    grouped_hostnames: dict[str, tuple[str, ...]]


def _format_db_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _parse_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def _default_positional_window(today: date | None = None) -> tuple[datetime, datetime]:
    today = today or datetime.now().date()
    end_day = today - timedelta(days=7)
    start_day = today - timedelta(days=14)
    return datetime.combine(start_day, datetime.min.time()), datetime.combine(end_day, datetime.min.time())


def infer_database(hostname: str) -> str:
    if hostname.startswith("rpc"):
        return "zen4"
    if hostname.startswith("rpg"):
        return "h100"
    raise ValueError(f"Unsupported compute hostname: {hostname}")


def build_query_targets(racks: list[int] | tuple[int, ...] = DEFAULT_RACKS) -> list[QueryTarget]:
    targets: list[QueryTarget] = []
    for rack in racks:
        rack_config = RACK_CONFIGS.get(rack)
        if rack_config is None:
            raise ValueError(f"Unknown rack: {rack}")
        for hostname in rack_config["compute_nodes"]:
            targets.append(QueryTarget(rack=rack, hostname=hostname, database=infer_database(hostname)))
    return targets


def build_query_groups(targets: list[QueryTarget]) -> list[QueryGroup]:
    grouped: dict[tuple[int, str, str], list[str]] = {}
    for target in targets:
        grouped.setdefault((target.rack, target.database, target.schema), []).append(target.hostname)
    return [
        QueryGroup(rack=rack, database=database, schema=schema, hostnames=tuple(hostnames))
        for (rack, database, schema), hostnames in sorted(grouped.items())
    ]


def _node_slot(hostname: str) -> int:
    return int(hostname.rsplit("-", 1)[1])


def build_position_group_map(rack: int, hostnames: list[str] | tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    if rack == 93:
        return {"gpu_all": tuple(sorted(hostnames, key=_node_slot))}
    lower = tuple(sorted((hostname for hostname in hostnames if 1 <= _node_slot(hostname) <= 7), key=_node_slot))
    upper = tuple(sorted((hostname for hostname in hostnames if 13 <= _node_slot(hostname) <= 20), key=_node_slot))
    groups: dict[str, tuple[str, ...]] = {}
    if lower:
        groups["lower"] = lower
    if upper:
        groups["upper"] = upper
    return groups


def build_positional_query_groups(racks: list[int] | tuple[int, ...]) -> list[PositionalQueryGroup]:
    groups: list[PositionalQueryGroup] = []
    for rack in racks:
        hostnames = tuple(RACK_CONFIGS[rack]["compute_nodes"])
        grouped_hostnames = build_position_group_map(rack, hostnames)
        flat_hostnames = [hostname for names in grouped_hostnames.values() for hostname in names]
        if not flat_hostnames:
            continue
        databases = {infer_database(hostname) for hostname in flat_hostnames}
        if len(databases) != 1:
            raise ValueError(f"Rack {rack} positional query spans multiple databases: {sorted(databases)}")
        groups.append(
            PositionalQueryGroup(
                rack=rack,
                database=databases.pop(),
                schema="idrac",
                grouped_hostnames=grouped_hostnames,
            )
        )
    return groups


def _quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _quote_sql_strings(values: tuple[str, ...] | list[str]) -> str:
    return ", ".join(_quote_sql_string(value) for value in values)


def _position_case_sql(grouped_hostnames: dict[str, tuple[str, ...]]) -> str:
    lines = ["CASE"]
    for group_name, hostnames in grouped_hostnames.items():
        lines.append(f"            WHEN n.hostname IN ({_quote_sql_strings(hostnames)}) THEN '{group_name}'")
    lines.append("            ELSE NULL")
    lines.append("        END")
    return "\n".join(lines)


def build_positional_query(
    rack: int,
    grouped_hostnames: dict[str, tuple[str, ...]],
    granularity: str,
    start_time: datetime,
    end_time: datetime,
    sample_seconds: int = POSITIONAL_DEFAULT_SAMPLE_SECONDS,
) -> str:
    if granularity not in POSITIONAL_GRANULARITIES:
        raise ValueError(f"Unsupported granularity: {granularity}")
    hostnames = [hostname for group in grouped_hostnames.values() for hostname in group]
    if not hostnames:
        raise ValueError("At least one hostname is required.")
    sample_filter = ""
    if sample_seconds > 0:
        sample_filter = f"      AND MOD(EXTRACT(EPOCH FROM fan.timestamp)::bigint, {sample_seconds}) = 0\n"
    return f"""
    WITH samples AS (
        SELECT
            {rack} AS rack,
            date_trunc('{granularity}', fan.timestamp) AS time_bin,
            n.hostname,
            {_position_case_sql(grouped_hostnames)} AS position_group,
            CASE
                WHEN LOWER(COALESCE(fan_units.units, 'W')) = 'mw' THEN fan.value / 1000.0
                WHEN LOWER(COALESCE(fan_units.units, 'W')) = 'kw' THEN fan.value * 1000.0
                ELSE fan.value
            END AS fan_watts,
            CASE
                WHEN LOWER(COALESCE(sys_units.units, 'W')) = 'mw' THEN sys.value / 1000.0
                WHEN LOWER(COALESCE(sys_units.units, 'W')) = 'kw' THEN sys.value * 1000.0
                ELSE sys.value
            END AS system_input_watts
        FROM idrac.totalfanpower fan
        LEFT JOIN idrac.systeminputpower sys
            ON fan.nodeid = sys.nodeid
           AND fan.timestamp = sys.timestamp
        LEFT JOIN public.nodes n ON fan.nodeid = n.nodeid
        LEFT JOIN public.metrics_definition fan_units ON LOWER(fan_units.metric_id) = LOWER('{METRIC_ID}')
        LEFT JOIN public.metrics_definition sys_units ON LOWER(sys_units.metric_id) = LOWER('{SYSTEM_INPUT_METRIC_ID}')
        WHERE n.hostname IN ({_quote_sql_strings(hostnames)})
          AND fan.timestamp BETWEEN '{_format_db_time(start_time)}' AND '{_format_db_time(end_time)}'
          AND fan.value IS NOT NULL
{sample_filter.rstrip()}
    )
    SELECT
        rack,
        time_bin,
        position_group,
        COUNT(DISTINCT hostname) AS node_count,
        COUNT(*) AS sample_count,
        percentile_cont(0.50) WITHIN GROUP (ORDER BY fan_watts) AS fan_median_watts,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY fan_watts) AS fan_p95_watts,
        AVG(fan_watts) AS fan_mean_watts,
        percentile_cont(0.50) WITHIN GROUP (ORDER BY system_input_watts) AS system_input_median_watts,
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY CASE
                WHEN system_input_watts > 0 THEN fan_watts / system_input_watts
                ELSE NULL
            END
        ) AS fan_ratio_median,
        percentile_cont(0.95) WITHIN GROUP (
            ORDER BY CASE
                WHEN system_input_watts > 0 THEN fan_watts / system_input_watts
                ELSE NULL
            END
        ) AS fan_ratio_p95
    FROM samples
    WHERE position_group IS NOT NULL
    GROUP BY rack, time_bin, position_group
    ORDER BY rack ASC, time_bin ASC, position_group ASC;
    """


def build_total_fan_power_query(hostname: str, start_time: datetime, end_time: datetime) -> str:
    return get_compute_metrics_with_joins(
        METRIC_ID,
        hostname=hostname,
        start_time=_format_db_time(start_time),
        end_time=_format_db_time(end_time),
        limit=0,
    )


def build_total_fan_power_batch_query(hostnames: tuple[str, ...] | list[str], start_time: datetime, end_time: datetime) -> str:
    if not hostnames:
        raise ValueError("At least one hostname is required.")
    table_name = METRIC_ID.lower()
    return f"""
    SELECT
        p.timestamp,
        n.hostname,
        s.source,
        f.fqdd,
        p.value,
        m.units
    FROM idrac.{table_name} p
    LEFT JOIN public.nodes n ON p.nodeid = n.nodeid
    LEFT JOIN public.source s ON p.source = s.id
    LEFT JOIN public.fqdd f ON p.fqdd = f.id
    LEFT JOIN public.metrics_definition m ON LOWER(m.metric_id) = LOWER('{METRIC_ID}')
    WHERE n.hostname IN ({_quote_sql_strings(hostnames)})
      AND p.timestamp BETWEEN '{_format_db_time(start_time)}' AND '{_format_db_time(end_time)}'
    ORDER BY n.hostname ASC, p.timestamp ASC;
    """


def build_total_fan_power_percentile_query(
    hostnames: tuple[str, ...] | list[str],
    start_time: datetime,
    end_time: datetime,
    sample_seconds: int = 300,
) -> str:
    if not hostnames:
        raise ValueError("At least one hostname is required.")
    table_name = METRIC_ID.lower()
    sample_filter = ""
    if sample_seconds > 0:
        sample_filter = f"          AND MOD(EXTRACT(EPOCH FROM p.timestamp)::bigint, {sample_seconds}) = 0\n"
    return f"""
    WITH samples AS (
        SELECT
            n.hostname,
            CASE
                WHEN LOWER(COALESCE(m.units, 'W')) = 'mw' THEN p.value / 1000.0
                WHEN LOWER(COALESCE(m.units, 'W')) = 'kw' THEN p.value * 1000.0
                ELSE p.value
            END AS value_watts
        FROM idrac.{table_name} p
        LEFT JOIN public.nodes n ON p.nodeid = n.nodeid
        LEFT JOIN public.metrics_definition m ON LOWER(m.metric_id) = LOWER('{METRIC_ID}')
        WHERE n.hostname IN ({_quote_sql_strings(hostnames)})
          AND p.timestamp BETWEEN '{_format_db_time(start_time)}' AND '{_format_db_time(end_time)}'
          AND p.value IS NOT NULL
{sample_filter.rstrip()}
    ),
    stats AS (
        SELECT
            hostname,
            COUNT(*) AS sample_count,
            MIN(value_watts) AS min_watts,
            percentile_cont(0.10) WITHIN GROUP (ORDER BY value_watts) AS p10_watts,
            percentile_cont(0.25) WITHIN GROUP (ORDER BY value_watts) AS p25_watts,
            percentile_cont(0.50) WITHIN GROUP (ORDER BY value_watts) AS median_watts,
            percentile_cont(0.75) WITHIN GROUP (ORDER BY value_watts) AS p75_watts,
            percentile_cont(0.90) WITHIN GROUP (ORDER BY value_watts) AS p90_watts,
            percentile_cont(0.95) WITHIN GROUP (ORDER BY value_watts) AS p95_watts,
            percentile_cont(0.99) WITHIN GROUP (ORDER BY value_watts) AS p99_watts,
            MAX(value_watts) AS max_watts,
            AVG(value_watts) AS mean_watts,
            STDDEV(value_watts) AS std_watts
        FROM samples
        GROUP BY hostname
    ),
    percentiles AS (
        SELECT
            s.hostname,
            gs.percentile,
            percentile_cont(gs.percentile / 100.0) WITHIN GROUP (ORDER BY s.value_watts) AS value_watts
        FROM samples s
        CROSS JOIN generate_series(0, 100) AS gs(percentile)
        GROUP BY s.hostname, gs.percentile
    )
    SELECT
        p.hostname,
        st.sample_count,
        st.min_watts,
        st.p10_watts,
        st.p25_watts,
        st.median_watts,
        st.p75_watts,
        st.p90_watts,
        st.p95_watts,
        st.p99_watts,
        st.max_watts,
        st.mean_watts,
        st.std_watts,
        p.percentile,
        p.value_watts,
        'W' AS units
    FROM percentiles p
    JOIN stats st ON p.hostname = st.hostname
    ORDER BY p.hostname ASC, p.percentile ASC;
    """


def query_node_total_fan_power(
    rack: int,
    hostname: str,
    database: str,
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame:
    query = build_total_fan_power_query(hostname, start_time, end_time)
    with get_pooled_connection(database, "idrac") as client:
        df = pd.read_sql_query(query, client.db_connection)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["rack"] = rack
    df["metric"] = METRIC_ID
    df["query_hostname"] = hostname
    return _with_value_watts(df)


def query_group_total_fan_power(group: QueryGroup, start_time: datetime, end_time: datetime) -> pd.DataFrame:
    query = build_total_fan_power_batch_query(group.hostnames, start_time, end_time)
    with get_pooled_connection(group.database, group.schema) as client:
        df = pd.read_sql_query(query, client.db_connection)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["rack"] = group.rack
    df["metric"] = METRIC_ID
    df["query_hostname"] = df["hostname"]
    return _with_value_watts(df)


def query_group_total_fan_power_percentiles(
    group: QueryGroup,
    start_time: datetime,
    end_time: datetime,
    sample_seconds: int = 300,
) -> pd.DataFrame:
    query = build_total_fan_power_percentile_query(group.hostnames, start_time, end_time, sample_seconds=sample_seconds)
    with get_pooled_connection(group.database, group.schema) as client:
        df = pd.read_sql_query(query, client.db_connection)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["rack"] = group.rack
    df["metric"] = METRIC_ID
    df["query_hostname"] = df["hostname"]
    return df


def _with_value_watts(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["value_watts"] = pd.to_numeric(result["value"], errors="coerce")
    if "units" not in result:
        result["units"] = "W"
        return result

    units = result["units"].fillna("W").astype(str).str.lower()
    result.loc[units == "mw", "value_watts"] = result.loc[units == "mw", "value_watts"] / 1000.0
    result.loc[units == "kw", "value_watts"] = result.loc[units == "kw", "value_watts"] * 1000.0
    return result


def query_targets(
    targets: list[QueryTarget],
    start_time: datetime,
    end_time: datetime,
    workers: int = 7,
    mode: str = "percentiles",
    sample_seconds: int = 300,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    groups = build_query_groups(targets)
    failures: list[dict[str, object]] = []
    worker_count = max(1, min(workers, len(groups))) if groups else 1
    print(f"Running {len(groups)} rack queries with {worker_count} workers ...", flush=True)

    def _run_group(group: QueryGroup) -> tuple[QueryGroup, pd.DataFrame]:
        print(
            f"Querying rack {group.rack} {len(group.hostnames)} nodes {METRIC_ID} from {group.database}.{group.schema} ...",
            flush=True,
        )
        if mode == "raw":
            return group, query_group_total_fan_power(group, start_time=start_time, end_time=end_time)
        return group, query_group_total_fan_power_percentiles(
            group,
            start_time=start_time,
            end_time=end_time,
            sample_seconds=sample_seconds,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_group = {executor.submit(_run_group, group): group for group in groups}
            for future in as_completed(future_to_group):
                group = future_to_group[future]
                try:
                    _, df = future.result()
                except Exception as exc:
                    print(f"  skipped rack {group.rack} {group.database}.{group.schema}: {exc}", flush=True)
                    failures.append(
                        {
                            "rack": group.rack,
                            "database": group.database,
                            "schema": group.schema,
                            "error": str(exc).strip(),
                        }
                    )
                    continue
                if not df.empty:
                    frames.append(df)
    if not frames:
        return pd.DataFrame(), failures
    return pd.concat(frames, ignore_index=True), failures


def query_positional_groups(
    groups: list[PositionalQueryGroup],
    start_time: datetime,
    end_time: datetime,
    granularity: str,
    workers: int = 7,
    sample_seconds: int = POSITIONAL_DEFAULT_SAMPLE_SECONDS,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, object]] = []
    worker_count = max(1, min(workers, len(groups))) if groups else 1
    print(f"Running {len(groups)} positional {granularity} queries with {worker_count} workers ...", flush=True)

    def _run_group(group: PositionalQueryGroup) -> tuple[PositionalQueryGroup, pd.DataFrame]:
        labels = ",".join(group.grouped_hostnames)
        print(
            f"Querying rack {group.rack} {granularity} positional groups [{labels}] from {group.database}.{group.schema} ...",
            flush=True,
        )
        query = build_positional_query(
            rack=group.rack,
            grouped_hostnames=group.grouped_hostnames,
            granularity=granularity,
            start_time=start_time,
            end_time=end_time,
            sample_seconds=sample_seconds,
        )
        with get_pooled_connection(group.database, group.schema) as client:
            df = pd.read_sql_query(query, client.db_connection)
        return group, df

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_group = {executor.submit(_run_group, group): group for group in groups}
            for future in as_completed(future_to_group):
                group = future_to_group[future]
                try:
                    _, df = future.result()
                except Exception as exc:
                    print(f"  skipped rack {group.rack} {granularity}: {exc}", flush=True)
                    failures.append(
                        {
                            "rack": group.rack,
                            "database": group.database,
                            "schema": group.schema,
                            "granularity": granularity,
                            "error": str(exc).strip(),
                        }
                    )
                    continue
                if not df.empty:
                    frames.append(df)
    if not frames:
        return pd.DataFrame(), failures
    combined = pd.concat(frames, ignore_index=True)
    combined["granularity"] = granularity
    return combined, failures


def write_query_plan(
    targets: list[QueryTarget],
    start_time: datetime,
    end_time: datetime,
    output_path: Path,
    mode: str,
    sample_seconds: int = 300,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"-- Metric: {METRIC_ID}",
        f"-- Window: {_format_db_time(start_time)} -> {_format_db_time(end_time)}",
        "",
    ]
    for group in build_query_groups(targets):
        lines.append(
            f"-- rack={group.rack} nodes={len(group.hostnames)} database={group.database}.{group.schema}"
        )
        if mode == "raw":
            lines.append(build_total_fan_power_batch_query(group.hostnames, start_time, end_time).strip())
        else:
            lines.append(
                build_total_fan_power_percentile_query(
                    group.hostnames,
                    start_time,
                    end_time,
                    sample_seconds=sample_seconds,
                ).strip()
            )
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_positional_query_plan(
    groups: list[PositionalQueryGroup],
    start_time: datetime,
    end_time: datetime,
    output_path: Path,
    sample_seconds: int = POSITIONAL_DEFAULT_SAMPLE_SECONDS,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"-- Metrics: {METRIC_ID}, {SYSTEM_INPUT_METRIC_ID}",
        f"-- Window: {_format_db_time(start_time)} -> {_format_db_time(end_time)}",
        f"-- Sample seconds: {sample_seconds}",
        "",
    ]
    for granularity in POSITIONAL_GRANULARITIES:
        for group in groups:
            labels = ", ".join(f"{name}:{len(hostnames)}" for name, hostnames in group.grouped_hostnames.items())
            lines.append(
                f"-- granularity={granularity} rack={group.rack} groups={labels} database={group.database}.{group.schema}"
            )
            lines.append(
                build_positional_query(
                    rack=group.rack,
                    grouped_hostnames=group.grouped_hostnames,
                    granularity=granularity,
                    start_time=start_time,
                    end_time=end_time,
                    sample_seconds=sample_seconds,
                ).strip()
            )
            lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def summarize_by_node(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "rack",
                "hostname",
                "count",
                "first_timestamp",
                "last_timestamp",
                "min_watts",
                "p10_watts",
                "p25_watts",
                "median_watts",
                "p75_watts",
                "p90_watts",
                "p95_watts",
                "p99_watts",
                "max_watts",
                "mean_watts",
                "std_watts",
            ]
        )

    if "sample_count" in df.columns and "percentile" in df.columns:
        columns = [
            "rack",
            "hostname",
            "sample_count",
            "min_watts",
            "p10_watts",
            "p25_watts",
            "median_watts",
            "p75_watts",
            "p90_watts",
            "p95_watts",
            "p99_watts",
            "max_watts",
            "mean_watts",
            "std_watts",
        ]
        return df[columns].drop_duplicates().sort_values(["rack", "hostname"]).reset_index(drop=True)

    rows = []
    for (rack, hostname), group in df.groupby(["rack", "hostname"], sort=True):
        values = pd.to_numeric(group["value_watts"], errors="coerce").dropna()
        rows.append(
            {
                "rack": rack,
                "hostname": hostname,
                "count": int(values.count()),
                "first_timestamp": str(group["timestamp"].min()) if "timestamp" in group else "",
                "last_timestamp": str(group["timestamp"].max()) if "timestamp" in group else "",
                "min_watts": values.min(),
                "p10_watts": values.quantile(0.10),
                "p25_watts": values.quantile(0.25),
                "median_watts": values.quantile(0.50),
                "p75_watts": values.quantile(0.75),
                "p90_watts": values.quantile(0.90),
                "p95_watts": values.quantile(0.95),
                "p99_watts": values.quantile(0.99),
                "max_watts": values.max(),
                "mean_watts": values.mean(),
                "std_watts": values.std(),
            }
        )
    return pd.DataFrame(rows)


def _ecdf(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    sorted_values = values.dropna().sort_values().reset_index(drop=True)
    if sorted_values.empty:
        return sorted_values, sorted_values
    probabilities = pd.Series((sorted_values.index + 1) / len(sorted_values))
    return sorted_values, probabilities


def plot_rack_ecdfs(
    df: pd.DataFrame,
    output_dir: Path,
    racks: list[int] | tuple[int, ...] = DEFAULT_RACKS,
    image_format: str = "png",
) -> list[Path]:
    cache_root = Path(tempfile.gettempdir()) / "repacss_power_profiling_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    grouped = dict(tuple(df.groupby("rack", sort=True))) if not df.empty else {}

    for rack in racks:
        rack_df = grouped.get(rack, pd.DataFrame())
        fig, ax = plt.subplots(figsize=(12, 7))
        plotted = 0
        if not rack_df.empty:
            for hostname, node_df in rack_df.groupby("hostname", sort=True):
                if "percentile" in node_df.columns:
                    plot_df = node_df.sort_values("percentile")
                    values = pd.to_numeric(plot_df["value_watts"], errors="coerce")
                    probabilities = pd.to_numeric(plot_df["percentile"], errors="coerce") / 100.0
                else:
                    values, probabilities = _ecdf(pd.to_numeric(node_df["value_watts"], errors="coerce"))
                if values.dropna().empty:
                    continue
                ax.plot(values, probabilities, linewidth=1.4, alpha=0.85, label=hostname)
                plotted += 1

        ax.set_title(f"Rack {rack} TotalFanPower distribution")
        ax.set_xlabel("TotalFanPower (W)")
        ax.set_ylabel("Empirical cumulative probability")
        ax.grid(True, linewidth=0.5, alpha=0.35)
        if plotted:
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize="small", frameon=False)
        else:
            ax.text(0.5, 0.5, "No TotalFanPower rows returned", transform=ax.transAxes, ha="center", va="center")
        fig.tight_layout()
        path = output_dir / f"rack_{rack}_total_fan_power_ecdf.{image_format}"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _prepare_matplotlib():
    cache_root = Path(tempfile.gettempdir()) / "repacss_power_profiling_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    import matplotlib.pyplot as plt

    return plt


def plot_positional_lines(
    df: pd.DataFrame,
    output_dir: Path,
    metric_column: str,
    ylabel: str,
    filename_suffix: str,
    image_format: str = "png",
) -> list[Path]:
    plt = _prepare_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if df.empty:
        return paths
    plot_df = df.copy()
    plot_df["time_bin"] = pd.to_datetime(plot_df["time_bin"], errors="coerce")
    for rack, rack_df in plot_df.groupby("rack", sort=True):
        fig, ax = plt.subplots(figsize=(12, 5.5))
        for position_group, group_df in rack_df.groupby("position_group", sort=True):
            ordered = group_df.sort_values("time_bin")
            ax.plot(ordered["time_bin"], ordered[metric_column], marker="o", linewidth=1.4, markersize=2.5, label=position_group)
        ax.set_title(f"Rack {rack} {ylabel}")
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.5, alpha=0.35)
        ax.legend(frameon=False)
        fig.autofmt_xdate()
        fig.tight_layout()
        path = output_dir / f"rack_{rack}_{filename_suffix}.{image_format}"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def build_positional_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (rack, time_bin, granularity), group in df.groupby(["rack", "time_bin", "granularity"], sort=True):
        values = {row["position_group"]: row for _, row in group.iterrows()}
        if "lower" in values and "upper" in values:
            lower = values["lower"]
            upper = values["upper"]
            lower_median = lower["fan_median_watts"]
            upper_median = upper["fan_median_watts"]
            ratio_delta = upper["fan_ratio_median"] - lower["fan_ratio_median"]
            rows.append(
                {
                    "rack": rack,
                    "time_bin": time_bin,
                    "granularity": granularity,
                    "lower_fan_median_watts": lower_median,
                    "upper_fan_median_watts": upper_median,
                    "fan_median_delta_upper_minus_lower": upper_median - lower_median,
                    "fan_median_percent_delta": ((upper_median - lower_median) / lower_median * 100.0) if lower_median else None,
                    "lower_fan_ratio_median": lower["fan_ratio_median"],
                    "upper_fan_ratio_median": upper["fan_ratio_median"],
                    "fan_ratio_delta_upper_minus_lower": ratio_delta,
                    "lower_sample_count": lower["sample_count"],
                    "upper_sample_count": upper["sample_count"],
                }
            )
        elif "gpu_all" in values:
            gpu = values["gpu_all"]
            rows.append(
                {
                    "rack": rack,
                    "time_bin": time_bin,
                    "granularity": granularity,
                    "lower_fan_median_watts": None,
                    "upper_fan_median_watts": None,
                    "fan_median_delta_upper_minus_lower": None,
                    "fan_median_percent_delta": None,
                    "lower_fan_ratio_median": None,
                    "upper_fan_ratio_median": None,
                    "fan_ratio_delta_upper_minus_lower": None,
                    "gpu_all_fan_median_watts": gpu["fan_median_watts"],
                    "gpu_all_fan_p95_watts": gpu["fan_p95_watts"],
                    "gpu_all_fan_ratio_median": gpu["fan_ratio_median"],
                    "gpu_all_sample_count": gpu["sample_count"],
                }
            )
    return pd.DataFrame(rows)


def build_positional_overview(comparison_df: pd.DataFrame) -> pd.DataFrame:
    if comparison_df.empty:
        return pd.DataFrame()
    rows = []
    for (rack, granularity), group in comparison_df.groupby(["rack", "granularity"], sort=True):
        def _numeric_column(column: str) -> pd.Series:
            if column not in group:
                return pd.Series(dtype="float64")
            return pd.to_numeric(group[column], errors="coerce").dropna()

        delta = _numeric_column("fan_median_delta_upper_minus_lower")
        ratio_delta = _numeric_column("fan_ratio_delta_upper_minus_lower")
        gpu_median = _numeric_column("gpu_all_fan_median_watts")
        if not delta.empty:
            rows.append(
                {
                    "rack": rack,
                    "granularity": granularity,
                    "mean_fan_median_delta_upper_minus_lower": delta.mean(),
                    "median_fan_median_delta_upper_minus_lower": delta.median(),
                    "upper_gt_lower_percent": (delta > 0).mean() * 100.0,
                    "mean_fan_ratio_delta_upper_minus_lower": ratio_delta.mean() if not ratio_delta.empty else None,
                    "time_bins": len(delta),
                }
            )
        elif not gpu_median.empty:
            rows.append(
                {
                    "rack": rack,
                    "granularity": granularity,
                    "gpu_all_mean_fan_median_watts": gpu_median.mean(),
                    "gpu_all_median_fan_median_watts": gpu_median.median(),
                    "time_bins": len(gpu_median),
                }
            )
    return pd.DataFrame(rows)


def write_positional_report(
    output_dir: Path,
    start_time: datetime,
    end_time: datetime,
    racks: list[int],
    worker_count: int,
    sample_seconds: int,
    hourly_rows: int,
    daily_rows: int,
    comparison_path: Path,
    overview_path: Path,
    overview_df: pd.DataFrame,
    hourly_path: Path,
    daily_path: Path,
    figure_paths: list[Path],
    failures: list[dict[str, object]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"
    lines = [
        "# Rack Position TotalFanPower Report",
        "",
        "## Query",
        "",
        f"- Metrics: `{METRIC_ID}`, `{SYSTEM_INPUT_METRIC_ID}`",
        f"- Window: `{_format_db_time(start_time)}` to `{_format_db_time(end_time)}`",
        f"- Racks: `{', '.join(map(str, racks))}`",
        "- Normal rack groups: `lower` slots 1-7, `upper` slots 13-20",
        "- Rack 93 group: `gpu_all` slots 1-8",
        f"- Worker count: {worker_count}",
        f"- Sample seconds: {sample_seconds}",
        f"- Hourly rows returned: {hourly_rows}",
        f"- Daily rows returned: {daily_rows}",
        "",
        "## Artifacts",
        "",
        f"- Hourly CSV: `{hourly_path.name}`",
        f"- Daily CSV: `{daily_path.name}`",
        f"- Comparison CSV: `{comparison_path.name}`",
        f"- Overview CSV: `{overview_path.name}`",
        "- Query plan: `query_plan.sql`",
        "",
        "## Findings Summary",
        "",
    ]
    if overview_df.empty:
        lines.append("- No overview rows were generated.")
    else:
        for _, row in overview_df.iterrows():
            if pd.notna(row.get("mean_fan_median_delta_upper_minus_lower")):
                lines.append(
                    "- rack {rack} {granularity}: mean upper-minus-lower median fan delta "
                    "{delta:.2f} W; upper > lower in {pct:.1f}% of time bins; mean fan-ratio delta {ratio:.5f}".format(
                        rack=int(row["rack"]),
                        granularity=row["granularity"],
                        delta=row["mean_fan_median_delta_upper_minus_lower"],
                        pct=row["upper_gt_lower_percent"],
                        ratio=row["mean_fan_ratio_delta_upper_minus_lower"],
                    )
                )
            else:
                lines.append(
                    "- rack {rack} {granularity}: gpu_all mean median fan power {value:.2f} W across {bins} time bins".format(
                        rack=int(row["rack"]),
                        granularity=row["granularity"],
                        value=row["gpu_all_mean_fan_median_watts"],
                        bins=int(row["time_bins"]),
                    )
                )
    lines.extend([
        "",
        "## Figures",
        "",
    ])
    for figure_path in figure_paths:
        lines.append(f"- `{figure_path.name}`")
    lines.extend(["", "## Failures", ""])
    if failures:
        for failure in failures:
            lines.append(
                f"- rack {failure['rack']} {failure.get('granularity', '')} {failure['database']}.{failure['schema']}: {failure['error']}"
            )
    else:
        lines.append("- None")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy()
    for column in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[column]):
            safe[column] = safe[column].astype(str)
    return safe


def export_results(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, timestamp: str, write_raw: bool, mode: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_csv": output_dir / f"rack_total_fan_power_summary_{timestamp}.csv",
    }
    summary.to_csv(paths["summary_csv"], index=False)
    if write_raw:
        data_kind = "raw" if mode == "raw" else "percentiles"
        paths["data_csv_gz"] = output_dir / f"rack_total_fan_power_{data_kind}_{timestamp}.csv.gz"
        _excel_safe(df).to_csv(paths["data_csv_gz"], index=False, compression="gzip")
    return paths


def write_run_report(
    output_dir: Path,
    start_time: datetime,
    end_time: datetime,
    racks: list[int],
    target_count: int,
    group_count: int,
    worker_count: int,
    mode: str,
    sample_seconds: int,
    row_count: int,
    summary_path: Path,
    raw_path: Path | None,
    figure_paths: list[Path],
    failures: list[dict[str, object]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.md"
    lines = [
        "# Monthly Rack TotalFanPower Report",
        "",
        "## Query",
        "",
        f"- Metric: `{METRIC_ID}`",
        f"- Window: `{_format_db_time(start_time)}` to `{_format_db_time(end_time)}`",
        f"- Racks: `{', '.join(map(str, racks))}`",
        f"- Compute nodes targeted: {target_count}",
        f"- Rack query groups: {group_count}",
        f"- Worker count: {worker_count}",
        f"- Query mode: `{mode}`",
        f"- Sample seconds: {sample_seconds}",
        f"- Rows returned: {row_count}",
        "",
        "## Artifacts",
        "",
        f"- Summary CSV: `{summary_path.name}`",
    ]
    if raw_path is not None:
        lines.append(f"- Raw CSV gzip: `{raw_path.name}`")
    lines.extend(["- Query plan: `query_plan.sql`", ""])
    lines.append("## Figures")
    lines.append("")
    for figure_path in figure_paths:
        lines.append(f"- `{figure_path.name}`")
    lines.extend(["", "## Failures", ""])
    if failures:
        for failure in failures:
            lines.append(
                f"- rack {failure['rack']} {failure['database']}.{failure['schema']}: {failure['error']}"
            )
    else:
        lines.append("- None")
    lines.append("")
    if row_count == 0:
        lines.extend(
            [
                "## Status",
                "",
                "No telemetry rows were returned. Check SSH/DNS connectivity, DB access, the query window, and metric availability.",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query one month of compute-node TotalFanPower and plot one ECDF distribution figure per rack."
    )
    parser.add_argument("--analysis", choices=["distribution", "positional"], default="distribution")
    parser.add_argument("--days", type=int, default=30, help="Recent window length when --start is not given.")
    parser.add_argument("--start", help="Start time in 'YYYY-MM-DD HH:MM:SS'.")
    parser.add_argument("--end", help="End time in 'YYYY-MM-DD HH:MM:SS'. Defaults to local now.")
    parser.add_argument("--racks", type=int, nargs="+", default=list(DEFAULT_RACKS), help="Rack numbers to query.")
    parser.add_argument("--output-dir", type=Path, default=Path("output") / "monthly_rack_fan_power")
    parser.add_argument("--plot-format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--workers", type=int, default=7, help="Parallel rack query workers.")
    parser.add_argument("--mode", choices=["percentiles", "raw"], default="percentiles", help="Use DB-side percentiles or pull raw samples.")
    parser.add_argument("--sample-seconds", type=int, default=300, help="Sampling interval for percentile mode; use 0 for no sampling.")
    parser.add_argument("--plan-only", action="store_true", help="Write SQL query plan without querying the database.")
    parser.add_argument("--no-raw", action="store_true", help="Skip writing the compressed raw CSV.")
    return parser.parse_args(argv)


def run_positional_analysis(args: argparse.Namespace) -> int:
    default_start, default_end = _default_positional_window()
    start_time = _parse_time(args.start) or default_start
    end_time = _parse_time(args.end) or default_end
    timestamp = end_time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"positional_{start_time:%Y%m%d}_{end_time:%Y%m%d}_{timestamp}"

    groups = build_positional_query_groups(args.racks)
    worker_count = max(1, min(args.workers, len(groups))) if groups else 1
    query_plan_path = run_dir / "query_plan.sql"
    write_positional_query_plan(groups, start_time, end_time, query_plan_path, sample_seconds=args.sample_seconds)
    print(f"Wrote SQL query plan: {query_plan_path}")
    if args.plan_only:
        print(
            f"Planned {len(groups) * len(POSITIONAL_GRANULARITIES)} positional queries "
            f"for racks: {', '.join(map(str, args.racks))}"
        )
        return 0

    hourly_df, hourly_failures = query_positional_groups(
        groups,
        start_time=start_time,
        end_time=end_time,
        granularity="hour",
        workers=args.workers,
        sample_seconds=args.sample_seconds,
    )
    daily_df, daily_failures = query_positional_groups(
        groups,
        start_time=start_time,
        end_time=end_time,
        granularity="day",
        workers=args.workers,
        sample_seconds=args.sample_seconds,
    )
    failures = hourly_failures + daily_failures

    run_dir.mkdir(parents=True, exist_ok=True)
    hourly_path = run_dir / f"positional_hourly_{timestamp}.csv"
    daily_path = run_dir / f"positional_daily_{timestamp}.csv"
    comparison_path = run_dir / f"positional_comparison_{timestamp}.csv"
    overview_path = run_dir / f"positional_overview_{timestamp}.csv"
    hourly_df.to_csv(hourly_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    comparison_df = build_positional_comparison(pd.concat([hourly_df, daily_df], ignore_index=True) if not hourly_df.empty or not daily_df.empty else pd.DataFrame())
    comparison_df.to_csv(comparison_path, index=False)
    overview_df = build_positional_overview(comparison_df)
    overview_df.to_csv(overview_path, index=False)

    figure_paths: list[Path] = []
    figure_paths.extend(
        plot_positional_lines(
            hourly_df,
            run_dir,
            metric_column="fan_median_watts",
            ylabel="Hourly median TotalFanPower (W)",
            filename_suffix="hourly_fan_median",
            image_format=args.plot_format,
        )
    )
    figure_paths.extend(
        plot_positional_lines(
            hourly_df,
            run_dir,
            metric_column="fan_p95_watts",
            ylabel="Hourly p95 TotalFanPower (W)",
            filename_suffix="hourly_fan_p95",
            image_format=args.plot_format,
        )
    )
    figure_paths.extend(
        plot_positional_lines(
            daily_df,
            run_dir,
            metric_column="fan_median_watts",
            ylabel="Daily median TotalFanPower (W)",
            filename_suffix="daily_fan_median",
            image_format=args.plot_format,
        )
    )
    figure_paths.extend(
        plot_positional_lines(
            daily_df,
            run_dir,
            metric_column="fan_ratio_median",
            ylabel="Daily median TotalFanPower / SystemInputPower",
            filename_suffix="daily_fan_ratio_median",
            image_format=args.plot_format,
        )
    )

    report_path = write_positional_report(
        output_dir=run_dir,
        start_time=start_time,
        end_time=end_time,
        racks=args.racks,
        worker_count=worker_count,
        sample_seconds=args.sample_seconds,
        hourly_rows=len(hourly_df),
        daily_rows=len(daily_df),
        comparison_path=comparison_path,
        overview_path=overview_path,
        overview_df=overview_df,
        hourly_path=hourly_path,
        daily_path=daily_path,
        figure_paths=figure_paths,
        failures=failures,
    )
    print(f"Wrote hourly CSV: {hourly_path}")
    print(f"Wrote daily CSV: {daily_path}")
    print(f"Wrote comparison CSV: {comparison_path}")
    print(f"Wrote overview CSV: {overview_path}")
    for path in figure_paths:
        print(f"Wrote figure: {path}")
    print(f"Wrote report: {report_path}")
    if not comparison_df.empty:
        preview_columns = [
            column
            for column in [
                "rack",
                "time_bin",
                "granularity",
                "fan_median_delta_upper_minus_lower",
                "fan_median_percent_delta",
                "fan_ratio_delta_upper_minus_lower",
                "gpu_all_fan_median_watts",
            ]
            if column in comparison_df.columns
        ]
        print(comparison_df[preview_columns].head(30).to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.analysis == "positional":
        return run_positional_analysis(args)

    end_time = _parse_time(args.end) or datetime.now()
    start_time = _parse_time(args.start) or (end_time - timedelta(days=args.days))
    timestamp = end_time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{start_time:%Y%m%d}_{end_time:%Y%m%d}_{timestamp}"

    targets = build_query_targets(args.racks)
    groups = build_query_groups(targets)
    worker_count = max(1, min(args.workers, len(groups))) if groups else 1
    query_plan_path = run_dir / "query_plan.sql"
    write_query_plan(targets, start_time, end_time, query_plan_path, mode=args.mode, sample_seconds=args.sample_seconds)
    print(f"Wrote SQL query plan: {query_plan_path}")

    if args.plan_only:
        print(
            f"Planned {len(groups)} rack queries "
            f"for {len(targets)} nodes across racks: {', '.join(map(str, args.racks))}"
        )
        return 0

    df, failures = query_targets(
        targets,
        start_time,
        end_time,
        workers=args.workers,
        mode=args.mode,
        sample_seconds=args.sample_seconds,
    )
    summary = summarize_by_node(df)
    export_paths = export_results(df, summary, run_dir, timestamp, write_raw=not args.no_raw, mode=args.mode)
    figure_paths = plot_rack_ecdfs(df, run_dir, racks=args.racks, image_format=args.plot_format)
    report_path = write_run_report(
        output_dir=run_dir,
        start_time=start_time,
        end_time=end_time,
        racks=args.racks,
        target_count=len(targets),
        group_count=len(groups),
        worker_count=worker_count,
        mode=args.mode,
        sample_seconds=args.sample_seconds,
        row_count=len(df),
        summary_path=export_paths["summary_csv"],
        raw_path=export_paths.get("data_csv_gz"),
        figure_paths=figure_paths,
        failures=failures,
    )

    for label, path in export_paths.items():
        print(f"Wrote {label}: {path}")
    for path in figure_paths:
        print(f"Wrote figure: {path}")
    print(f"Wrote report: {report_path}")
    if df.empty:
        print("No rows returned. Check DB connectivity, time window, and whether TotalFanPower exists for these nodes.")
    else:
        count_column = "sample_count" if "sample_count" in summary.columns else "count"
        print(summary[["rack", "hostname", count_column, "median_watts", "p95_watts"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
