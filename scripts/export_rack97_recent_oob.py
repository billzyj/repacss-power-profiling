#!/usr/bin/env python3
"""Export recent Rack 97 OOB power rows into one workbook."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oob.backends.monster_db.backend import MonsterDBBackend


def _safe_sheet_name(hostname: str) -> str:
    return hostname.replace("/", "_").replace("\\", "_")[:31]


def _excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy()
    for column in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[column]):
            safe[column] = safe[column].astype(str)
    return safe


def main() -> int:
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=10)
    timestamp = end_time.strftime("%Y%m%d_%H%M%S")

    output_dir = Path("output") / "rack97_redfish_settings"
    output_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = output_dir / f"rack97_oob_last10min_{timestamp}.xlsx"
    combined_csv_path = output_dir / f"rack97_oob_last10min_{timestamp}_combined.csv"

    backend = MonsterDBBackend("zen4")
    nodes = [f"rpc-97-{idx}" for idx in range(1, 21)]
    frames: dict[str, pd.DataFrame] = {}
    summary_rows = []

    print(f"Window: {start_time:%Y-%m-%d %H:%M:%S} -> {end_time:%Y-%m-%d %H:%M:%S}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for hostname in nodes:
            print(f"Querying {hostname} ...", flush=True)
            df = backend.query_metrics(hostname, start_time=start_time, end_time=end_time, limit=0)
            if not df.empty:
                df = df.sort_values(["metric", "timestamp", "source", "fqdd"], na_position="last").reset_index(drop=True)
            frames[hostname] = df
            metrics = ",".join(sorted(df["metric"].dropna().unique())) if "metric" in df else ""
            summary_rows.append(
                {
                    "hostname": hostname,
                    "rows": len(df),
                    "first_timestamp": str(df["timestamp"].min()) if "timestamp" in df and not df.empty else "",
                    "last_timestamp": str(df["timestamp"].max()) if "timestamp" in df and not df.empty else "",
                    "metric_count": df["metric"].nunique() if "metric" in df and not df.empty else 0,
                    "metrics": metrics,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    combined_df = pd.concat(
        [df.assign(query_hostname=hostname) for hostname, df in frames.items()],
        ignore_index=True,
    )

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        query_info = pd.DataFrame(
            [
                {"key": "query_start_local", "value": start_time.strftime("%Y-%m-%d %H:%M:%S")},
                {"key": "query_end_local", "value": end_time.strftime("%Y-%m-%d %H:%M:%S")},
                {"key": "window_minutes", "value": 10},
                {"key": "node_count", "value": len(nodes)},
                {"key": "total_rows", "value": len(combined_df)},
            ]
        )
        query_info.to_excel(writer, sheet_name="summary", index=False, startrow=0)
        summary_df.to_excel(writer, sheet_name="summary", index=False, startrow=len(query_info) + 3)
        for hostname, df in frames.items():
            _excel_safe(df).to_excel(writer, sheet_name=_safe_sheet_name(hostname), index=False)

        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

    _excel_safe(combined_df).to_csv(combined_csv_path, index=False)
    print(f"Wrote workbook: {workbook_path}")
    print(f"Wrote combined CSV: {combined_csv_path}")
    print(summary_df[["hostname", "rows", "first_timestamp", "last_timestamp", "metric_count"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
