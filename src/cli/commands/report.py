#!/usr/bin/env python3
"""
Reporting CLI commands
"""

import click
import sys
import os
from datetime import datetime, timedelta, date

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from reporting.excel import ExcelReporter
from reporting.formats import ReportFormatter
from services.power_service import PowerAnalysisService


@click.command()
@click.option('--output', 
              help='Output file path (default: auto-generated)')
@click.option('--databases', 
              multiple=True,
              type=click.Choice(['h100', 'zen4', 'infra']),
              default=['h100', 'zen4', 'infra'],
              help='Databases to include in report')
@click.option('--sheets', 
              multiple=True,
              help='Specific sheets to include (optional)')
def excel(output, databases, sheets):
    """
    Generate comprehensive Excel power metrics report.
    
    Examples:
        # Generate report with all databases
        python -m src.cli report excel
        
        # Generate report for specific databases
        python -m src.cli report excel --databases h100 zen4
        
        # Generate report with custom output path
        python -m src.cli report excel --output custom_report.xlsx
    """
    
    if not output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = f"power_metrics_report_{timestamp}.xlsx"
    
    click.echo(f"📊 Generating Excel report: {output}")
    click.echo(f"🗄️  Databases: {', '.join(databases)}")
    if sheets:
        click.echo(f"📋 Sheets: {', '.join(sheets)}")
    click.echo()
    
    try:
        # Initialize reporter
        reporter = ExcelReporter()
        
        # Generate report
        reporter.generate_report(
            databases=list(databases),
            output_path=output,
            specific_sheets=list(sheets) if sheets else None
        )
        
        click.echo(f"✅ Excel report generated successfully: {output}")
        
    except Exception as e:
        click.echo(f"❌ Error generating Excel report: {e}")
        sys.exit(1)


@click.command()
@click.option('--rack', 
              type=int,
              help='Specific rack number (91-97)')
@click.option('--output', 
              help='Output directory (default: output/rack/)')
@click.option('--start-time', 
              help='Start time (YYYY-MM-DD HH:MM:SS)')
@click.option('--end-time', 
              help='End time (YYYY-MM-DD HH:MM:SS)')
def rack_report(rack, output, start_time, end_time):
    """
    Generate rack-level power analysis reports.
    
    Examples:
        # Analyze all racks
        python -m src.cli report rack
        
        # Analyze specific rack
        python -m src.cli report rack --rack 97
        
        # Analyze with custom time range
        python -m src.cli report rack --start-time "2025-01-01 00:00:00" --end-time "2025-01-02 00:00:00"
    """
    
    if not output:
        output = "output/rack/"
    
    click.echo(f"🏗️  Generating rack analysis report...")
    if rack:
        click.echo(f"🔧 Rack: {rack}")
    else:
        click.echo(f"🔧 Racks: All (91-97)")
    click.echo(f"📁 Output: {output}")
    if start_time and end_time:
        click.echo(f"📅 Time range: {start_time} to {end_time}")
    click.echo()
    
    try:
        # Initialize reporter
        reporter = ExcelReporter()
        
        # Generate rack analysis
        reporter.generate_rack_analysis(
            rack_number=rack,
            output_dir=output,
            start_time=start_time,
            end_time=end_time
        )
        
        click.echo(f"✅ Rack analysis report generated successfully")
        
    except Exception as e:
        click.echo(f"❌ Error generating rack analysis: {e}")
        sys.exit(1)


@click.command()
@click.option('--format', 
              type=click.Choice(['excel', 'csv', 'json', 'html']), 
              default='excel',
              help='Output format')
@click.option('--output', 
              help='Output file path')
@click.option('--template', 
              help='Custom report template')
def custom(format, output, template):
    """
    Generate custom reports with various formats.
    
    Examples:
        # Generate HTML report
        python -m src.cli report custom --format html --output report.html
        
        # Generate CSV report
        python -m src.cli report custom --format csv --output data.csv
    """
    
    if not output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = f"custom_report_{timestamp}.{format}"
    
    click.echo(f"📊 Generating {format.upper()} report: {output}")
    if template:
        click.echo(f"📋 Template: {template}")
    click.echo()
    
    try:
        # Initialize formatter
        formatter = ReportFormatter()
        
        # Generate custom report
        formatter.generate_custom_report(
            format=format,
            output_path=output,
            template=template
        )
        
        click.echo(f"✅ Custom report generated successfully: {output}")
        
    except Exception as e:
        click.echo(f"❌ Error generating custom report: {e}")
        sys.exit(1)


@click.command(name="pue-daily")
@click.option("--start-day", type=str, required=True, help="Start date (YYYY-MM-DD), required")
@click.option("--end-day", type=str, help="End date (YYYY-MM-DD), default: today (UTC)")
@click.option("--days", type=int, help="[Deprecated] Number of days back from now (use --start-day instead)")
@click.option("--start-date", type=str, help="[Deprecated] Start date (use --start-day instead)")
@click.option("--end-date", type=str, help="[Deprecated] End date (use --end-day instead)")
@click.option("--output", type=str, help="Output Excel path (default: auto-generated under output/)")
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging")
def pue_daily(start_day, end_day, days, start_date, end_date, output, debug):
    """
    Export daily PUE values to an Excel file.

    Each row has:
    - date
    - IRC total energy (kWh)
    - PDU total energy (kWh)
    - PUE

    Examples:
        # Query from 2025-07-17 to today
        python -m src.cli pue-daily --start-day 2025-07-17

        # Query from 2025-07-17 to 2025-08-17
        python -m src.cli pue-daily --start-day 2025-07-17 --end-day 2025-08-17
    """
    import logging

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" if debug else "%(asctime)s - %(levelname)s - %(message)s",
    )

    # Determine time range (UTC)
    now = datetime.utcnow()
    
    # Parse start day (required)
    try:
        start_d = datetime.strptime(start_day, "%Y-%m-%d").date()
    except ValueError:
        click.echo(f"❌ Invalid start-day format: {start_day}. Use YYYY-MM-DD format.")
        sys.exit(1)
    
    # Parse end day (optional, default to today)
    if end_day:
        try:
            end_d = datetime.strptime(end_day, "%Y-%m-%d").date()
        except ValueError:
            click.echo(f"❌ Invalid end-day format: {end_day}. Use YYYY-MM-DD format.")
            sys.exit(1)
    else:
        end_d = now.date()
    
    # Validate date range
    if start_d > end_d:
        click.echo(f"❌ Start date ({start_d}) must be before or equal to end date ({end_d})")
        sys.exit(1)

    # Start at 00:00 UTC; end at now if end_d is today, otherwise at next day's 00:00 UTC
    start_dt = datetime.combine(start_d, datetime.min.time())
    if end_d == now.date():
        end_dt = now
    else:
        end_dt = datetime.combine(end_d + timedelta(days=1), datetime.min.time())

    if not output:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        os.makedirs("output/pue", exist_ok=True)
        output = f"output/pue/pue_daily_{start_d.isoformat()}_to_{end_d.isoformat()}_{ts}.xlsx"

    click.echo("📊 Generating daily PUE report...")
    click.echo(f"📅 Date range (UTC): {start_d.isoformat()} to {end_d.isoformat()}")
    click.echo(f"🕒 Time window (UTC): {start_dt} to {end_dt}")
    click.echo(f"📁 Output: {output}")
    if debug:
        click.echo("🔍 Debug mode: ON")
    click.echo()

    try:
        service = PowerAnalysisService("infra")
        result = service.calculate_pue_daily(start_dt, end_dt)
        if isinstance(result, dict) and "error" in result:
            click.echo(f"❌ Failed: {result['error']}")
            sys.exit(1)

        df = result["data"]
        # Friendly column names for Excel
        df = df.rename(
            columns={
                "date": "date",
                "irc_energy_kwh": "irc_total_kwh",
                "pdu_energy_kwh": "pdu_total_kwh",
                "pue": "pue",
            }
        )

        import pandas as pd

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Daily_PUE", index=False)

        click.echo(f"✅ Done: {output}")
    except Exception as e:
        click.echo(f"❌ Error generating daily PUE report: {e}")
        sys.exit(1)
