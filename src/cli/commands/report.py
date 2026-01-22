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


@click.command(name="rack-cop")
@click.option("--start-time", type=str, help="Start time (YYYY-MM-DD HH:MM:SS)")
@click.option("--end-time", type=str, help="End time (YYYY-MM-DD HH:MM:SS)")
@click.option("--hours", type=int, default=168, help="Number of hours to analyze (default: 168 = 1 week)")
@click.option("--output", type=str, help="Output Excel path (default: auto-generated under output/)")
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging")
def rack_cop(start_time, end_time, hours, output, debug):
    """
    Calculate rack-level power consumption and COP (Coefficient of Performance) for all racks.
    
    For each rack with an IRC node, calculates:
    - CompressorPower energy (kWh)
    - CondenserFanPower energy (kWh)
    - CoolDemand energy (kWh)
    - COP = CoolDemand / (CompressorPower + CondenserFanPower)
    
    Examples:
        # Calculate COP for last week (168 hours)
        python -m src.cli rack-cop
        
        # Calculate COP for custom time range
        python -m src.cli rack-cop --start-time "2025-01-01 00:00:00" --end-time "2025-01-08 00:00:00"
        
        # Calculate COP for last 24 hours
        python -m src.cli rack-cop --hours 24
    """
    import logging
    import pandas as pd

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" if debug else "%(asctime)s - %(levelname)s - %(message)s",
    )

    # Parse time range
    if start_time and end_time:
        start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        end_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
    elif start_time:
        start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        end_dt = start_dt + timedelta(hours=hours)
    else:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(hours=hours)

    if not output:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        os.makedirs("output/rack", exist_ok=True)
        output = f"output/rack/rack_cop_{start_dt.strftime('%Y%m%d_%H%M%S')}_to_{end_dt.strftime('%Y%m%d_%H%M%S')}.xlsx"

    click.echo("📊 Calculating rack-level COP...")
    click.echo(f"📅 Time range: {start_dt} to {end_dt}")
    click.echo(f"📁 Output: {output}")
    if debug:
        click.echo("🔍 Debug mode: ON")
    click.echo()

    try:
        service = PowerAnalysisService("infra")
        result = service.calculate_rack_cop(start_dt, end_dt)
        
        if isinstance(result, dict) and "error" in result:
            click.echo(f"❌ Failed: {result['error']}")
            sys.exit(1)

        # Display results
        click.echo("=" * 70)
        click.echo("Rack-Level COP Analysis Results")
        click.echo("=" * 70)
        click.echo()
        
        racks = result.get('racks', {})
        summary = result.get('summary', {})
        
        # Display summary
        click.echo(f"Total Racks: {summary.get('total_racks', 0)}")
        click.echo(f"Racks with Data: {summary.get('racks_with_data', 0)}")
        if summary.get('avg_cop') is not None:
            click.echo(f"Average COP: {summary.get('avg_cop', 0):.4f}")
        click.echo()
        
        # Display per-rack results
        click.echo("Rack Details:")
        click.echo("-" * 70)
        click.echo(f"{'Rack':<6} {'IRC Node':<12} {'Compressor (kWh)':<18} {'Fan (kWh)':<15} {'CoolDemand (kWh)':<18} {'COP':<10}")
        click.echo("-" * 70)
        
        for rack_num in sorted(racks.keys()):
            rack_data = racks[rack_num]
            cop_str = f"{rack_data['cop']:>8.4f}" if rack_data['cop'] is not None else f"{'N/A':>8}"
            click.echo(
                f"{rack_num:<6} "
                f"{rack_data['irc_node']:<12} "
                f"{rack_data['compressor_power_kwh']:>16.4f} "
                f"{rack_data['condenser_fan_power_kwh']:>13.4f} "
                f"{rack_data['cool_demand_kwh']:>16.4f} "
                f"{cop_str}"
            )
        
        click.echo()
        click.echo("=" * 70)
        
        # Export to Excel
        rows = []
        for rack_num in sorted(racks.keys()):
            rack_data = racks[rack_num]
            rows.append({
                'rack': rack_num,
                'irc_node': rack_data['irc_node'],
                'compressor_power_kwh': rack_data['compressor_power_kwh'],
                'condenser_fan_power_kwh': rack_data['condenser_fan_power_kwh'],
                'cool_demand_kwh': rack_data['cool_demand_kwh'],
                'cop': rack_data['cop'] if rack_data['cop'] is not None else None
            })
        
        df = pd.DataFrame(rows)
        
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Rack_COP", index=False)
            
            # Add summary sheet
            summary_df = pd.DataFrame([{
                'start_time': start_dt,
                'end_time': end_dt,
                'duration_hours': summary.get('duration_hours', 0),
                'total_racks': summary.get('total_racks', 0),
                'racks_with_data': summary.get('racks_with_data', 0),
                'avg_cop': summary.get('avg_cop', None)
            }])
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
        
        click.echo(f"✅ Excel report saved: {output}")
        
    except Exception as e:
        click.echo(f"❌ Error generating rack COP report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@click.command(name="rack-cop-daily")
@click.option("--start-day", type=str, required=True, help="Start date (YYYY-MM-DD), required")
@click.option("--end-day", type=str, help="End date (YYYY-MM-DD), default: today (UTC)")
@click.option("--max-gap-minutes", type=int, default=10, show_default=True, help="Max time gap (minutes) for interpolation. Gaps larger than this are treated as zero power.")
@click.option("--output", type=str, help="Output Excel path (default: auto-generated under output/)")
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging")
def rack_cop_daily(start_day, end_day, max_gap_minutes, output, debug):
    """
    Calculate daily rack-level power consumption and COP for a time range.
    
    For each rack (91-96) and each day, calculates:
    - CompressorPower energy (kWh)
    - CondenserFanPower energy (kWh)
    - CoolDemand energy (kWh)
    - COP = CoolDemand / (CompressorPower + CondenserFanPower)
    
    Output Excel columns: Date, Rack, IRC Node, Compressor (kWh), Fan (kWh), CoolDemand (kWh), COP
    
    Examples:
        # Calculate daily COP from start date to today
        python -m src.cli rack-cop-daily --start-day 2025-01-01
        
        # Calculate daily COP for a specific date range
        python -m src.cli rack-cop-daily --start-day 2025-01-01 --end-day 2025-06-30
    """
    import logging
    import pandas as pd

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s" if debug else "%(asctime)s - %(levelname)s - %(message)s",
    )

    # Parse date arguments
    try:
        start_dt = datetime.strptime(start_day, '%Y-%m-%d')
        start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    except ValueError:
        click.echo(f"❌ Invalid start-day format: {start_day}. Use YYYY-MM-DD")
        sys.exit(1)
    
    if end_day:
        try:
            end_dt = datetime.strptime(end_day, '%Y-%m-%d')
            end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
        except ValueError:
            click.echo(f"❌ Invalid end-day format: {end_day}. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        # Default to today (UTC)
        end_dt = datetime.utcnow()
        end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    if end_dt < start_dt:
        click.echo(f"❌ end-day must be after start-day")
        sys.exit(1)

    if not output:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        os.makedirs("output/rack", exist_ok=True)
        output = f"output/rack/rack_cop_daily_{start_dt.strftime('%Y%m%d')}_to_{end_dt.strftime('%Y%m%d')}_{ts}.xlsx"

    click.echo("📊 Calculating daily rack-level COP...")
    click.echo(f"📅 Date range: {start_dt.date()} to {end_dt.date()}")
    click.echo(f"📁 Output: {output}")
    if debug:
        click.echo("🔍 Debug mode: ON")
    click.echo()

    try:
        service = PowerAnalysisService("infra")
        result = service.calculate_rack_cop_daily(start_dt, end_dt, max_gap_minutes=max_gap_minutes)
        
        if isinstance(result, dict) and "error" in result:
            click.echo(f"❌ Failed: {result['error']}")
            sys.exit(1)

        df = result["data"]
        
        # Rename columns for Excel (friendly names)
        df = df.rename(
            columns={
                "date": "Date",
                "rack": "Rack",
                "irc_node": "IRC Node",
                "compressor_power_kwh": "Compressor (kWh)",
                "condenser_fan_power_kwh": "Fan (kWh)",
                "cool_demand_kwh": "CoolDemand (kWh)",
                "cop": "COP",
            }
        )

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Daily_Rack_COP", index=False)
            
            # Add summary sheet
            summary = result.get("summary", {})
            summary_df = pd.DataFrame([{
                'start_date': summary.get('start_date'),
                'end_date': summary.get('end_date'),
                'days': summary.get('days', 0),
                'racks': summary.get('racks', 0),
            }])
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        click.echo(f"✅ Done: {output}")
    except Exception as e:
        click.echo(f"❌ Error generating daily rack COP report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
