# REPACSS Power Measurement - Usage Guide

This guide provides detailed usage examples, advanced features, and troubleshooting information for the REPACSS Power Measurement client.

For the Slurm integration and deployment changes needed for OOB, in-band, and hybrid profiling, see [Slurm Profiling Modes and Required Changes](SLURM_PROFILING_MODES.md).

## Table of Contents

- [Excel Reporting](#excel-reporting)
- [Advanced Usage Examples](#advanced-usage-examples)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Configuration Details](#configuration-details)

## Excel Reporting

The project includes a comprehensive Excel report generator that collects power metrics from all databases and outputs them to separate sheets for easy analysis.

### Generate Power Metrics Excel Report

```bash
python src/scripts/run_public_queries.py
```

This will:
- Connect to both H100/ZEN4 and INFRA databases
- Collect all power-related metrics and metadata
- Generate a timestamped Excel file: `power_metrics_report_YYYYMMDD_HHMMSS.xlsx`
- Create separate sheets for different metric categories

### Excel Report Contents

#### H100/ZEN4 Database Sheets
- **Power_Metrics**: All power-related metrics with definitions
- **Power_By_Units**: Power metrics grouped by units (mW, W, kW)
- **High_Accuracy_Power**: Power metrics with accuracy > 0.95
- **Power_By_Type**: Power metrics grouped by type
- **Power_FQDD_Info**: Power metrics with FQDD information
- **Comprehensive_Mapping**: Complete device mapping
- **Metrics_With_Sources**: Power metrics with source information
- **Metrics_With_FQDDS**: Power metrics with FQDD information

#### INFRA Database Sheets
- **All_IRC_Metrics**: All IRC infrastructure metrics
- **IRC_Power_Metrics**: IRC power-related metrics
- **IRC_Metrics_By_Units**: IRC metrics grouped by units
- **IRC_Metrics_By_Type**: IRC metrics grouped by data type
- **All_Nodes**: All nodes (PDUs and rack cooling systems)
- **PDU_Nodes**: PDU-specific nodes
- **Rack_Cooling_Nodes**: Rack cooling system nodes
- **Node_Count_By_Type**: Node count summary by type
- **Nodes_With_Metrics**: Nodes with their available metrics

## Advanced Usage Examples

### Multi-Database Operations

```python
from core.database import connect_to_all_databases, disconnect_all, get_all_clients
from core.config import config

# Connect to all databases
connect_to_all_databases()
clients = get_all_clients()

try:
    # Compare power across databases
    for db_name, client in clients.items():
        summary = client.get_computepower_summary()
        print(f"{db_name}: Avg {summary['avg_power']:.1f}W, Max {summary['max_power']:.1f}W")
        
finally:
    disconnect_all()
```

### Time-Range Analysis

```python
from datetime import datetime, timedelta
from core.client import REPACSSPowerClient, DatabaseConfig, SSHConfig
from core.config import config

client = REPACSSPowerClient(
    DatabaseConfig(**config.get_database_config("h100")),
    SSHConfig(**config.get_ssh_config())
)

try:
    client.connect()
    
    # Get power metrics for last 6 hours
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=6)
    
    metrics = client.get_computepower_metrics(
        start_time=start_time,
        end_time=end_time
    )
    
    # Analyze power trends
    for metric in metrics:
        print(f"Node {metric['nodeid']}: {metric['value']:.1f}W at {metric['timestamp']}")
        
finally:
    client.disconnect()
```

### Custom Query Execution

```python
# Execute custom SQL queries
query = """
SELECT 
    nodeid, 
    AVG(value) as avg_power,
    MAX(value) as max_power,
    MIN(value) as min_power
FROM idrac.computepower
WHERE timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY nodeid
ORDER BY avg_power DESC
"""

results = client.execute_query(query)
for row in results:
    print(f"Node {row['nodeid']}: Avg {row['avg_power']:.1f}W, Max {row['max_power']:.1f}W")
```

### Node-Specific Analysis

```python
# Get detailed metrics for a specific node
node_id = "4"
metrics = client.get_computepower_metrics(node_id=node_id, limit=100)

# Calculate statistics
values = [m['value'] for m in metrics]
avg_power = sum(values) / len(values)
max_power = max(values)
min_power = min(values)

print(f"Node {node_id} Power Analysis:")
print(f"  Average: {avg_power:.1f}W")
print(f"  Maximum: {max_power:.1f}W")
print(f"  Minimum: {min_power:.1f}W")
```

## API Reference

### Core Client Methods

#### `get_computepower_metrics(limit=None, node_id=None, start_time=None, end_time=None)`
Get compute power metrics from iDRAC schema.

**Parameters:**
- `limit` (int, optional): Maximum number of records to return
- `node_id` (str, optional): Specific node ID to filter by
- `start_time` (datetime, optional): Start time for time range
- `end_time` (datetime, optional): End time for time range

**Returns:** List of dictionaries with power metrics

#### `get_boardtemperature_metrics(limit=None, node_id=None, start_time=None, end_time=None)`
Get board temperature metrics.

**Parameters:** Same as `get_computepower_metrics()`

**Returns:** List of dictionaries with temperature metrics

#### `get_computepower_summary(node_id=None)`
Get power summary statistics.

**Parameters:**
- `node_id` (str, optional): Specific node ID, or None for all nodes

**Returns:** Dictionary with summary statistics

#### `get_idrac_cluster_summary()`
Get iDRAC metrics summary for entire cluster.

**Returns:** Dictionary with cluster-wide statistics

#### `get_available_idrac_metrics()`
Get list of available iDRAC metrics.

**Returns:** List of metric names

#### `execute_query(query)`
Execute custom SQL queries.

**Parameters:**
- `query` (str): SQL query string

**Returns:** Query results as list of dictionaries

## Example Output

```
Starting power metrics data collection...
==================================================
Collecting H100/ZEN4 power metrics...
Collecting INFRA power metrics...

Creating Excel report...
Writing H100/ZEN4 power metrics...
  - Power_Metrics: 15 rows
  - Power_By_Units: 3 rows
  - High_Accuracy_Power: 8 rows
  ...
Writing INFRA power metrics...
  - All_IRC_Metrics: 25 rows
  - IRC_Power_Metrics: 5 rows
  ...

Excel report created successfully: power_metrics_report_20241201_143022.xlsx

Report summary:
  - H100/ZEN4 sheets: 8
  - INFRA sheets: 9
  - Output file: power_metrics_report_20241201_143022.xlsx
```

## Troubleshooting

### Common Issues

1. **"No module named 'openpyxl'"**
   ```bash
   pip install pandas openpyxl
   ```

2. **"SSH authentication failed"**
   - Check SSH key permissions: `chmod 600 /path/to/your/private/key`
   - Verify SSH key path in `config.py`
   - Test SSH connection manually: `ssh -i /path/to/your/private/key username@hostname`

3. **"Database connection failed"**
   - Check database credentials in `config.py`
   - Verify SSH tunnel is established properly
   - Ensure TimescaleDB is running and accessible

4. **"Permission denied"**
   - Ensure write permissions in the output directory
   - Check if Excel file is open in another application

5. **"No data" in sheets**
   - Verify database contains the expected tables
   - Check if metrics_definition table exists
   - Ensure power-related metrics are defined

6. **"At least one sheet must be visible"**
   - This error occurs when no data is retrieved from databases
   - Check database connections and table existence
   - Verify SSH tunnel is working properly

### Database Schema Requirements

The script expects the following tables to exist:

#### H100/ZEN4 Database
- `public.metrics_definition` - Metric definitions
- `public.source` - Source information
- `public.fqdd` - FQDD information

#### INFRA Database  
- `public.metrics_definition` - IRC metric definitions
- `public.nodes` - Node information

### SSH Tunnel Connection

The script uses SSH tunnel connections to access the databases securely. Make sure:

1. **SSH key is properly configured**:
   ```bash
   chmod 600 /path/to/your/private/key
   ssh -i /path/to/your/private/key username@hostname
   ```

2. **SSH configuration is correct** in `config.py`:
   ```python
   ssh_hostname: str = "your.ssh.host.com"
   ssh_port: int = 22
   ssh_username: str = "your_ssh_username"
   ssh_private_key_path: str = "/path/to/your/private/key"
   ```

3. **Database configuration is correct**:
   ```python
   db_host: str = "localhost"  # From SSH perspective
   db_port: int = 5432
   db_user: str = "your_database_user"
   db_password: str = "your_database_password"
   ```

## Advanced Usage

### Custom Database Configurations

You can modify the script to connect to different databases by updating the configuration in `src/config.py` and the database names in the script.

### Adding New Query Types

To add new query types, edit the query files in the `src/queries/` directory:

- `h100_zen4_public_queries.py` - H100/ZEN4 public schema queries
- `infra_public_queries.py` - INFRA public schema queries

### Custom Excel Output

Modify the `create_excel_report()` function in `run_public_queries.py` to customize the Excel output format.

## File Structure

```
src/
├── run_public_queries.py      # Main report generator
├── config.py                  # SSH and database configuration
├── queries/                   # Query definitions
│   ├── h100_zen4_public_queries.py
│   └── infra_public_queries.py
└── common/
    ├── repacss_power_client.py   # SSH tunnel client library
    └── database_utils.py         # Database connection utilities
```

## Dependencies

- `psycopg2-binary>=2.9.0` - PostgreSQL adapter
- `pandas>=1.5.0` - Data manipulation
- `openpyxl>=3.0.0` - Excel file generation
- `paramiko==3.3.1` - SSH protocol implementation
- `sshtunnel>=0.4.0` - SSH tunnel management
