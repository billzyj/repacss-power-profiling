# REPACSS Power Measurement

Power measurement from out-of-band and in-band solutions for the REPACSS cluster.

## Overview

This project is being refactored into a major-version architecture with two first-class power domains:

- `Out-of-Band (OOB)`: database/API-backed post-job and offline power queries
- `In-Band (IB)`: Slurm-driven runtime sampling on compute nodes

The long-term source layout is organized around three primary implementation domains plus entrypoint and test layers:

- `shared/`: cross-domain contracts, config, Slurm helpers, common analysis/export foundations
- `oob/`: OOB query semantics, backend adapters, and headnode job-end workflows
- `inband/`: IB collectors, runtime lifecycle, staging, and aggregation
- `cli/`: user-facing program entrypoints, primarily for OOB query/export workflows
- `tests/`: unit, integration, and end-to-end validation

IB is not designed as a standalone manual collection workflow. Its primary execution model is Slurm-driven (`Prolog + Epilog + EpilogSlurmctld`), while the top-level CLI is primarily an OOB query/export surface plus limited debug/status helpers.

## Current Migration Status

The repository is currently in transition from the legacy `src/` monolith to the new domain-oriented layout above.

Current expectations:

- new refactor work should prefer `shared/`, `oob/`, `inband/`, and `cli/`
- `src/` is temporarily retained as a compatibility layer while the migration is still in progress
- `src/` will be reduced and eventually removed after the new architecture is fully validated

## Architecture Layout

Planned steady-state structure:

```text
repacss-power-profiling/
├── shared/      # cross-domain foundations
├── oob/         # out-of-band query and export logic
├── inband/      # in-band collectors and Slurm-bound runtime flow
├── cli/         # user-facing command entrypoints
├── tests/
│   ├── unit/
│   │   ├── shared/
│   │   ├── oob/
│   │   └── inband/
│   ├── integration/
│   │   ├── oob/
│   │   └── inband/
│   └── e2e/
└── src/         # temporary compatibility layer during migration
```

## Features

- **Secure SSH tunnel connection** to TimescaleDB using `sshtunnel`
- **Multi-database support** for different cluster databases
- Query power consumption, temperature, and utilization metrics from various schemas
- Support for time-range queries and aggregations
- Cluster-wide and node-specific summaries
- Custom SQL query execution
- Robust error handling and connection management
- Real-time power monitoring across multiple nodes
- Infrastructure monitoring (PDU, IRC, compressors, airflow)
- **Excel report generation** with comprehensive power metrics from all databases
- **Comprehensive rack power analysis** for all racks (91-97) with validation
- **Power consumption validation** comparing compute nodes vs PDU measurements
- **Smart power estimation** for unmeasured components (switches, AMD nodes, etc.)

## Quick Start (Refactored CLI)

### 1. Install

```bash
git clone <repository-url>
cd repacss-power-profiling
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

**Setup Flow:**
1. **Template → .env**: `setup.py` copies `env.template` to create your personal `.env` file
2. **Edit credentials**: You fill in your actual database and SSH information
3. **Secure storage**: `.env` file is automatically protected from git commits

```bash
# Step 1: Create your .env file from template
python3 setup.py

# Step 2: Edit the generated file with your credentials
# File location: src/database/config/.env
# Edit with your actual database host, username, password, SSH details, etc.
```

**What to Edit in `.env`:**
```bash
# Database Settings (replace with your actual values)
REPACSS_DB_HOST=your.database.host.com
REPACSS_DB_USER=your_database_username
REPACSS_DB_PASSWORD=your_database_password

# SSH Settings (replace with your actual values)
REPACSS_SSH_HOSTNAME=your.ssh.host.com
REPACSS_SSH_USERNAME=your_ssh_username
REPACSS_SSH_KEY_PATH=/path/to/your/private/key
```

**SSH Key Requirements:**
- **Supported formats**: RSA, Ed25519, ECDSA
- **DSS keys are deprecated** and not supported in newer paramiko versions
- **Convert DSS keys**: `ssh-keygen -p -m RFC4716 -f /path/to/your/key`
- **Generate new key**: `ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519`

### 3. Test

```bash
# Show effective configuration
python3 -m cli config show

# Validate configuration
python3 -m cli config test

# Backward-compatible legacy entrypoint still works during migration
python3 -m src.cli config show
```

### 3a. Fresh Clone Smoke Test

Use this path when validating the repository on a remote machine after a fresh `git clone`.
This smoke test is intended to avoid DB or SSH credentials and should remain the minimum validation path for refactor work.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 tests/run_tests.py --type unit
```

### 4. Use

```bash
# General OOB query
python3 -m cli oob query --hostname rpg-93-1 --start "2025-01-01 00:00:00" --end "2025-01-01 01:00:00"

# General OOB query with file output
python3 -m cli oob query --hostname rpc-91-2 --start "2025-01-01 00:00:00" --end "2025-01-01 01:00:00" --output output/query.csv --format csv

# Manual OOB job export
python3 -m cli oob job --job-id 96597 --user kalebuch --nodelist rpg-93-6 --start "2026-04-13 00:00:00" --end "2026-04-13 01:00:00" --outdir output/job-96597

# Generic export surface (currently OOB-backed in P0-P2)
python3 -m cli export --job 96597 --user kalebuch --nodelist rpg-93-6 --start "2026-04-13 00:00:00" --end "2026-04-13 01:00:00" --output output/export-96597
```

## Available Interfaces

- **CLI (primary user interface)**: `python3 -m cli ...`
- **Legacy compatibility CLI**: `python3 -m src.cli ...`
- **OOB programmatic APIs**: moving into `oob/` and `shared/`
- **IB runtime entrypoints**: Slurm `Prolog`, `Epilog`, and `EpilogSlurmctld`
- **Legacy scripts and legacy modules**: temporarily retained under `src/` for compatibility during migration

## Refactor Workflow

For large refactors, use a temporary working plan at `docs/_working/refactor-plan.md`.

Rules:

1. The temporary refactor plan may use mixed Chinese and English for collaboration between the user, Codex, and Claude Code.
2. Stable repository documentation must remain in English.
3. Any durable decisions from the temporary plan must be moved into `README.md` or `docs/*`.
4. Remove the temporary plan file before the refactor branch is finalized.

Refactor direction for this repository:

- primary source domains are `shared/`, `oob/`, and `inband/`
- `cli/` is the program entry layer rather than a fourth business domain
- `tests/` mirrors behavior and architecture validation, not just file layout
- `src/` remains temporary until the new architecture fully replaces the legacy implementation

### CLI Commands

```bash
python3 -m cli --help
python3 -m cli config show
python3 -m cli config test
python3 -m cli oob query --help
python3 -m cli oob job --help
python3 -m cli export --help
```

Current CLI scope in P0-P2:

- `config show`
- `config test`
- `oob query`
- `oob job`
- `export`

The top-level CLI is currently OOB-focused.
In-band runtime collection remains Slurm-driven and is not yet exposed as a primary manual CLI workflow.

### Analysis Types

- **PUE Analysis**: Cluster-wide Power Usage Effectiveness calculation
  - Formula: PUE = (Total IRC Power + Total PDU Power) / Total PDU Power
  - Calculates energy consumption for all IRC and PDU nodes
  - Provides PUE value with interpretation (Excellent/Good/Average/Poor)
- **Rack COP Analysis**: Rack-level Coefficient of Performance calculation
  - Formula: COP = CoolDemand / (CompressorPower + CondenserFanPower)
  - Calculates power consumption and cooling efficiency for each rack (91-96)
  - Provides CompressorPower, CondenserFanPower, CoolDemand energy (kWh) and COP value
  - **Daily Rack COP**: Daily breakdown of rack-level COP over a time period
    - Output columns: Date, Rack, IRC Node, Compressor (kWh), Fan (kWh), CoolDemand (kWh), COP
- **Rack Analysis**: Multi-rack power validation with estimation
- **Node Analysis**: Individual node power consumption tracking
- **Infrastructure Analysis**: PDU and cooling system power monitoring
- **GPU Analysis**: H100 GPU power consumption
- **CPU Analysis**: Zen4 CPU power consumption

## Basic Usage (Programmatic)

```python
from services.power_service import PowerAnalysisService
from datetime import datetime, timedelta

service = PowerAnalysisService('h100')
end = datetime.now(); start = end - timedelta(hours=6)
node_results = service.analyze_node_power('rpg-93-1', start, end)
print(node_results.get('summary', {}))
```

## Power Analysis with Energy Calculation

Energy and boundary handling are implemented in `analysis/energy.py` and used via the service layer.

### Cumulative Energy Logic

The system calculates energy consumption with proper boundary handling:

1. **Boundary Rows**: When query results don't include exact start/end times, boundary rows are added
2. **First Row**: Cumulative energy starts at 0.0
3. **Second Row**: Uses first power reading as average (no previous reading to average with)
4. **Remaining Rows**: Uses trapezoidal rule (average of current and previous power readings)

### Example Output

```python
# Query: 23:00:00 - 23:30:00
# Data: 23:00:01 - 23:29:59

timestamp           metric        value  power_w  energy_interval_kwh  cumulative_energy_kwh
23:00:00           systempower   150.0   150.0    0.0                  0.0                    # Start boundary
23:00:01           systempower   150.0   150.0    0.000042             0.000042               # First interval
23:00:31           systempower   155.0   155.0    0.00127              0.00131                # Trapezoidal
23:01:01           systempower   148.0   148.0    0.00126              0.00257                # Trapezoidal
23:01:31           systempower   152.0   152.0    0.00125              0.00382                # Trapezoidal
...
23:29:59           systempower   152.0   152.0    0.00125              0.04567                # Last data
23:30:00           systempower   152.0   152.0    0.0                  0.04567                # End boundary
```

### DataFrame Columns

The analysis returns a DataFrame with the following columns:

- `timestamp`: DateTime of the power reading
- `hostname`: Node hostname
- `value`: Original power reading (in original units)
- `units`: Unit from metrics_definition table (mW, W, kW)
- `metric`: Metric ID (table name)
- `power_w`: Power converted to Watts
- `time_diff_seconds`: Time difference from previous reading
- `avg_power_w`: Average power between current and previous readings
- `energy_interval_kwh`: Energy consumed in this time interval
- `cumulative_energy_kwh`: Total energy consumed up to this timestamp

### Node Type Support

- **Compute Nodes** (rpg-*, rpc-*): Automatically queries all power-related metrics from database
- **PDU Nodes** (pdu-*): Uses predefined PDU power metrics
- **IRC Nodes** (irc-*): Uses predefined IRC power metrics (CompressorPower, CondenserFanPower, etc.)

## Rack Power Analysis

The system provides comprehensive rack-level power analysis with validation across all racks (91-97):

### Rack Analysis Types

- **Rack 97 (Accurate)**: Direct comparison between compute nodes and PDU measurements
- **Racks 91, 94, 96 (Estimated Switches)**: Includes 2kW estimation for ethernet + infiniband switches
- **Rack 92 (Estimated AMD)**: Includes 1kW estimation for 2 AMD test nodes
- **Rack 93 (Estimated Mixed)**: Includes 3kW estimation for multiple components
- **Rack 95 (Estimated Switches)**: Includes 4kW estimation for switches + hammerspace nodes

### Analysis Features

- **Power Validation**: Compares compute node power consumption with PDU measurements
- **Energy Calculation**: Tracks total energy consumption over time periods
- **Smart Estimation**: Accounts for unmeasured components (switches, additional nodes)
- **Excel Reports**: Generates comprehensive reports with validation summaries
- **Multiple Sheets**: Separate sheets for compute nodes, PDU nodes, validation, and power comparison

### Output Structure

```
output/rack/
├── rack91_power_analysis_20250910_000000.xlsx
├── rack92_power_analysis_20250910_000000.xlsx
├── rack93_power_analysis_20250910_000000.xlsx
├── rack94_power_analysis_20250910_000000.xlsx
├── rack95_power_analysis_20250910_000000.xlsx
├── rack96_power_analysis_20250910_000000.xlsx
└── rack97_power_analysis_20250910_000000.xlsx
```

### Validation Logic

- **Accurate Analysis (Rack 97)**: Direct compute vs PDU comparison
- **Estimated Analysis**: Shows both raw and adjusted differences
- **Tolerance Levels**: 
  - ✅ GOOD: Within 10% difference
  - ⚠️ ACCEPTABLE: Within 20% difference  
  - ❌ NEEDS INVESTIGATION: >20% difference

## Documentation

- **[Usage Guide](docs/USAGE_GUIDE.md)** - Detailed usage examples, Excel reporting, and troubleshooting
- **[Project Structure](docs/PROJECT_STRUCTURE.md)** - Technical architecture and development details

## Dependencies

See `requirements_new.txt` (includes `click` and optional `.env` support).

Key:
- `psycopg2-binary`, `sshtunnel`, `paramiko`
- `pandas`, `openpyxl`
- `click`

## Architecture Overview

### Recent Improvements

**Enhanced Structure (Latest):**
- **Renamed `core/` → `database/`**: Clearer purpose for database-related modules
- **Added `constants/` module**: Centralized node lists and metric definitions
- **Enhanced `utils/` module**: Added query helpers and node detection utilities
- **Split `power_utils.py`**: Moved functions to focused modules (analysis, utils, constants)
- **Moved `examples/` and `tests/`**: Now parallel to `src/` for better organization
- **Improved imports**: Updated all modules to use new structure

### Project Structure
```
REPACSS-power-profiling/
├── src/                        # Application code
│   ├── cli/                    # CLI interface
│   ├── database/               # Database connections (renamed from core/)
│   │   └── config/             # Configuration (template + loader)
│   ├── analysis/               # Power analysis modules
│   ├── queries/                # Database queries
│   ├── services/               # Business logic
│   ├── reporting/              # Report generation
│   ├── constants/              # Node lists, metrics (new)
│   └── utils/                  # Utilities (enhanced)
├── examples/                    # Usage examples (moved from src/)
├── tests/                      # Test suite (moved from src/)
├── docs/                       # Documentation
├── output/                     # Generated reports
└── README.md
```

### Layer Structure
```
CLI (src/cli)
  └─ thin commands delegating to services
Services (src/services)
  └─ orchestration/business logic (e.g., PowerAnalysisService)
Analysis (src/analysis)
  ├─ power.py (processing)
  └─ energy.py (energy calc)
Queries (src/queries)
  └─ manager.py + schema-specific builders
Database (src/database)          # ← Renamed from core/
  ├─ client.py, database.py
  ├─ connection_pool.py
  └─ config_new.py (.env support)
Constants (src/constants)         # ← New module
  ├─ nodes.py (node definitions)
  └─ metrics.py (metric definitions)
Utils (src/utils)                # ← Enhanced
  ├─ conversions.py
  ├─ data_processing.py
  ├─ query_helpers.py
  └─ node_detection.py
Reporting (src/reporting)
  └─ excel.py, formats.py
```

### Data Flow
```
┌─────────────────────────────────────┐
│        Presentation Layer           │
├─────────────────┬───────────────────┤
│   CLI (Input)   │  Reporting (Output)│
│                 │                   │
│ • Parse args    │ • Format results  │
│ • Validate     │ • Display summary │
│ • Translate    │ • Export files    │
│ • Delegate     │ • User-friendly   │
└─────────────────┴───────────────────┘
         ↓
┌─────────────────┐
│  Service Layer  │ ← Business logic, orchestration  
├─────────────────┤
│ Analysis Layer  │ ← Power/energy calculations
├─────────────────┤
│  Query Layer    │ ← Database queries, validation
├─────────────────┤
│  Core Layer     │ ← Connections, configuration
└─────────────────┘
         ↓
    Database
         ↓
    Raw Results
         ↓
    Back to Reporting Layer
```

### Module Responsibilities

**Presentation Layer:**
- **CLI (`src/cli/`)**: Input side - accepts user commands, validates parameters, translates to service calls
- **Reporting (`src/reporting/`)**: Output side - formats results, displays summaries, exports files

**Business Logic Layer:**
- **Services (`src/services/`)**: Orchestration - coordinates analysis operations, business rules

**Data Processing Layer:**
- **Analysis (`src/analysis/`)**: Calculations - power analysis, energy computation, data processing
- **Queries (`src/queries/`)**: Data access - SQL generation, query validation, database interaction

**Infrastructure Layer:**
- **Database (`src/database/`)**: Foundation - database connections, configuration, connection pooling
- **Constants (`src/constants/`)**: Definitions - node lists, metric definitions, system constants
- **Utils (`src/utils/`)**: Utilities - conversions, data processing, query helpers, node detection

### Request Flow Example
```
User: python -m src.cli analyze --database h100 --hostname rpg-93-1
  ↓
CLI: Parse arguments, validate input
  ↓
Service: PowerAnalysisService.analyze_node_power()
  ↓
Analysis: PowerAnalyzer.analyze_power() + EnergyCalculator.calculate_energy()
  ↓
Queries: QueryManager.get_power_metrics()
  ↓
Database: ConnectionPool → Database
  ↓
Database: Execute SQL, return raw data
  ↓
Analysis: Process data, calculate energy, create summaries
  ↓
Service: Combine results, create business response
  ↓
Reporting: Format results (Excel/CSV/JSON) + display summary
  ↓
CLI: Show results to user + save files
```

## License

[License information]
