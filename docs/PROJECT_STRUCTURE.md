# REPACSS Power Measurement - Project Structure

## Overview

This document provides detailed technical architecture and development information for the REPACSS Power Measurement project.

## Directory Structure

```
repacss-power-measurement/
├── LICENSE                    # Project license
├── README.md                  # Main documentation and quick start
├── docs/                      # Documentation directory
│   ├── USAGE_GUIDE.md        # Detailed usage examples and troubleshooting
│   ├── PROJECT_STRUCTURE.md   # This file - technical architecture
│   └── _working/             # Temporary refactor planning artifacts
│       └── refactor-plan.md  # Temporary mixed-language working plan, deleted before merge
├── requirements.txt           # Python dependencies
├── setup.py                   # Setup script for easy installation
├── .gitignore                # Git ignore rules
└── src/                      # Source code directory
    ├── core/                 # Core infrastructure and utilities
    │   ├── client.py         # Main client library
    │   ├── database.py       # Database connection utilities
    │   └── config.py         # Configuration management
    ├── queries/              # All SQL query definitions
    │   ├── compute/          # Compute nodes (H100 + Zen4) queries
    │   │   ├── public.py     # Public schema queries
    │   │   └── idrac.py      # iDRAC schema queries
    │   └── infra/            # Infrastructure queries
    │       ├── public.py     # Public schema queries
    │       └── irc_pdu.py    # IRC and PDU queries
    ├── scripts/              # Executable scripts and runners
    │   ├── run_public_queries.py      # Main report generator
    │   ├── run_h100_queries.py        # H100-specific runner
    │   ├── run_node_level_queries.py  # Node-level analysis
    │   └── run_rack_related_queries.py # Rack analysis
    ├── examples/             # Example usage and demos
    │   ├── basic_usage.py    # Comprehensive usage examples
    │   └── test_connection.py # Connection testing
    └── templates/            # Configuration templates
        └── config_template.py # Template for configuration
```

## Core Components

### Main Client (`src/core/client.py`)

The main client library that provides:
- SSH tunnel connection management
- Database connection handling
- Query execution methods
- Power and temperature metric retrieval
- Cluster summary calculations

**Key Classes:**
- `DatabaseConfig`: Database connection configuration
- `SSHConfig`: SSH tunnel configuration  
- `REPACSSPowerClient`: Main client class

**Key Methods:**
- `connect()`: Establish SSH tunnel and database connection
- `disconnect()`: Close connections
- `get_computepower_metrics()`: Get power consumption data
- `get_boardtemperature_metrics()`: Get temperature data
- `get_idrac_cluster_summary()`: Get cluster-wide summaries
- `execute_query()`: Execute custom SQL queries

### Database Utilities (`src/core/database.py`)

Database connection management utilities:
- Multi-database connection handling
- Connection pooling and management
- Database-specific client creation
- Connection cleanup and error handling

**Key Classes:**
- `DatabaseConnectionManager`: Manages multiple database connections

**Key Functions:**
- `connect_to_database()`: Connect to a specific database
- `connect_to_all_databases()`: Connect to all configured databases
- `disconnect_all()`: Close all connections
- `get_client()`: Get client for specific database

### Configuration (`src/core/config.py`)

Configuration management system:
- Database connection settings
- SSH tunnel parameters
- Database schema mappings
- Security credentials management

**Supported Databases:**
- `h100`: H100 cluster database (idrac schema)
- `zen4`: ZEN4 cluster database (idrac schema)  
- `infra`: Infrastructure database (irc, pdu schemas)

### Query Collections (`src/queries/`)

Organized query collections for different use cases:

#### Compute Queries (`src/queries/compute/`)

**Public Schema Queries (`public.py`)**
- Power metrics definitions from `public.metrics_definition`
- FQDD (Fully Qualified Device Descriptor) information
- Metrics grouped by units and types
- High-accuracy power metrics
- Comprehensive device mapping

**iDRAC Schema Queries (`idrac.py`)**
- Recent power and temperature metrics
- Node and cluster summaries
- Time-range analysis queries
- Power efficiency analysis
- High power/temperature event detection
- Unified API for power metrics with joins

#### Infrastructure Queries (`src/queries/infra/`)

**Public Schema Queries (`public.py`)**
- Infrastructure metrics definitions
- Compressor and airflow metrics
- Run hours and system metrics
- Infrastructure-specific queries

**IRC/PDU Schema Queries (`irc_pdu.py`)**
- PDU (Power Distribution Unit) power metrics
- IRC (Infrastructure) temperature and humidity monitoring
- Infrastructure efficiency analysis
- Alert conditions (high power, temperature, low humidity)

### Scripts (`src/scripts/`)

#### Main Report Generator (`run_public_queries.py`)
Comprehensive Excel report generator:
- Multi-database power metrics collection
- Excel file generation with multiple sheets
- Metric data with cross-joins for human-readable names
- Timestamped output files

#### H100-Specific Runner (`run_h100_queries.py`)
Advanced script for running queries across multiple databases:
- Simultaneous multi-database connections
- Metrics definition queries
- Power analysis queries
- Time range analysis

#### Node-Level Analysis (`run_node_level_queries.py`)
Node-specific analysis and visualization:
- Individual node power analysis
- Temperature monitoring
- Power trend analysis
- Node comparison reports

#### Rack Analysis (`run_rack_related_queries.py`)
Rack-level infrastructure analysis:
- Rack power consumption
- Cooling efficiency analysis
- Infrastructure monitoring

### Examples (`src/examples/`)

#### Basic Usage (`basic_usage.py`)
Comprehensive demonstration of client capabilities:
- Single database operations
- Multi-database comparisons
- Available metrics discovery
- Recent metrics retrieval
- Cluster summaries
- Custom query execution

#### Test Connection (`test_connection.py`)
Simple connection testing script:
- SSH tunnel verification
- Database connectivity test
- Table existence checks
- Basic query execution

## Database Schemas

### H100 and ZEN4 Databases
- **public**: Metrics definitions and metadata
- **idrac**: iDRAC power and temperature data (default)
- **slurm**: Slurm job and resource management

### INFRA Database
- **public**: Infrastructure metrics definitions
- **irc**: Infrastructure monitoring (temperature, humidity, airflow)
- **pdu**: Power Distribution Unit data (default)

## Security Features

- SSH tunnel encryption for database connections
- Configuration file excluded from version control
- Private key authentication
- Connection isolation per database
- Automatic connection cleanup

## Error Handling

- Robust SSH tunnel management
- Database connection error recovery
- Query execution error handling
- Graceful disconnection on errors
- Comprehensive logging

## Dependencies

- `psycopg2-binary`: PostgreSQL adapter
- `paramiko`: SSH protocol implementation
- `sshtunnel`: SSH tunnel management

## Development Workflow

1. **Setup**: Run `python setup.py`
2. **Configure**: Edit `src/core/config.py` with credentials
3. **Test**: Run `python src/examples/test_connection.py`
4. **Develop**: Use `src/examples/basic_usage.py` as reference
5. **Deploy**: Ensure `src/core/config.py` is in `.gitignore`

## Temporary Planning Artifacts

`docs/_working/` is reserved for temporary refactor collaboration artifacts.

Rules:

1. Files in `docs/_working/` may be used by both Codex and Claude Code during active refactor work.
2. Temporary planning files may use mixed Chinese and English when that helps collaboration.
3. Stable project documentation must remain in English.
4. Durable decisions must be copied into English documentation before the refactor is finalized.
5. Temporary planning files must be deleted before merge or long-term publication.

## File Naming Conventions

- **Core files**: `core/client.py`, `core/database.py`, `core/config.py`
- **Configuration**: `templates/config_template.py` → `core/config.py`
- **Examples**: `examples/basic_usage.py`, `examples/test_connection.py`
- **Query collections**: `queries/{system}/{schema}.py`
- **Scripts**: `scripts/run_{purpose}.py`

## Architecture Principles

### Separation of Concerns
- **Core**: Infrastructure and utilities
- **Queries**: SQL query definitions organized by system
- **Scripts**: Executable applications
- **Examples**: Learning and testing materials
- **Templates**: Configuration templates

### Modularity
- Each directory has a clear, single purpose
- Related functionality is grouped together
- Dependencies are minimized between modules
- Clear interfaces between components

### Maintainability
- Consistent naming conventions
- Clear documentation and examples
- Proper error handling and logging
- Version control best practices

## Best Practices

- Always use virtual environments
- Never commit `src/core/config.py`
- Test connections before running queries
- Use appropriate schemas for each database
- Handle connection cleanup in try/finally blocks
- Use parameterized queries for security
- Follow the established directory structure
- Add proper error handling to new features
- Update documentation when adding new functionality
