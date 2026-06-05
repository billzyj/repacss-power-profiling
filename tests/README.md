# REPACSS Power Profiling - Test Suite

This directory contains the comprehensive test suite for the REPACSS Power Profiling system, organized using the test pyramid approach for optimal coverage and performance.

## 📁 Test Structure

```
tests/
├── README.md                 # This file - testing overview
├── conftest.py              # Shared pytest configuration and fixtures
├── unit/                    # Fast, isolated unit tests (70% of tests)
│   ├── __init__.py
│   ├── test_conversions.py  # Power unit conversions
│   ├── test_node_detection.py # Node type detection logic
│   ├── test_energy_calculation.py # Energy calculation math
│   ├── test_query_helpers.py # SQL query building
│   └── test_node_detection.py # Node type detection
├── integration/             # Component interaction tests (20% of tests)
│   ├── __init__.py
│   ├── test_database_connections.py # SSH tunnel + DB connections
│   ├── test_power_analysis.py # Service layer integration
│   ├── test_cli_commands.py # CLI command integration
│   └── test_reporting.py # Excel report generation
├── e2e/                     # End-to-end workflow tests (10% of tests)
│   ├── __init__.py
│   ├── test_rack_analysis.py # Full rack analysis workflow
│   ├── test_energy_reporting.py # Complete energy reporting
│   └── test_multi_database.py # Cross-database operations
└── fixtures/                # Test data and reusable fixtures
    ├── __init__.py
    ├── power_data.py        # Sample power data
    ├── database_setup.py    # Test database configuration
    ├── ssh_config.py        # Test SSH configurations
    └── sample_queries.py    # Test SQL queries
```

## 🧪 Test Types

### Unit Tests (70% - Fast & Isolated)
- **Purpose**: Test individual functions/methods in isolation
- **Speed**: Very fast (milliseconds)
- **Dependencies**: Mocked or stubbed
- **Examples**: Power conversions, node detection, energy calculations

### Integration Tests (20% - Component Interaction)
- **Purpose**: Test how multiple components work together
- **Speed**: Medium (seconds)
- **Dependencies**: Real test databases, SSH tunnels
- **Examples**: Database connections, service workflows, CLI commands

### End-to-End Tests (10% - Complete Workflows)
- **Purpose**: Test complete user workflows from start to finish
- **Speed**: Slow (minutes)
- **Dependencies**: Full system setup
- **Examples**: Full rack analysis, multi-database operations

## 🚀 Running Tests

### Run All Tests
```bash
# Run entire test suite
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run with verbose output
pytest -v
```

### Run Specific Test Types
```bash
# Unit tests only (fast)
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/

# Specific test file
pytest tests/unit/test_conversions.py
```

### Run with Different Verbosity
```bash
# Quiet mode
pytest -q

# Very verbose (shows all test names)
pytest -v

# Extra verbose (shows print statements)
pytest -s
```

## 🔧 Test Configuration

### Environment Setup
```bash
# Install minimal unit-test smoke dependencies
pip install -r requirements.txt -r tests/requirements-unit.txt

# Install full optional test tooling
pip install -r requirements.txt -r tests/requirements-test.txt

# Set test environment variables
export REPACSS_TEST_MODE=true
export REPACSS_DB_HOST=test-db-host
export REPACSS_SSH_HOSTNAME=test-ssh-host
```

### Test Database Setup
Tests use a separate test database to avoid affecting production data:
- **Test DB**: `repacss_test`
- **Test Schema**: `test_schema`
- **Test Data**: Sample power metrics from fixtures

## 📊 Test Coverage Goals

| Component | Unit Tests | Integration Tests | E2E Tests | Total Coverage |
|-----------|------------|-------------------|-----------|----------------|
| **Database** | 90% | 80% | 70% | 85% |
| **Analysis** | 95% | 85% | 75% | 90% |
| **CLI** | 85% | 90% | 80% | 87% |
| **Reporting** | 90% | 85% | 70% | 85% |
| **Overall** | 90% | 85% | 75% | 87% |

## 🎯 Test Categories

### Unit Tests
- **Conversions**: Power unit conversions, time calculations
- **Node Detection**: Hostname parsing, node type classification
- **Energy Math**: Energy calculations, boundary handling
- **Query Helpers**: SQL query building, parameter validation
- **Data Processing**: DataFrame operations, data cleaning

### Integration Tests
- **Database**: SSH tunnel creation, connection pooling, query execution
- **Services**: Power analysis service, energy calculation service
- **CLI**: Command parsing, argument validation, output formatting
- **Reporting**: Excel generation, chart creation, file I/O

### End-to-End Tests
- **Rack Analysis**: Complete rack power analysis workflow
- **Energy Reporting**: Full energy consumption reporting
- **Multi-Database**: Cross-database operations and data aggregation
- **Error Handling**: Error scenarios and recovery testing

## 🛠️ Test Fixtures

### Power Data Fixtures
```python
# Sample power data for different node types
@pytest.fixture
def h100_power_data():
    """Sample H100 GPU node power data"""
    return load_test_data('h100_sample.csv')

@pytest.fixture
def zen4_power_data():
    """Sample ZEN4 CPU node power data"""
    return load_test_data('zen4_sample.csv')
```

### Database Fixtures
```python
@pytest.fixture
def test_database():
    """Test database connection"""
    conn = create_test_db()
    yield conn
    conn.close()

@pytest.fixture
def test_ssh_tunnel():
    """Test SSH tunnel configuration"""
    return SSHConfig(
        hostname='test-server',
        username='test-user',
        private_key_path='test-key'
    )
```

## 📈 Continuous Integration

### GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt -r tests/requirements-test.txt
      - name: Run unit tests
        run: pytest tests/unit/ --cov=src
      - name: Run integration tests
        run: pytest tests/integration/
```

## 🐛 Debugging Tests

### Common Issues
1. **Import Errors**: Ensure `src/` is in Python path
2. **Database Connection**: Check test database configuration
3. **SSH Tunnel**: Verify test SSH credentials
4. **Data Dependencies**: Ensure test fixtures are loaded

### Debug Commands
```bash
# Run single test with debug output
pytest tests/unit/test_conversions.py::test_convert_watts_to_kwh -v -s

# Run with pdb debugger
pytest --pdb tests/integration/test_database_connections.py

# Show test collection
pytest --collect-only
```

## 📝 Writing New Tests

### Unit Test Template
```python
def test_function_name():
    """Test description"""
    # Arrange
    input_data = "test_input"
    expected_output = "expected_result"
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result == expected_output
```

### Integration Test Template
```python
def test_component_integration():
    """Test component interaction"""
    # Setup
    service = PowerAnalysisService('test_db')
    
    # Execute
    result = service.analyze_power('test_node', start_time, end_time)
    
    # Verify
    assert 'power_data' in result
    assert len(result['power_data']) > 0
```

## 🎯 Best Practices

1. **Test Naming**: Use descriptive test names that explain what is being tested
2. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
3. **Mock External Dependencies**: Use mocks for database, SSH, and file operations in unit tests
4. **Test Data**: Use fixtures for consistent test data across tests
5. **Error Cases**: Test both success and failure scenarios
6. **Performance**: Keep unit tests fast (< 100ms), integration tests reasonable (< 5s)
7. **Coverage**: Aim for high coverage but focus on critical paths

## 📚 Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Coverage](https://pytest-cov.readthedocs.io/)
- [Python Testing Best Practices](https://docs.python.org/3/library/unittest.html)
- [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development)
