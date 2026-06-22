"""
Shared pytest configuration and fixtures
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add repo root before src so top-level packages such as scripts/ are not
# shadowed by compatibility packages under src/.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(REPO_ROOT, 'src')
for path in (SRC_ROOT, REPO_ROOT):
    if path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(1, SRC_ROOT)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment for all tests"""
    # Set test environment variables
    os.environ['REPACSS_TEST_MODE'] = 'true'
    os.environ['REPACSS_DB_HOST'] = 'test-db-host'
    os.environ['REPACSS_SSH_HOSTNAME'] = 'test-ssh-host'
    
    yield
    
    # Cleanup after test
    test_vars = ['REPACSS_TEST_MODE', 'REPACSS_DB_HOST', 'REPACSS_SSH_HOSTNAME']
    for var in test_vars:
        if var in os.environ:
            del os.environ[var]


@pytest.fixture
def mock_all_external_dependencies():
    """Mock all external dependencies for unit tests"""
    with patch('src.database.client.subprocess.Popen') as mock_ssh, \
         patch('src.database.client.psycopg2.connect') as mock_db, \
         patch('src.database.database.DatabaseManager') as mock_manager:
        
        # Configure mocks
        mock_ssh.return_value.poll.return_value = None
        mock_db.return_value.info.host = 'localhost'
        mock_db.return_value.info.port = 5432
        mock_db.return_value.info.user = 'test_user'
        mock_db.return_value.info.password = 'test_password'
        mock_db.return_value.info.dbname = 'test_db'
        
        yield {
            'ssh': mock_ssh,
            'database': mock_db,
            'manager': mock_manager
        }


@pytest.fixture
def sample_time_range():
    """Sample time range for testing"""
    from datetime import datetime, timedelta
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    return start_time, end_time


@pytest.fixture
def test_hostnames():
    """Sample hostnames for testing"""
    return {
        'h100_nodes': ['rpg-93-1', 'rpg-93-2', 'rpg-93-3'],
        'zen4_nodes': ['rpc-91-1', 'rpc-91-2', 'rpc-91-3'],
        'irc_nodes': ['irc-node-1', 'irc-node-2'],
        'pdu_nodes': ['pdu-node-1', 'pdu-node-2']
    }


@pytest.fixture
def test_rack_numbers():
    """Sample rack numbers for testing"""
    return [91, 92, 93, 94, 95, 96, 97]


@pytest.fixture
def mock_power_analysis_service():
    """Mock power analysis service for testing"""
    with patch('src.services.power_service.PowerAnalysisService') as mock_service:
        mock_instance = MagicMock()
        mock_service.return_value = mock_instance
        
        # Configure default return values
        mock_instance.analyze_node_power.return_value = {
            'hostname': 'test-node',
            'energy_consumption': {
                'CPUPower': 1.5,
                'GPUPower': 2.3,
                'SystemPower': 4.1
            },
            'total_energy_kwh': 7.9
        }
        
        yield mock_instance


@pytest.fixture
def mock_energy_calculator():
    """Mock energy calculator for testing"""
    with patch('src.analysis.energy.EnergyCalculator') as mock_calc:
        mock_instance = MagicMock()
        mock_calc.return_value = mock_instance
        
        # Configure default return values
        mock_instance.calculate_energy.return_value = {
            'CPUPower': 1.5,
            'GPUPower': 2.3,
            'SystemPower': 4.1
        }
        
        yield mock_instance


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on directory"""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.slow)
