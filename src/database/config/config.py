#!/usr/bin/env python3
"""
Enhanced configuration management for REPACSS Power Measurement
Supports environment variables and improved security
"""

import os
import getpass
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    host: str
    port: int
    database: str
    username: str
    password: str
    ssl_mode: str
    schema: str


@dataclass
class SSHConfig:
    """SSH tunnel configuration"""
    hostname: str
    port: int
    username: str
    private_key_path: str
    passphrase: str
    keepalive_interval: int


@dataclass
class Config:
    """Enhanced configuration for database and SSH connection with environment variable support"""
    
    # Database settings (with environment variable fallbacks)
    db_host: str = None
    db_port: int = None
    db_default_name: str = None
    db_user: str = None
    db_password: str = None
    db_ssl_mode: str = None
    
    # SSH tunnel settings (with environment variable fallbacks)
    ssh_hostname: str = None
    ssh_port: int = None
    ssh_username: str = None
    ssh_private_key_path: str = None
    ssh_passphrase: str = None
    ssh_keepalive_interval: int = None
    
    # Database schemas configuration
    _database_schemas: dict = None
    
    def __post_init__(self):
        # Load from environment variables with fallbacks
        self.db_host = os.getenv('REPACSS_DB_HOST', self.db_host or 'localhost')
        self.db_port = int(os.getenv('REPACSS_DB_PORT', self.db_port or 5432))
        self.db_default_name = os.getenv('REPACSS_DB_DEFAULT', self.db_default_name or 'h100')
        self.db_user = os.getenv('REPACSS_DB_USER', self.db_user or 'monster')
        self.db_password = os.getenv('REPACSS_DB_PASSWORD', self.db_password or 'repacss')
        self.db_ssl_mode = os.getenv('REPACSS_DB_SSL_MODE', self.db_ssl_mode or 'prefer')
        
        # Note: treat empty strings in .env as "unset" so users can rely on
        # system ssh defaults (current username, ssh-agent, ~/.ssh/config).
        ssh_hostname_env = os.getenv('REPACSS_SSH_HOSTNAME')
        self.ssh_hostname = (ssh_hostname_env or "").strip() or (self.ssh_hostname or 'narumuu.ttu.edu')

        ssh_port_env = os.getenv('REPACSS_SSH_PORT')
        self.ssh_port = int((ssh_port_env or "").strip() or (self.ssh_port or 22))

        ssh_username_env = os.getenv('REPACSS_SSH_USERNAME')
        self.ssh_username = (ssh_username_env or "").strip() or (self.ssh_username or getpass.getuser())

        ssh_key_env = os.getenv('REPACSS_SSH_KEY_PATH')
        self.ssh_private_key_path = (ssh_key_env or "").strip() or (self.ssh_private_key_path or "")
        # Normalize empty -> None to mean "use default keys/agent"
        if not self.ssh_private_key_path:
            self.ssh_private_key_path = None

        ssh_passphrase_env = os.getenv('REPACSS_SSH_PASSPHRASE')
        self.ssh_passphrase = (ssh_passphrase_env or "").strip() or (self.ssh_passphrase or '')

        ssh_keepalive_env = os.getenv('REPACSS_SSH_KEEPALIVE')
        self.ssh_keepalive_interval = int((ssh_keepalive_env or "").strip() or (self.ssh_keepalive_interval or 60))
        
        if self._database_schemas is None:
            self._database_schemas = {
                "h100": {
                    "schemas": ["public", "idrac", "slurm"],
                    "default_schema": "idrac"
                },
                "zen4": {
                    "schemas": ["public", "idrac", "slurm"],
                    "default_schema": "idrac"
                },
                "infra": {
                    "schemas": ["public", "irc", "pdu"],
                    "default_schema": "pdu"
                }
            }
    
    @property
    def databases(self) -> List[str]:
        """Get list of available databases"""
        return list(self._database_schemas.keys())
    
    def get_database_config(self, database_name: str = None, schema: str = None) -> DatabaseConfig:
        """Get database configuration for a specific database and schema"""
        if database_name is None:
            database_name = self.db_default_name
        
        if schema is None:
            schema = self._database_schemas.get(database_name, {}).get("default_schema", "public")
        
        return DatabaseConfig(
            host=self.db_host,
            port=self.db_port,
            database=database_name,
            username=self.db_user,
            password=self.db_password,
            ssl_mode=self.db_ssl_mode,
            schema=schema
        )
    
    def get_ssh_config(self) -> SSHConfig:
        """Get SSH configuration"""
        return SSHConfig(
            hostname=self.ssh_hostname,
            port=self.ssh_port,
            username=self.ssh_username,
            private_key_path=self.ssh_private_key_path,
            passphrase=self.ssh_passphrase,
            keepalive_interval=self.ssh_keepalive_interval
        )
    
    def get_available_schemas(self, database_name: str) -> List[str]:
        """Get available schemas for a specific database"""
        return self._database_schemas.get(database_name, {}).get("schemas", ["public"])
    
    def get_default_schema(self, database_name: str) -> str:
        """Get default schema for a specific database"""
        return self._database_schemas.get(database_name, {}).get("default_schema", "public")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check database settings
        if not self.db_host:
            issues.append("Database host not configured")
        if not self.db_user:
            issues.append("Database user not configured")
        if not self.db_password:
            issues.append("Database password not configured")
        
        # Check SSH settings
        if not self.ssh_hostname:
            issues.append("SSH hostname not configured")
        # Username and key are optional: system ssh can use current user and
        # default keys/ssh-agent based on ~/.ssh/config.
        
        # Check SSH key file
        if self.ssh_private_key_path and not Path(self.ssh_private_key_path).exists():
            issues.append(f"SSH private key file not found: {self.ssh_private_key_path}")
        
        return issues
    
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return len(self.validate_config()) == 0
    
    def load_from_env_file(self, env_file_path: str = None):
        """Load configuration from .env file"""
        if env_file_path is None:
            env_file_path = str(Path(__file__).resolve().parent / ".env")

        env_path = Path(env_file_path)
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()

            self.__post_init__()
    
    def save_to_env_file(self, env_file_path: str = '.env'):
        """Save current configuration to .env file"""
        env_content = [
            "# REPACSS Power Measurement Configuration",
            "# Database Settings",
            f"REPACSS_DB_HOST={self.db_host}",
            f"REPACSS_DB_PORT={self.db_port}",
            f"REPACSS_DB_DEFAULT={self.db_default_name}",
            f"REPACSS_DB_USER={self.db_user}",
            f"REPACSS_DB_PASSWORD={self.db_password}",
            f"REPACSS_DB_SSL_MODE={self.db_ssl_mode}",
            "",
            "# SSH Settings",
            f"REPACSS_SSH_HOSTNAME={self.ssh_hostname}",
            f"REPACSS_SSH_PORT={self.ssh_port}",
            f"REPACSS_SSH_USERNAME={self.ssh_username}",
            f"REPACSS_SSH_KEY_PATH={self.ssh_private_key_path}",
            f"REPACSS_SSH_PASSPHRASE={self.ssh_passphrase}",
            f"REPACSS_SSH_KEEPALIVE={self.ssh_keepalive_interval}",
        ]
        
        with open(env_file_path, 'w') as f:
            f.write('\n'.join(env_content))
        
        print(f"✅ Configuration saved to {env_file_path}")
        print("⚠️  Remember to add .env to .gitignore to keep credentials secure")


# Global configuration instance
config = Config()

# Try to load from .env file if it exists
config.load_from_env_file()