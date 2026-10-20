"""Legacy eGauge API connector package.

Unsupported: kept as the last version that worked against the REPACSS eGauge meter.
No current REPACSS workflow depends on it.
"""

from .client import EGaugeClient
from .config import EGaugeAPIConfig, EGaugeSSHConfig, EGaugeSettings, get_eguage_settings

__all__ = [
    "EGaugeAPIConfig",
    "EGaugeClient",
    "EGaugeSettings",
    "EGaugeSSHConfig",
    "get_eguage_settings",
]
