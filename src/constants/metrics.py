"""
Metric definitions for REPACSS Power Measurement
"""

# IRC (Infrastructure) power metrics for PUE calculation
# Only CompressorPower and CondenserFanPower represent actual power consumption
IRC_POWER_METRICS = [
    'CompressorPower',  # Unit: kW
    'CondenserFanPower',  # Unit: W
]

# All IRC metrics (for other analysis purposes)
IRC_ALL_METRICS = [
    'CompressorPower', 
    'CondenserFanPower', 
    'CoolDemand', 
    'CoolOutput', 
    'TotalAirSideCoolingDemand', 
    'TotalSensibleCoolingPower'
]

IRC_SENSING_FREQUENCY = 120 #seconds

# PDU (Power Distribution Unit) power metrics
PDU_POWER_METRICS = ['pdu']

PDU_SENSING_FREQUENCY = 60 #seconds

COMPUTE_SENSING_FREQUENCY = 5 #seconds

# Metrics to exclude from graphs (not power consumption)
EXCLUDED_METRICS = [
    'systemheadroominstantaneous'  # This is remaining wattage, not consumption
]

# Derived metrics that are not real power consumption
DERIVED_METRICS = [
    'computepower',              # Compute power that is not wasted
    'systemheadroominstantaneous'  # This is remaining wattage, not consumption
]
