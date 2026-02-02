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

# Raw metrics queried from DB. Pie: Total = In, PSU loss = In - Out,
# Others = Total - (CPU+Memory+Storage+Fan+PSU_loss [+ GPU for H100]).
H100_METRICS = [
    'TotalCPUPower',
    'TotalMemoryPower',
    'TotalStoragePower',
    'TotalFanPower',
    'SystemInputPower',   # In = total power consumption
    'SystemOutputPower',  # Out; PSU loss = In - Out
    'PowerConsumption',   # mW GPU per FQDD; sum 4 FQDDs for pie
]

ZEN4_METRICS = [
    'TotalCPUPower',
    'TotalMemoryPower',
    'TotalStoragePower',
    'TotalFanPower',
    'SystemInputPower',   # In = total power consumption
    'SystemOutputPower',  # Out; PSU loss = In - Out
]