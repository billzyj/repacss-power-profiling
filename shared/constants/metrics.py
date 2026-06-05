"""Metric definitions for REPACSS power profiling."""

IRC_POWER_METRICS = [
    "CompressorPower",
    "CondenserFanPower",
]

IRC_ALL_METRICS = [
    "CompressorPower",
    "CondenserFanPower",
    "CoolDemand",
    "CoolOutput",
    "TotalAirSideCoolingDemand",
    "TotalSensibleCoolingPower",
]

IRC_SENSING_FREQUENCY = 120

PDU_POWER_METRICS = ["pdu"]

PDU_SENSING_FREQUENCY = 60

COMPUTE_SENSING_FREQUENCY = 5

EXCLUDED_METRICS = [
    "systemheadroominstantaneous",
]

DERIVED_METRICS = [
    "computepower",
    "systemheadroominstantaneous",
]

H100_METRICS = [
    "TotalCPUPower",
    "TotalMemoryPower",
    "TotalStoragePower",
    "TotalFanPower",
    "SystemInputPower",
    "SystemOutputPower",
    "PowerConsumption",
]

ZEN4_METRICS = [
    "TotalCPUPower",
    "TotalMemoryPower",
    "TotalStoragePower",
    "TotalFanPower",
    "SystemInputPower",
    "SystemOutputPower",
]
