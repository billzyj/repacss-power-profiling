"""NVIDIA GPU energy collector backed by NVML's total energy counter.

`nvmlDeviceGetTotalEnergyConsumption` reports each GPU's energy since the driver was
loaded, in millijoules, on Volta and newer GPUs. Like the RAPL collector, this one
records only the raw readings, so CPU and GPU energy follow one method at finalization.
NVML is the `libnvidia-ml.so.1` library that ships with the driver, called through
ctypes, so no Python NVML package is needed. curvu reads NVIDIA GPUs through NVML too.

Each GPU is a domain whose path, and column name, is its UUID and whose group is `gpu`.
The 64-bit counter does not wrap; a smaller reading means the driver was reloaded, so
finalization treats that interval's energy as unknown.

The collector samples every GPU on the node. That equals the job's allocation only
when the job has the node exclusively, which the in-band report already assumes.
"""

from __future__ import annotations

import ctypes
from typing import Any, Callable, Dict, List, Optional

from inband.collectors.base import CollectorProbe, PollingCSVCollector
from shared.errors import InbandCollectorError

NVML_LIBRARY = "libnvidia-ml.so.1"
NVML_SUCCESS = 0
GPU_GROUP = "gpu"


class NvmlError(InbandCollectorError):
    """Raised when an NVML call returns an error code."""


class NvmlLibrary:
    """The few NVML calls this collector needs, through ctypes."""

    def __init__(self, path: str = NVML_LIBRARY):
        lib = ctypes.CDLL(path)
        lib.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        lib.nvmlDeviceGetUUID.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        lib.nvmlDeviceGetName.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
        lib.nvmlDeviceGetTotalEnergyConsumption.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
        lib.nvmlSystemGetDriverVersion.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        self._lib = lib
        self._check(lib.nvmlInit_v2(), "nvmlInit_v2")

    @staticmethod
    def _check(code: int, call: str) -> None:
        if code != NVML_SUCCESS:
            raise NvmlError(f"{call} returned NVML error {code}")

    def _string(self, function: Callable[..., int], *args: Any) -> str:
        buffer = ctypes.create_string_buffer(96)
        self._check(function(*args, buffer, len(buffer)), function.__name__)
        return buffer.value.decode("utf-8", errors="replace")

    def driver_version(self) -> str:
        return self._string(self._lib.nvmlSystemGetDriverVersion)

    def device_count(self) -> int:
        count = ctypes.c_uint()
        self._check(self._lib.nvmlDeviceGetCount_v2(ctypes.byref(count)), "nvmlDeviceGetCount_v2")
        return count.value

    def handle(self, index: int) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        self._check(self._lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)), "nvmlDeviceGetHandleByIndex_v2")
        return handle

    def uuid(self, handle: ctypes.c_void_p) -> str:
        return self._string(self._lib.nvmlDeviceGetUUID, handle)

    def name(self, handle: ctypes.c_void_p) -> str:
        return self._string(self._lib.nvmlDeviceGetName, handle)

    def energy_mj(self, handle: ctypes.c_void_p) -> int:
        energy = ctypes.c_ulonglong()
        self._check(
            self._lib.nvmlDeviceGetTotalEnergyConsumption(handle, ctypes.byref(energy)),
            "nvmlDeviceGetTotalEnergyConsumption",
        )
        return energy.value


class NVMLCollector(PollingCSVCollector):
    """In-band collector that samples each NVIDIA GPU's NVML energy counter."""

    name = "nvml"

    def __init__(self, library_factory: Callable[[], Any] = NvmlLibrary):
        self._library_factory = library_factory
        self._library: Any = None
        self._devices: Optional[List[Dict[str, Any]]] = None
        self._driver_version = ""
        self._error = ""

    def _discover(self) -> List[Dict[str, Any]]:
        """Open NVML and check each GPU's counter once per collector instance."""
        if self._devices is not None:
            return self._devices
        self._devices = []
        try:
            library = self._library_factory()
            self._driver_version = library.driver_version()
            for index in range(library.device_count()):
                handle = library.handle(index)
                device = {"index": index, "uuid": library.uuid(handle), "model": library.name(handle), "handle": handle}
                try:
                    library.energy_mj(handle)
                    device.update(status="ok", message="")
                except NvmlError as exc:
                    device.update(status="unreadable", message=str(exc))
                self._devices.append(device)
            self._library = library
        except (OSError, AttributeError, NvmlError) as exc:
            self._error = str(exc)
        return self._devices

    def _usable(self) -> List[Dict[str, Any]]:
        return [device for device in self._discover() if device["status"] == "ok"]

    def manifest_payload(self) -> Dict[str, Any]:
        devices = self._discover()
        usable = self._usable()
        if usable and len(usable) == len(devices):
            status = "ready"
        elif usable:
            status = "partial"
        else:
            status = "unavailable"
        return {
            "schema_version": 1,
            "kind": "repacss_nvml_manifest",
            "status": status,
            "reason": self._error or ("" if devices else "NVML reported no GPUs"),
            "driver_version": self._driver_version,
            "counter_unit": "mJ",
            "domains": [
                {
                    "path": device["uuid"],
                    "name": f"gpu{device['index']}",
                    "group": GPU_GROUP,
                    "index": device["index"],
                    "model": device["model"],
                    "status": device["status"],
                    "message": device["message"],
                }
                for device in devices
            ],
        }

    def probe(self) -> CollectorProbe:
        usable = self._usable()
        if not usable:
            reason = self._error or "no GPU with a readable NVML energy counter"
            return CollectorProbe(name=self.name, available=False, reason=reason)
        return CollectorProbe(
            name=self.name,
            available=True,
            details={"driver_version": self._driver_version, "gpu_uuids": [device["uuid"] for device in usable]},
        )

    def csv_columns(self) -> List[str]:
        return [device["uuid"] for device in self._usable()]

    def prepare_runtime(self) -> Dict[str, Any]:
        return {"devices": self._usable()}

    def collect_rows(self, runtime: Dict[str, Any]) -> List[Dict[str, Any]]:
        readings: Dict[str, Any] = {}
        for device in runtime["devices"]:
            try:
                readings[device["uuid"]] = self._library.energy_mj(device["handle"])
            except NvmlError:
                continue  # a failed read becomes an empty cell and a longer interval
        return [readings] if readings else []
