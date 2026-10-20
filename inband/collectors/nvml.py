"""NVIDIA GPU energy collector backed by NVML's total energy counter.

`nvmlDeviceGetTotalEnergyConsumption` reports each GPU's energy since the driver was
loaded, in millijoules, on Volta and newer GPUs. The collector differences consecutive
readings exactly as the RAPL collector does, so CPU and GPU energy follow one method.
NVML is the `libnvidia-ml.so.1` library that ships with the driver, called through
ctypes, so no Python NVML package is needed. curvu reads NVIDIA GPUs through NVML too.

Rows use the RAPL collector's columns: each GPU is a domain whose path is its UUID and
whose group is `gpu`, and millijoules are stored as microjoules. The 64-bit counter
does not wrap; a smaller reading means the driver was reloaded, so that interval's
energy is unknown and the row is flagged `wrap_ambiguous`.

The collector samples every GPU on the node. That equals the job's allocation only
when the job has the node exclusively, which the in-band report already assumes.
"""

from __future__ import annotations

import ctypes
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from inband.collectors.base import CollectorProbe, PollingCSVCollector
from inband.collectors.rapl import COUNTER_COLUMNS
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
        return list(COUNTER_COLUMNS)

    def prepare_runtime(self) -> Dict[str, Any]:
        return {"devices": self._usable(), "previous": {}}

    def collect_rows(self, runtime: Dict[str, Any], elapsed_seconds: float) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        timestamp = datetime.now(timezone.utc).isoformat()
        hostname = self.hostname()
        for device in runtime["devices"]:
            try:
                energy_uj = self._library.energy_mj(device["handle"]) * 1000
            except NvmlError:
                # A failed read shows up as a longer interval on the next row.
                continue
            read_at = time.monotonic()

            delta_uj = interval_seconds = power_watts = None
            reset = False
            previous = runtime["previous"].get(device["uuid"])
            if previous is not None:
                previous_uj, previous_at = previous
                interval_seconds = round(read_at - previous_at, 6)
                if energy_uj >= previous_uj:
                    delta_uj = energy_uj - previous_uj
                    if interval_seconds > 0:
                        power_watts = round(delta_uj / 1_000_000.0 / interval_seconds, 6)
                else:
                    reset = True  # the driver was reloaded; this interval's energy is unknown
            runtime["previous"][device["uuid"]] = (energy_uj, read_at)

            rows.append(
                {
                    "timestamp": timestamp,
                    "elapsed_seconds": round(elapsed_seconds, 6),
                    "hostname": hostname,
                    "profile": self.name,
                    "domain_path": device["uuid"],
                    "domain_name": f"gpu{device['index']}",
                    "domain_group": GPU_GROUP,
                    "energy_uj": energy_uj,
                    "delta_uj": delta_uj,
                    "interval_seconds": interval_seconds,
                    "power_watts": power_watts,
                    "wrapped": False,
                    "wrap_ambiguous": reset,
                }
            )
        return rows
