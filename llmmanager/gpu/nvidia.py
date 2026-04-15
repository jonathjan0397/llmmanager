"""NVIDIA GPU provider via pynvml (from nvidia-ml-py package)."""

from __future__ import annotations

import asyncio
import warnings

from llmmanager.exceptions import GPUQueryError
from llmmanager.gpu.base import AbstractGPUProvider
from llmmanager.models.gpu import GPUInfo, GPUVendor, VRAMInfo


class NvidiaProvider(AbstractGPUProvider):
    vendor = GPUVendor.NVIDIA

    @classmethod
    def is_available(cls) -> bool:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                import pynvml  # noqa: F401
            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
            return True
        except Exception:
            return False

    async def initialize(self) -> None:
        await asyncio.to_thread(self._init_sync)

    def _init_sync(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import pynvml
        pynvml.nvmlInit()

    async def get_all_gpus(self) -> list[GPUInfo]:
        return await asyncio.to_thread(self._query_sync)

    def _query_sync(self) -> list[GPUInfo]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                import pynvml
            count = pynvml.nvmlDeviceGetCount()
            gpus: list[GPUInfo] = []
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()

                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram = VRAMInfo(
                    total_mb=mem.total / 1024**2,
                    used_mb=mem.used / 1024**2,
                    free_mb=mem.free / 1024**2,
                )

                util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:
                    temp = None

                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except Exception:
                    power = None

                try:
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                except Exception:
                    power_limit = None

                try:
                    fan = pynvml.nvmlDeviceGetFanSpeed(handle)
                except Exception:
                    fan = None

                try:
                    driver = pynvml.nvmlSystemGetDriverVersion()
                    if isinstance(driver, bytes):
                        driver = driver.decode()
                except Exception:
                    driver = None

                try:
                    cuda_ver = pynvml.nvmlSystemGetCudaDriverVersion_v2()
                    major, minor = divmod(cuda_ver, 1000)
                    cuda = f"{major}.{minor // 10}"
                except Exception:
                    cuda = None

                gpus.append(GPUInfo(
                    index=i,
                    name=name,
                    vendor=GPUVendor.NVIDIA,
                    vram=vram,
                    utilization_pct=float(util.gpu),
                    temperature_c=float(temp) if temp is not None else None,
                    power_watts=power,
                    power_limit_watts=power_limit,
                    fan_speed_pct=float(fan) if fan is not None else None,
                    driver_version=driver,
                    cuda_version=cuda,
                ))
            return gpus
        except Exception as exc:
            raise GPUQueryError(f"NVIDIA query failed: {exc}") from exc

    async def shutdown(self) -> None:
        await asyncio.to_thread(self._shutdown_sync)

    def _shutdown_sync(self) -> None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                import pynvml
            pynvml.nvmlShutdown()
        except Exception:
            pass
