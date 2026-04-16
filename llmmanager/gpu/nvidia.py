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

    async def set_fan_speed(self, gpu_index: int, speed_pct: int) -> tuple[bool, str]:
        return await asyncio.to_thread(self._set_fan_sync, gpu_index, max(0, min(100, speed_pct)))

    def _set_fan_sync(self, gpu_index: int, speed_pct: int) -> tuple[bool, str]:
        import sys
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

            # nvmlDeviceSetFanSpeed_v2 / nvmlDeviceSetFanControlPolicy are Linux-only
            # and not available in the Windows NVML driver.
            if sys.platform == "win32":
                return False, (
                    "Fan control via NVML is not available on Windows. "
                    "Use MSI Afterburner or your GPU vendor's software instead."
                )

            try:
                num_fans = pynvml.nvmlDeviceGetNumFans(handle)
            except Exception:
                num_fans = 1

            NVML_FAN_POLICY_MANUAL = 1  # nvml.h: TEMPERATURE_CONTINOUS_SW=0, MANUAL=1
            for fan_idx in range(num_fans):
                pynvml.nvmlDeviceSetFanControlPolicy(handle, fan_idx, NVML_FAN_POLICY_MANUAL)
                pynvml.nvmlDeviceSetFanSpeed_v2(handle, fan_idx, speed_pct)

            return True, f"GPU {gpu_index}: fans set to {speed_pct}%"
        except AttributeError:
            return False, (
                "pynvml version does not support fan writes. "
                "Upgrade: pip install --upgrade nvidia-ml-py"
            )
        except Exception as exc:
            msg = str(exc)
            if "Insufficient Permissions" in msg or "NOT_SUPPORTED" in msg:
                return False, (
                    f"Permission denied — fan control requires root on Linux. "
                    f"Run: sudo llmmanager  (or grant nvidia-persistenced access)"
                )
            return False, f"NVML error: {msg}"

    async def set_fan_auto(self, gpu_index: int) -> tuple[bool, str]:
        return await asyncio.to_thread(self._set_fan_auto_sync, gpu_index)

    def _set_fan_auto_sync(self, gpu_index: int) -> tuple[bool, str]:
        import sys
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                import pynvml
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

            if sys.platform == "win32":
                return False, "Fan control via NVML is not available on Windows."

            try:
                num_fans = pynvml.nvmlDeviceGetNumFans(handle)
            except Exception:
                num_fans = 1

            NVML_FAN_POLICY_AUTO = 0  # nvml.h: TEMPERATURE_CONTINOUS_SW=0
            for fan_idx in range(num_fans):
                pynvml.nvmlDeviceSetFanControlPolicy(
                    handle, fan_idx, NVML_FAN_POLICY_AUTO
                )

            return True, f"GPU {gpu_index}: fans returned to automatic control"
        except AttributeError:
            return False, "pynvml version does not support fan writes."
        except Exception as exc:
            msg = str(exc)
            if "Insufficient Permissions" in msg or "NOT_SUPPORTED" in msg:
                return False, "Permission denied — fan control requires root on Linux."
            return False, f"NVML error: {msg}"

    async def set_fan_speed_sudo(self, gpu_index: int, speed_pct: int, sudo_password: str) -> tuple[bool, str]:
        speed_pct = max(0, min(100, speed_pct))
        script = (
            f"import pynvml; pynvml.nvmlInit(); "
            f"h=pynvml.nvmlDeviceGetHandleByIndex({gpu_index}); "
            f"n=pynvml.nvmlDeviceGetNumFans(h) if hasattr(pynvml,'nvmlDeviceGetNumFans') else 1; "
            f"[pynvml.nvmlDeviceSetFanControlPolicy(h,f,1) or "
            f"pynvml.nvmlDeviceSetFanSpeed_v2(h,f,{speed_pct}) for f in range(n)]"
        )
        ok, msg = await asyncio.to_thread(self._run_sudo_python, script, sudo_password)
        return (True, f"GPU {gpu_index}: fans set to {speed_pct}%") if ok else (False, msg)

    async def set_fan_auto_sudo(self, gpu_index: int, sudo_password: str) -> tuple[bool, str]:
        script = (
            f"import pynvml; pynvml.nvmlInit(); "
            f"h=pynvml.nvmlDeviceGetHandleByIndex({gpu_index}); "
            f"n=pynvml.nvmlDeviceGetNumFans(h) if hasattr(pynvml,'nvmlDeviceGetNumFans') else 1; "
            f"[pynvml.nvmlDeviceSetFanControlPolicy(h,f,0) for f in range(n)]"
        )
        ok, msg = await asyncio.to_thread(self._run_sudo_python, script, sudo_password)
        return (True, f"GPU {gpu_index}: fans returned to automatic control") if ok else (False, msg)

    def _run_sudo_python(self, script: str, sudo_password: str) -> tuple[bool, str]:
        import subprocess, sys
        try:
            proc = subprocess.run(
                ["sudo", "-S", sys.executable, "-c", script],
                input=f"{sudo_password}\n".encode(),
                capture_output=True,
                timeout=30,
            )
            if proc.returncode != 0:
                err = proc.stderr.decode().strip()
                if any(w in err.lower() for w in ("incorrect password", "authentication failure", "sorry")):
                    return False, "Incorrect sudo password."
                return False, f"sudo error: {err}"
            return True, "ok"
        except subprocess.TimeoutExpired:
            return False, "sudo timed out."
        except Exception as exc:
            return False, str(exc)

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
