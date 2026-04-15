"""Intel GPU provider — parses xpu-smi subprocess output."""

from __future__ import annotations

import asyncio
import json
import shutil

from llmmanager.exceptions import GPUQueryError
from llmmanager.gpu.base import AbstractGPUProvider
from llmmanager.models.gpu import GPUInfo, GPUVendor, VRAMInfo


class IntelProvider(AbstractGPUProvider):
    vendor = GPUVendor.INTEL

    @classmethod
    def is_available(cls) -> bool:
        return shutil.which("xpu-smi") is not None

    async def initialize(self) -> None:
        pass

    async def get_all_gpus(self) -> list[GPUInfo]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "xpu-smi", "dump", "-d", "-1", "-m",
                "0,1,2,5,17,18,19",  # util, power, freq, temp, vram used/free/total
                "-j",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            return self._parse(stdout.decode())
        except asyncio.TimeoutError as exc:
            raise GPUQueryError("xpu-smi timed out") from exc
        except Exception as exc:
            raise GPUQueryError(f"Intel GPU query failed: {exc}") from exc

    def _parse(self, output: str) -> list[GPUInfo]:
        try:
            data = json.loads(output)
        except json.JSONDecodeError as exc:
            raise GPUQueryError(f"Failed to parse xpu-smi JSON: {exc}") from exc

        gpus: list[GPUInfo] = []
        for device in data.get("device_list", []):
            device_id = device.get("device_id", 0)
            metrics = {m["metrics_type"]: m.get("value") for m in device.get("metrics", [])}

            def _f(key: str) -> float | None:
                v = metrics.get(key)
                try:
                    return float(v) if v is not None else None
                except (TypeError, ValueError):
                    return None

            total_mb = _f("XPUM_STATS_MEMORY_USED") or 0.0
            used_mb = _f("XPUM_STATS_MEMORY_UTILIZATION") or 0.0
            free_mb = max(total_mb - used_mb, 0.0)

            vram = VRAMInfo(total_mb=total_mb, used_mb=used_mb, free_mb=free_mb)

            gpus.append(GPUInfo(
                index=device_id,
                name=device.get("device_name", f"Intel GPU {device_id}"),
                vendor=GPUVendor.INTEL,
                vram=vram,
                utilization_pct=_f("XPUM_STATS_GPU_UTILIZATION") or 0.0,
                temperature_c=_f("XPUM_STATS_GPU_CORE_TEMPERATURE"),
                power_watts=_f("XPUM_STATS_POWER"),
            ))
        return gpus

    async def shutdown(self) -> None:
        pass
