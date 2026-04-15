"""CPU-only fallback provider — psutil-based, no GPU stats."""

from __future__ import annotations

from llmmanager.gpu.base import AbstractGPUProvider
from llmmanager.models.gpu import GPUInfo, GPUVendor, VRAMInfo


class CPUOnlyProvider(AbstractGPUProvider):
    vendor = GPUVendor.CPU_ONLY

    @classmethod
    def is_available(cls) -> bool:
        return True  # Always available as the final fallback

    async def initialize(self) -> None:
        pass

    async def get_all_gpus(self) -> list[GPUInfo]:
        import psutil
        ram = psutil.virtual_memory()
        # Represent system RAM as the "VRAM" for CPU inference
        vram = VRAMInfo(
            total_mb=ram.total / 1024**2,
            used_mb=ram.used / 1024**2,
            free_mb=ram.available / 1024**2,
        )
        return [GPUInfo(
            index=0,
            name="CPU (System RAM)",
            vendor=GPUVendor.CPU_ONLY,
            vram=vram,
            utilization_pct=psutil.cpu_percent(interval=None),
        )]

    async def shutdown(self) -> None:
        pass
