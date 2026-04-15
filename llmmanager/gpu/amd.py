"""AMD GPU provider — parses rocm-smi subprocess output."""

from __future__ import annotations

import asyncio
import json
import shutil

from llmmanager.exceptions import GPUQueryError
from llmmanager.gpu.base import AbstractGPUProvider
from llmmanager.models.gpu import GPUInfo, GPUVendor, VRAMInfo


class AMDProvider(AbstractGPUProvider):
    vendor = GPUVendor.AMD

    @classmethod
    def is_available(cls) -> bool:
        return shutil.which("rocm-smi") is not None

    async def initialize(self) -> None:
        pass  # No persistent state needed

    async def get_all_gpus(self) -> list[GPUInfo]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "rocm-smi", "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            return self._parse(stdout.decode())
        except asyncio.TimeoutError as exc:
            raise GPUQueryError("rocm-smi timed out") from exc
        except Exception as exc:
            raise GPUQueryError(f"AMD query failed: {exc}") from exc

    def _parse(self, output: str) -> list[GPUInfo]:
        try:
            data = json.loads(output)
        except json.JSONDecodeError as exc:
            raise GPUQueryError(f"Failed to parse rocm-smi JSON: {exc}") from exc

        gpus: list[GPUInfo] = []
        for i, (key, card) in enumerate(data.items()):
            if key == "system":
                continue

            def _float(val: str | None) -> float | None:
                try:
                    return float(str(val).replace("%", "").replace("W", "").strip())
                except (TypeError, ValueError):
                    return None

            total_mb = _float(card.get("VRAM Total Memory (B)"))
            used_mb = _float(card.get("VRAM Total Used Memory (B)"))
            if total_mb is not None:
                total_mb /= 1024**2
            if used_mb is not None:
                used_mb /= 1024**2
            free_mb = (total_mb - used_mb) if (total_mb and used_mb) else 0.0

            vram = VRAMInfo(
                total_mb=total_mb or 0.0,
                used_mb=used_mb or 0.0,
                free_mb=free_mb,
            )

            gpus.append(GPUInfo(
                index=i,
                name=card.get("Card Series", f"AMD GPU {i}"),
                vendor=GPUVendor.AMD,
                vram=vram,
                utilization_pct=_float(card.get("GPU use (%)")) or 0.0,
                temperature_c=_float(card.get("Temperature (Sensor edge) (C)")),
                power_watts=_float(card.get("Average Graphics Package Power (W)")),
                fan_speed_pct=_float(card.get("Fan speed (%)")),
            ))
        return gpus

    async def shutdown(self) -> None:
        pass
