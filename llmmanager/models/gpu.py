"""GPU telemetry domain models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GPUVendor(str, Enum):
    NVIDIA   = "nvidia"
    AMD      = "amd"
    INTEL    = "intel"
    CPU_ONLY = "cpu_only"


@dataclass
class VRAMInfo:
    total_mb: float
    used_mb: float
    free_mb: float

    @property
    def used_pct(self) -> float:
        return (self.used_mb / self.total_mb * 100) if self.total_mb > 0 else 0.0

    @property
    def free_pct(self) -> float:
        return 100.0 - self.used_pct


@dataclass
class GPUInfo:
    index: int
    name: str
    vendor: GPUVendor
    vram: VRAMInfo
    utilization_pct: float
    temperature_c: float | None = None
    power_watts: float | None = None
    power_limit_watts: float | None = None
    fan_speed_pct: float | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
