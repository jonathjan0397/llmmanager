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

    async def set_fan_speed(self, gpu_index: int, speed_pct: int) -> tuple[bool, str]:
        speed_pct = max(0, min(100, speed_pct))
        pwm_value = int(speed_pct / 100 * 255)
        ok, msg = await asyncio.to_thread(self._sysfs_write, gpu_index, "pwm1_enable", "1")
        if not ok:
            return False, msg
        ok, msg = await asyncio.to_thread(self._sysfs_write, gpu_index, "pwm1", str(pwm_value))
        if ok:
            return True, f"GPU {gpu_index}: fans set to {speed_pct}% (pwm {pwm_value})"
        return False, msg

    async def set_fan_auto(self, gpu_index: int) -> tuple[bool, str]:
        ok, msg = await asyncio.to_thread(self._sysfs_write, gpu_index, "pwm1_enable", "2")
        if ok:
            return True, f"GPU {gpu_index}: fans returned to automatic control"
        return False, msg

    def _sysfs_write(self, gpu_index: int, filename: str, value: str) -> tuple[bool, str]:
        import glob as _glob
        import sys
        if sys.platform == "win32":
            return False, "AMD sysfs fan control is Linux-only."
        pattern = f"/sys/class/drm/card{gpu_index}/device/hwmon/hwmon*/{filename}"
        paths = _glob.glob(pattern)
        if not paths:
            return False, f"sysfs path not found: {pattern}"
        try:
            with open(paths[0], "w") as f:
                f.write(value)
            return True, "ok"
        except PermissionError:
            return False, "Permission denied — fan control requires root on Linux."
        except Exception as exc:
            return False, str(exc)

    async def set_fan_speed_sudo(self, gpu_index: int, speed_pct: int, sudo_password: str) -> tuple[bool, str]:
        speed_pct = max(0, min(100, speed_pct))
        pwm_value = int(speed_pct / 100 * 255)
        import glob as _glob
        pattern = f"/sys/class/drm/card{gpu_index}/device/hwmon/hwmon*"
        dirs = _glob.glob(pattern)
        if not dirs:
            return False, f"hwmon path not found for card{gpu_index}"
        hwmon = dirs[0]
        ok, msg = await asyncio.to_thread(self._sudo_sysfs_write, f"{hwmon}/pwm1_enable", "1", sudo_password)
        if not ok:
            return False, msg
        ok, msg = await asyncio.to_thread(self._sudo_sysfs_write, f"{hwmon}/pwm1", str(pwm_value), sudo_password)
        return (True, f"GPU {gpu_index}: fans set to {speed_pct}%") if ok else (False, msg)

    async def set_fan_auto_sudo(self, gpu_index: int, sudo_password: str) -> tuple[bool, str]:
        import glob as _glob
        pattern = f"/sys/class/drm/card{gpu_index}/device/hwmon/hwmon*"
        dirs = _glob.glob(pattern)
        if not dirs:
            return False, f"hwmon path not found for card{gpu_index}"
        hwmon = dirs[0]
        ok, msg = await asyncio.to_thread(self._sudo_sysfs_write, f"{hwmon}/pwm1_enable", "2", sudo_password)
        return (True, f"GPU {gpu_index}: fans returned to automatic control") if ok else (False, msg)

    def _sudo_sysfs_write(self, path: str, value: str, sudo_password: str) -> tuple[bool, str]:
        import subprocess
        try:
            proc = subprocess.run(
                ["sudo", "-S", "sh", "-c", f"echo {value} > {path}"],
                input=f"{sudo_password}\n".encode(),
                capture_output=True,
                timeout=10,
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
        pass
