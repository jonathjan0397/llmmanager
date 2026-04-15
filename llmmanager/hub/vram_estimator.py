"""VRAM requirement estimator — estimates from parameter count and quantization."""

from __future__ import annotations

# Bytes per parameter per quantization level
_BYTES_PER_PARAM: dict[str, float] = {
    "f32":    4.0,
    "f16":    2.0,
    "bf16":   2.0,
    "q8_0":   1.0,
    "q6_k":   0.75,
    "q5_k_m": 0.625,
    "q5_k_s": 0.625,
    "q5_0":   0.625,
    "q4_k_m": 0.5,
    "q4_k_s": 0.5,
    "q4_0":   0.5,
    "q3_k_m": 0.375,
    "q3_k_s": 0.375,
    "q2_k":   0.3125,
    "awq":    0.5,
    "gptq":   0.5,
    "fp8":    1.0,
}

# Overhead multiplier for KV cache, activations, etc.
_OVERHEAD_FACTOR = 1.2


def estimate_vram_mb(
    parameter_count_b: float,
    quantization: str | None,
) -> float:
    """
    Estimate VRAM requirement in MB.

    Args:
        parameter_count_b: Model size in billions of parameters.
        quantization: Quantization format string (case-insensitive). None = f16.

    Returns:
        Estimated VRAM in MB.
    """
    quant = (quantization or "f16").lower()
    # Normalize common aliases
    quant = quant.replace("-", "_").replace(" ", "_")
    bpp = _BYTES_PER_PARAM.get(quant, 2.0)  # default to f16

    params = parameter_count_b * 1e9
    bytes_needed = params * bpp * _OVERHEAD_FACTOR
    return bytes_needed / (1024 ** 2)


def fits_in_vram(
    parameter_count_b: float,
    quantization: str | None,
    available_vram_mb: float,
    headroom_pct: float = 10.0,
) -> tuple[bool, float]:
    """
    Returns (fits, estimated_mb) — True if the model fits with headroom.
    headroom_pct reserves that % of available_vram_mb as free margin.
    """
    effective_available = available_vram_mb * (1.0 - headroom_pct / 100.0)
    estimated = estimate_vram_mb(parameter_count_b, quantization)
    return estimated <= effective_available, estimated
