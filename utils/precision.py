from contextlib import nullcontext
from typing import Any

import torch


PRECISION_ALIASES = {
    "float32": "float32",
    "fp32": "float32",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
}


def resolve_precision(precision: str | None) -> str:
    if precision is None:
        return "float32"

    normalized = precision.lower()
    if normalized not in PRECISION_ALIASES:
        valid = ", ".join(sorted(PRECISION_ALIASES))
        raise ValueError(f"Unsupported precision {precision!r}. Expected one of: {valid}.")

    return PRECISION_ALIASES[normalized]


def dtype_from_precision(precision: str) -> torch.dtype:
    precision = resolve_precision(precision)
    if precision == "bfloat16":
        return torch.bfloat16
    if precision == "float16":
        return torch.float16
    return torch.float32


def autocast_for_precision(device: torch.device, precision: str) -> Any:
    precision = resolve_precision(precision)
    if precision == "float32":
        return nullcontext()

    return torch.autocast(
        device_type=device.type,
        dtype=dtype_from_precision(precision),
    )
