"""Optional Helion kernels used by the KWT model.

Helion currently targets Linux GPU environments with Triton. The kernels in this
module are guarded so CPU/MPS runs keep using PyTorch implementations.
"""

import importlib.util
import math
import os
import platform
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

SQRT_2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

try:
    import helion
    import helion.language as hl
except ImportError:  # pragma: no cover - depends on optional package availability.
    helion = None
    hl = None


def _can_compile_helion(x: torch.Tensor) -> bool:
    if helion is None or hl is None:
        return False
    if os.environ.get("HELION_INTERPRET") == "1":
        return True
    return (
        platform.system() == "Linux"
        and x.is_cuda
        and importlib.util.find_spec("triton") is not None
    )


if helion is not None and hl is not None:

    @helion.kernel(autotune_effort="none", static_shapes=False)
    def _helion_gelu_forward(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.numel()):
            vals = x[tile]
            out[tile] = vals * 0.5 * (1.0 + torch.erf(vals / SQRT_2))
        return out

    @helion.kernel(autotune_effort="none", static_shapes=False)
    def _helion_gelu_backward(
        grad_output: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        grad_input = torch.empty_like(x)
        for tile in hl.tile(x.numel()):
            vals = x[tile]
            cdf = 0.5 * (1.0 + torch.erf(vals / SQRT_2))
            pdf_term = vals * INV_SQRT_2PI * torch.exp(-0.5 * vals * vals)
            grad_input[tile] = grad_output[tile] * (cdf + pdf_term)
        return grad_input

else:
    _helion_gelu_forward = None
    _helion_gelu_backward = None


class _HelionGELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        if _can_compile_helion(x) and _helion_gelu_forward is not None:
            return _helion_gelu_forward(x)
        return F.gelu(x)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor]:
        grad_output = grad_outputs[0]
        (x,) = ctx.saved_tensors
        if _can_compile_helion(x) and _helion_gelu_backward is not None:
            return (_helion_gelu_backward(grad_output, x),)

        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            y = F.gelu(x)
        return (torch.autograd.grad(y, x, grad_output)[0],)


class HelionGELU(nn.Module):
    def __init__(self, enabled: bool = False):
        super().__init__()
        self.enabled = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return F.gelu(x)
        return _HelionGELUFunction.apply(x)
