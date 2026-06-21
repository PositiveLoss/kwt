from typing import Any

from torch import nn, optim


def _parameters_are_cuda(net: nn.Module) -> bool:
    return any(parameter.is_cuda for parameter in net.parameters())


def get_optimizer(net: nn.Module, opt_config: dict[str, Any]) -> optim.Optimizer:
    """Creates optimizer based on config.

    Args:
        net (nn.Module): Model instance.
        opt_config (dict): Dict containing optimizer settings.

    Raises:
        ValueError: Unsupported optimizer type.

    Returns:
        optim.Optimizer: Optimizer instance.
    """

    opt_type = opt_config["opt_type"]
    opt_kwargs = dict(opt_config["opt_kwargs"])

    if opt_type == "adamw":
        optimizer = optim.AdamW(net.parameters(), **opt_kwargs)
    elif opt_type == "adamw_fused":
        if _parameters_are_cuda(net):
            opt_kwargs.setdefault("fused", True)
        optimizer = optim.AdamW(net.parameters(), **opt_kwargs)
    elif opt_type == "radam":
        optimizer = optim.RAdam(net.parameters(), **opt_kwargs)
    else:
        raise ValueError(f"Unsupported optimizer {opt_type}")

    return optimizer
