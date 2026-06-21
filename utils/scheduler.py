from typing import cast
from typing import Any
from typing import Literal

from torch import optim
from torch.optim import lr_scheduler


class WarmUpLR(lr_scheduler._LRScheduler):
    """WarmUp learning rate scheduler.

    Args:
        optimizer (optim.Optimizer): Optimizer instance
        total_iters (int): steps_per_epoch * n_warmup_epochs
        last_epoch (int): Final epoch. Defaults to -1.
    """

    def __init__(
        self, optimizer: optim.Optimizer, total_iters: int, last_epoch: int = -1
    ):
        """Initializer for WarmUpLR"""

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:  # pyrefly: ignore[bad-override]
        """Learning rate will be set to base_lr * last_epoch / total_iters."""

        return [
            float(base_lr) * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]


def get_scheduler(
    optimizer: optim.Optimizer,
    scheduler_config: dict[str, Any],
    total_iters: int,
) -> lr_scheduler.LRScheduler:
    """Gets scheduler.

    Args:
        optimizer (optim.Optimizer): Optimizer instance.
        scheduler_config (dict): Scheduler settings.
        total_iters (int): Total optimizer steps.

    Raises:
        ValueError: Unsupported scheduler type.

    Returns:
        lr_scheduler._LRScheduler: Scheduler instance.
    """

    scheduler_type = scheduler_config["scheduler_type"]

    if scheduler_type == "cosine_annealing":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            total_iters,
            eta_min=float(scheduler_config.get("eta_min", 1e-8)),
        )
    elif scheduler_type == "one_cycle_lr":
        if total_iters < 20:
            raise ValueError(
                "one_cycle_lr requires at least 20 optimizer steps. "
                f"Got total_steps={total_iters}."
            )
        max_lr = scheduler_config.get("max_lr")
        if max_lr is None:
            max_lr = max(group["lr"] for group in optimizer.param_groups)
        anneal_strategy = scheduler_config.get("anneal_strategy", "cos")
        if anneal_strategy not in {"cos", "linear"}:
            raise ValueError("scheduler.anneal_strategy must be 'cos' or 'linear'.")
        anneal_strategy = cast(Literal["cos", "linear"], anneal_strategy)
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(max_lr),
            total_steps=total_iters,
            pct_start=float(scheduler_config.get("pct_start", 0.1)),
            anneal_strategy=anneal_strategy,
            div_factor=float(scheduler_config.get("div_factor", 25.0)),
            final_div_factor=float(scheduler_config.get("final_div_factor", 10000.0)),
            three_phase=bool(scheduler_config.get("three_phase", False)),
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return cast(lr_scheduler.LRScheduler, scheduler)
