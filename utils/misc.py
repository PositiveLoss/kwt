"""Miscellaneous helper functions."""

import os
import random
from datetime import datetime
from typing import Any

import numpy as np
import torch
import trackio
from models.kwt import KWT, kwt_from_name
from torch import nn, optim
from utils.checkpoint import save_checkpoint
from utils.types import Config


def seed_everything(seed: int) -> None:
    """Set manual seed.

    Args:
        seed (int): Supplied seed.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Set seed {seed}")


def count_params(model: nn.Module) -> int:
    """Counts number of parameters in a model.

    Args:
        model (torch.nn.Module): Model instance for which number of params is to be counted.

    Returns:
        int: Parameter count.
    """

    return sum(p.numel() for p in model.parameters())


def calc_step(epoch: int, n_batches: int, batch_index: int) -> int:
    """Calculates current step.

    Args:
        epoch (int): Current epoch.
        n_batches (int): Number of batches in dataloader.
        batch_index (int): Current batch index.

    Returns:
        int: Current step.
    """
    return (epoch - 1) * n_batches + (1 + batch_index)


def log(log_dict: dict[str, Any], step: int, config: Config) -> None:
    """Handles logging for metric tracking server, local disk and stdout.

    Args:
        log_dict (dict): Log metric dict.
        step (int): Current step.
        config (dict): Config dict.
    """

    if config["exp"].get("trackio", False):
        trackio.log(log_dict, step=step)

    log_message = f"Step: {step} | " + " | ".join(
        [f"{k}: {v}" for k, v in log_dict.items()]
    )

    # write logs to disk
    if config["exp"]["log_to_file"]:
        log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")

        with open(log_file, "a+") as f:
            f.write(log_message + "\n")

    # show logs in stdout
    if config["exp"]["log_to_stdout"]:
        print(log_message)


def log_event(
    message: str,
    config: Config | None = None,
    log_file: str | None = None,
) -> None:
    """Write a timestamped training event to stdout and the run log file."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"

    if config is None or config["exp"].get("log_to_stdout", True):
        print(log_message)

    target_log_file = log_file
    if (
        target_log_file is None
        and config is not None
        and config["exp"].get("log_to_file", False)
        and "save_dir" in config["exp"]
    ):
        target_log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")

    if target_log_file is not None:
        with open(target_log_file, "a+") as f:
            f.write(log_message + "\n")


def get_model(model_config: dict[str, Any]) -> nn.Module:
    """Creates model from config dict.

    Args:
        model_config (dict): Dict containing model config params. If the "name" key is not None, other params are ignored.

    Returns:
        nn.Module: Model instance.
    """

    if model_config["name"] is not None:
        return kwt_from_name(model_config["name"])
    else:
        return KWT(**model_config)


def save_model(
    epoch: int,
    val_acc: float,
    save_path: str,
    net: nn.Module,
    optimizer: optim.Optimizer | None = None,
    scheduler_state_dict: dict[str, Any] | None = None,
    step: int = 0,
    best_acc: float | None = None,
    log_file: str | None = None,
) -> None:
    """Saves checkpoint.

    Args:
        epoch (int): Current epoch.
        val_acc (float): Validation accuracy.
        save_path (str): Checkpoint path.
        net (nn.Module): Model instance.
        optimizer (optim.Optimizer, optional): Optimizer. Defaults to None.
        scheduler_state_dict (dict, optional): Scheduler states. Defaults to None.
        step (int): Global training step.
        best_acc (float, optional): Best validation accuracy so far.
        log_file (str, optional): Log file. Defaults to None.
    """

    save_checkpoint(
        save_path,
        epoch=epoch,
        val_acc=val_acc,
        model_state_dict=net.state_dict(),
        optimizer_state_dict=optimizer.state_dict() if optimizer is not None else None,
        scheduler_state_dict=scheduler_state_dict,
        step=step,
        best_acc=best_acc,
    )

    log_message = f"Saved {save_path} with accuracy {val_acc}."
    print(log_message)

    if log_file is not None:
        with open(log_file, "a+") as f:
            f.write(log_message + "\n")
