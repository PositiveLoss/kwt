import os
import time
from collections.abc import Callable, Sized
from typing import TypeAlias, cast

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.misc import log, save_model
from utils.scheduler import WarmUpLR
from utils.types import Config

Criterion: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Schedulers: TypeAlias = dict[str, WarmUpLR | optim.lr_scheduler.LRScheduler | None]


def train_single_batch(
    net: nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor,
    criterion: Criterion,
    device: torch.device,
    loss_scale: float = 1.0,
) -> tuple[float, int]:
    """Performs forward/backward for one microbatch.

    Args:
        net (nn.Module): Model instance.
        data (torch.Tensor): Data tensor, of shape (batch_size, dim_1, ... , dim_N).
        targets (torch.Tensor): Target tensor, of shape (batch_size).
        criterion (Callable): Loss function.
        device (torch.device): Device.
        loss_scale (float): Divisor for gradient accumulation.

    Returns:
        float: Loss scalar.
        int: Number of correct preds.
    """

    data, targets = data.to(device), targets.to(device)

    outputs = net(data)
    loss = criterion(outputs, targets)
    (loss / loss_scale).backward()

    correct = outputs.argmax(1).eq(targets).sum()
    return loss.item(), correct.item()


def _accumulation_group_size(
    batch_index: int, n_batches: int, grad_accum_steps: int
) -> int:
    group_start = (batch_index // grad_accum_steps) * grad_accum_steps
    group_end = min(group_start + grad_accum_steps, n_batches)
    return group_end - group_start


@torch.no_grad()
def evaluate(
    net: nn.Module,
    criterion: Criterion,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Performs inference.

    Args:
        net (nn.Module): Model instance.
        criterion (Callable): Loss function.
        dataloader (DataLoader): Test or validation loader.
        device (torch.device): Device.

    Returns:
        accuracy (float): Accuracy.
        float: Loss scalar.
    """

    net.eval()
    correct = 0
    running_loss = 0.0

    for data, targets in tqdm(dataloader):
        data, targets = data.to(device), targets.to(device)
        out = net(data)
        correct += out.argmax(1).eq(targets).sum().item()
        loss = criterion(out, targets)
        running_loss += loss.item()

    net.train()
    dataset_size = len(cast(Sized, dataloader.dataset))
    accuracy = correct / dataset_size
    return accuracy, running_loss / len(dataloader)


def train(
    net: nn.Module,
    optimizer: optim.Optimizer,
    criterion: Criterion,
    trainloader: DataLoader,
    valloader: DataLoader,
    schedulers: Schedulers,
    config: Config,
) -> None:
    """Trains model.

    Args:
        net (nn.Module): Model instance.
        optimizer (optim.Optimizer): Optimizer instance.
        criterion (Callable): Loss function.
        trainloader (DataLoader): Training data loader.
        valloader (DataLoader): Validation data loader.
        schedulers (dict): Dict containing schedulers.
        config (dict): Config dict.
    """

    step = 0
    best_acc = 0.0
    device = config["hparams"]["device"]
    log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")
    grad_accum_steps = int(config["hparams"].get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("hparams.grad_accum_steps must be >= 1.")

    ############################
    # start training
    ############################
    net.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(config["hparams"]["n_epochs"]):
        t0 = time.time()
        running_loss = 0.0
        correct = 0
        n_batches = len(trainloader)

        for batch_index, (data, targets) in enumerate(trainloader):
            active_scheduler = None
            if (
                schedulers["warmup"] is not None
                and epoch < config["hparams"]["scheduler"]["n_warmup"]
            ):
                active_scheduler = schedulers["warmup"]

            elif schedulers["scheduler"] is not None:
                active_scheduler = schedulers["scheduler"]

            ####################
            # optimization step
            ####################

            loss, corr = train_single_batch(
                net,
                data,
                targets,
                criterion,
                device,
                loss_scale=_accumulation_group_size(
                    batch_index, n_batches, grad_accum_steps
                ),
            )
            running_loss += loss
            correct += corr

            should_step_optimizer = (batch_index + 1) % grad_accum_steps == 0 or (
                batch_index + 1
            ) == n_batches
            if should_step_optimizer:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if active_scheduler is not None:
                    active_scheduler.step()

            if not step % config["exp"]["log_freq"]:
                log_dict = {
                    "epoch": epoch,
                    "loss": loss,
                    "grad_accum_steps": grad_accum_steps,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                log(log_dict, step, config)

            step += 1

        #######################
        # epoch complete
        #######################

        log_dict = {
            "epoch": epoch,
            "time_per_epoch": time.time() - t0,
            "train_acc": correct / len(cast(Sized, trainloader.dataset)),
            "avg_loss_per_ep": running_loss / len(trainloader),
        }
        log(log_dict, step, config)

        if not epoch % config["exp"]["val_freq"]:
            val_acc, avg_val_loss = evaluate(net, criterion, valloader, device)
            log_dict = {"epoch": epoch, "val_loss": avg_val_loss, "val_acc": val_acc}
            log(log_dict, step, config)

            # save best val ckpt
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(config["exp"]["save_dir"], "best.pth")
                save_model(epoch, val_acc, save_path, net, optimizer, log_file)

    ###########################
    # training complete
    ###########################

    val_acc, avg_val_loss = evaluate(net, criterion, valloader, device)
    log_dict = {"epoch": epoch, "val_loss": avg_val_loss, "val_acc": val_acc}
    log(log_dict, step, config)

    # save final ckpt
    save_path = os.path.join(config["exp"]["save_dir"], "last.pth")
    save_model(epoch, val_acc, save_path, net, optimizer, log_file)
