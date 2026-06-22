import os
import time
from collections.abc import Callable, Sized
from typing import TypeAlias, cast

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.checkpoint import checkpoint_path
from utils.misc import log, log_event, save_model
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
    num_classes: int | None = None,
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

    validate_targets(targets, num_classes)
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


def validate_targets(targets: torch.Tensor, num_classes: int | None) -> None:
    if num_classes is None or targets.numel() == 0:
        return

    min_target = int(targets.min().item())
    max_target = int(targets.max().item())
    if min_target < 0 or max_target >= num_classes:
        raise ValueError(
            "Target label index out of range: "
            f"min={min_target}, max={max_target}, num_classes={num_classes}. "
            "Check label_map and hparams.model.num_classes."
        )


def get_schedulers_state_dict(schedulers: Schedulers) -> dict[str, object]:
    return {
        name: scheduler.state_dict() if scheduler is not None else None
        for name, scheduler in schedulers.items()
    }


@torch.no_grad()
def evaluate(
    net: nn.Module,
    criterion: Criterion,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int | None = None,
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
    if num_classes is None:
        num_classes = int(getattr(criterion, "cls", 0)) or None

    for data, targets in tqdm(dataloader):
        validate_targets(targets, num_classes)
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
    start_epoch: int = 0,
    start_step: int = 0,
    best_acc: float = 0.0,
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

    step = start_step
    device = config["hparams"]["device"]
    log_file = os.path.join(config["exp"]["save_dir"], "training_log.txt")
    grad_accum_steps = int(config["hparams"].get("grad_accum_steps", 1))
    num_classes = int(config["hparams"]["model"]["num_classes"])
    if grad_accum_steps < 1:
        raise ValueError("hparams.grad_accum_steps must be >= 1.")

    ############################
    # start training
    ############################
    net.train()
    optimizer.zero_grad(set_to_none=True)
    log_event(
        f"Training loop started on {device}: epochs={config['hparams']['n_epochs']}, "
        f"train_batches={len(trainloader)}, val_batches={len(valloader)}, "
        f"grad_accum_steps={grad_accum_steps}, start_epoch={start_epoch}, "
        f"start_step={start_step}, best_acc={best_acc:.6f}.",
        config,
    )

    if start_epoch >= config["hparams"]["n_epochs"]:
        log_event(
            f"Skipping training because start_epoch={start_epoch} is >= "
            f"n_epochs={config['hparams']['n_epochs']}.",
            config,
        )
        return

    epoch = start_epoch
    for epoch in range(start_epoch, config["hparams"]["n_epochs"]):
        t0 = time.time()
        running_loss = 0.0
        correct = 0
        n_batches = len(trainloader)
        log_event(f"Epoch {epoch} started.", config)

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
                num_classes=num_classes,
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
        log_event(
            f"Epoch {epoch} finished in {log_dict['time_per_epoch']:.2f}s.",
            config,
        )

        if not epoch % config["exp"]["val_freq"]:
            log_event(f"Validation started for epoch {epoch}.", config)
            val_acc, avg_val_loss = evaluate(
                net, criterion, valloader, device, num_classes=num_classes
            )
            log_dict = {"epoch": epoch, "val_loss": avg_val_loss, "val_acc": val_acc}
            log(log_dict, step, config)
            log_event(
                f"Validation finished for epoch {epoch}: "
                f"val_loss={avg_val_loss:.6f}, val_acc={val_acc:.6f}.",
                config,
            )

            # save best val ckpt
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = checkpoint_path(config["exp"]["save_dir"], "best")
                log_event(
                    f"New best validation accuracy {best_acc:.6f}; saving checkpoint.",
                    config,
                )
                save_model(
                    epoch,
                    val_acc,
                    save_path,
                    net,
                    optimizer,
                    scheduler_state_dict=get_schedulers_state_dict(schedulers),
                    step=step,
                    best_acc=best_acc,
                    log_file=log_file,
                )

    ###########################
    # training complete
    ###########################

    log_event("Final validation started.", config)
    val_acc, avg_val_loss = evaluate(
        net, criterion, valloader, device, num_classes=num_classes
    )
    log_dict = {"epoch": epoch, "val_loss": avg_val_loss, "val_acc": val_acc}
    log(log_dict, step, config)
    log_event(
        f"Final validation finished: val_loss={avg_val_loss:.6f}, "
        f"val_acc={val_acc:.6f}.",
        config,
    )

    # save final ckpt
    save_path = checkpoint_path(config["exp"]["save_dir"], "last")
    save_model(
        epoch,
        val_acc,
        save_path,
        net,
        optimizer,
        scheduler_state_dict=get_schedulers_state_dict(schedulers),
        step=step,
        best_acc=best_acc,
        log_file=log_file,
    )
