from __future__ import annotations

import json
import math
import os
import time
from argparse import ArgumentParser, Namespace
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from train import ensure_data_lists, fast_forward_scheduler, validate_label_map
from utils.checkpoint import checkpoint_path, load_checkpoint
from utils.dataset import (
    CARFAC_BACKEND_JAX,
    GoogleSpeechDataset,
    get_carfac_backend,
    get_feature_cache_path,
    get_feature_type,
    validate_feature_config,
)
from utils.loss import LabelSmoothingLoss
from utils.misc import (
    count_params,
    get_model,
    log,
    log_event,
    save_model,
    seed_everything,
)
from utils.opt import get_optimizer
from utils.precision import resolve_precision
from utils.scheduler import WarmUpLR, get_scheduler
from utils.trainer import (
    Criterion,
    _accumulation_group_size,
    get_schedulers_state_dict,
    validate_targets,
)
from utils.types import Config


def import_xla() -> tuple[Any, Any, Any]:
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TPU training requires PyTorch/XLA. On a TPU VM, install matching "
            "`torch` and `torch_xla[tpu]`, then run with `PJRT_DEVICE=TPU`."
        ) from exc
    return xm, pl, xmp


def xla_runtime() -> Any | None:
    try:
        import torch_xla.runtime as xr
    except ModuleNotFoundError:
        return None
    return xr


def xla_device(xm: Any) -> torch.device:
    try:
        import torch_xla
    except ModuleNotFoundError:
        return xm.xla_device()
    if hasattr(torch_xla, "device"):
        return torch_xla.device()
    return xm.xla_device()


def xla_rank(xm: Any) -> int:
    xr = xla_runtime()
    if xr is not None and hasattr(xr, "global_ordinal"):
        return int(xr.global_ordinal())
    if hasattr(xm, "get_ordinal"):
        return int(xm.get_ordinal())
    return 0


def xla_world_size(xm: Any) -> int:
    xr = xla_runtime()
    if xr is not None and hasattr(xr, "world_size"):
        return int(xr.world_size())
    if hasattr(xm, "xrt_world_size"):
        return int(xm.xrt_world_size())
    return 1


def load_config(config_file: str, precision: str | None = None) -> Config:
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["exp"]["device"] = "xla"
    config["hparams"]["device"] = "xla"
    config["hparams"]["precision"] = resolve_precision(
        precision or config["hparams"].get("precision", "bfloat16")
    )
    return config


def is_master(xm: Any) -> bool:
    return xla_rank(xm) == 0


def master_log_event(message: str, config: Config, xm: Any) -> None:
    if is_master(xm):
        log_event(message, config)


def master_log(log_dict: dict[str, Any], step: int, config: Config, xm: Any) -> None:
    if is_master(xm):
        log(log_dict, step, config)


def read_split(path: str) -> list[str]:
    with open(path, "r") as f:
        return [line for line in f.read().rstrip().split("\n") if line]


def validate_precomputed_feature_cache(
    config: Config,
    args: Namespace,
    splits: dict[str, list[str]],
) -> None:
    audio_settings = config["hparams"]["audio"]
    feature_cache_dir = (
        config["exp"].get("feature_cache_dir")
        if config["exp"].get("feature_cache", False)
        else None
    )
    if (
        feature_cache_dir is None
        or get_feature_type(audio_settings) != "carfac"
        or get_carfac_backend(audio_settings) != CARFAC_BACKEND_JAX
    ):
        return

    missing_count = 0
    missing_examples: list[str] = []
    total = 0
    for data_list in splits.values():
        for path in data_list:
            total += 1
            cache_path = get_feature_cache_path(path, audio_settings, feature_cache_dir)
            if cache_path.exists():
                continue
            missing_count += 1
            if len(missing_examples) < 5:
                missing_examples.append(path)

    if not missing_count:
        return

    command = (
        "PJRT_DEVICE=TPU uv run python carfac/prepare_features.py "
        f"--data-root {config['data_root']} "
        f"--out-dir {feature_cache_dir} "
        f"--config {args.conf}"
    )
    examples = "\n".join(f"  - {path}" for path in missing_examples)
    raise RuntimeError(
        "JAX CARFAC training requires precomputed feature-cache files. "
        f"Missing {missing_count}/{total} cache entries under {feature_cache_dir}.\n"
        f"Build them first with:\n  {command}\n"
        f"Example missing source files:\n{examples}"
    )


def make_loader(
    data_list: list[str],
    config: Config,
    train: bool,
    rank: int,
    world_size: int,
    split_name: str,
) -> tuple[DataLoader, DistributedSampler]:
    audio_settings = config["hparams"]["audio"]
    feature_cache_dir = (
        config["exp"].get("feature_cache_dir")
        if config["exp"].get("feature_cache", False)
        else None
    )
    with open(config["label_map"], "r") as f:
        label_map = json.load(f)

    dataset = GoogleSpeechDataset(
        data_list=data_list,
        label_map=label_map,
        audio_settings=audio_settings,
        aug_settings=config["hparams"]["augment"] if train else None,
        cache=config["exp"]["cache"],
        feature_cache_dir=feature_cache_dir,
        split_name=split_name,
        config=config if rank == 0 else None,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=train,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["hparams"]["batch_size"],
        sampler=sampler,
        num_workers=config["exp"]["n_workers"],
        pin_memory=False,
        drop_last=False,
    )
    return loader, sampler


def reduce_sum(value: float | int, device: torch.device, xm: Any) -> float:
    if xla_world_size(xm) == 1:
        return float(value)
    tensor = torch.tensor(float(value), device=device)
    reduced = xm.all_reduce(xm.REDUCE_SUM, tensor)
    return float(reduced.item())


def reduce_mean(value: float, device: torch.device, xm: Any) -> float:
    world_size = xla_world_size(xm)
    if world_size == 1:
        return float(value)
    tensor = torch.tensor(float(value), device=device)
    reduced = xm.all_reduce(xm.REDUCE_SUM, tensor)
    return float((reduced / world_size).item())


def xla_autocast(device: torch.device, precision: str) -> Any:
    if precision == "float32":
        from contextlib import nullcontext

        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def train_one_batch(
    model: nn.Module,
    data: torch.Tensor,
    targets: torch.Tensor,
    criterion: Criterion,
    device: torch.device,
    loss_scale: float,
    num_classes: int,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_targets(targets, num_classes)
    with xla_autocast(device, precision):
        outputs = model(data)
        loss = criterion(outputs, targets)
    (loss / loss_scale).backward()
    correct = outputs.argmax(1).eq(targets).sum()
    count = torch.tensor(targets.numel(), device=device, dtype=torch.float32)
    return loss.detach(), correct.detach().to(torch.float32), count


@torch.no_grad()
def evaluate_xla(
    model: nn.Module,
    criterion: Criterion,
    loader: DataLoader,
    device_loader: Any,
    device: torch.device,
    config: Config,
    xm: Any,
) -> tuple[float, float]:
    model.eval()
    precision = str(config["hparams"]["precision"])
    num_classes = int(config["hparams"]["model"]["num_classes"])
    local_correct = 0
    local_count = 0
    local_loss_sum = 0.0
    local_batches = 0

    for data, targets in tqdm(device_loader, disable=not is_master(xm)):
        validate_targets(targets, num_classes)
        with xla_autocast(device, precision):
            outputs = model(data)
            loss = criterion(outputs, targets)
        local_correct += int(outputs.argmax(1).eq(targets).sum().item())
        local_count += int(targets.numel())
        local_loss_sum += float(loss.item())
        local_batches += 1

    correct = reduce_sum(local_correct, device, xm)
    count = reduce_sum(local_count, device, xm)
    loss_sum = reduce_sum(local_loss_sum, device, xm)
    batches = reduce_sum(local_batches, device, xm)
    model.train()
    return correct / max(count, 1.0), loss_sum / max(batches, 1.0)


def build_schedulers(
    optimizer: torch.optim.Optimizer,
    trainloader: DataLoader,
    config: Config,
) -> tuple[dict[str, WarmUpLR | Any | None], int, int]:
    schedulers: dict[str, WarmUpLR | Any | None] = {"warmup": None, "scheduler": None}
    scheduler_config = config["hparams"]["scheduler"]
    scheduler_type = scheduler_config["scheduler_type"]
    grad_accum_steps = int(config["hparams"].get("grad_accum_steps", 1))
    optimizer_steps_per_epoch = math.ceil(len(trainloader) / grad_accum_steps)

    if scheduler_config["n_warmup"] and scheduler_type != "one_cycle_lr":
        schedulers["warmup"] = WarmUpLR(
            optimizer,
            total_iters=optimizer_steps_per_epoch * scheduler_config["n_warmup"],
        )

    total_iters = 0
    if scheduler_type is not None:
        if scheduler_type == "one_cycle_lr":
            total_iters = optimizer_steps_per_epoch * scheduler_config["max_epochs"]
        else:
            total_iters = optimizer_steps_per_epoch * max(
                1,
                scheduler_config["max_epochs"] - scheduler_config["n_warmup"],
            )
        schedulers["scheduler"] = get_scheduler(
            optimizer, scheduler_config, total_iters
        )

    return schedulers, optimizer_steps_per_epoch, total_iters


def save_xla_checkpoint(
    name: str,
    epoch: int,
    val_acc: float,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    schedulers: dict[str, WarmUpLR | Any | None],
    step: int,
    best_acc: float,
    config: Config,
    xm: Any,
) -> None:
    if not is_master(xm):
        return
    save_model(
        epoch,
        val_acc,
        checkpoint_path(config["exp"]["save_dir"], name),
        model,
        optimizer,
        scheduler_state_dict=get_schedulers_state_dict(schedulers),
        step=step,
        best_acc=best_acc,
        log_file=os.path.join(config["exp"]["save_dir"], "training_log.txt"),
    )


def rendezvous_if_distributed(tag: str, xm: Any) -> None:
    if xla_world_size(xm) > 1:
        xm.rendezvous(tag)


def train_xla_worker(index: int, args: Namespace, base_config: Config) -> None:
    xm, pl, _ = import_xla()
    device = xla_device(xm)
    rank = xla_rank(xm)
    world_size = xla_world_size(xm)

    config = dict(base_config)
    config["exp"] = dict(base_config["exp"])
    config["hparams"] = dict(base_config["hparams"])
    config["hparams"]["device"] = device
    seed_everything(int(config["hparams"]["seed"]) + rank)

    if is_master(xm):
        config["exp"]["save_dir"] = os.path.join(
            config["exp"]["exp_dir"], config["exp"]["exp_name"]
        )
        os.makedirs(config["exp"]["save_dir"], exist_ok=True)
        with open(os.path.join(config["exp"]["save_dir"], "settings.txt"), "w+") as f:
            f.write(yaml.dump(config))
        log_event(f"XLA run directory: {config['exp']['save_dir']}", config)
    rendezvous_if_distributed("save_dir_ready", xm)
    if "save_dir" not in config["exp"]:
        config["exp"]["save_dir"] = os.path.join(
            config["exp"]["exp_dir"], config["exp"]["exp_name"]
        )
    trackio_active = False
    if is_master(xm) and config["exp"].get("trackio", False):
        import trackio

        trackio.init(
            project=config["exp"]["proj_name"],
            name=config["exp"]["exp_name"],
            space_id=config["exp"].get("trackio_space_id"),
            server_url=config["exp"].get("trackio_server_url"),
            config=config["hparams"],
        )
        trackio_active = True

    train_list = read_split(config["train_list_file"])
    val_list = read_split(config["val_list_file"])
    test_list = read_split(config["test_list_file"])
    validate_precomputed_feature_cache(
        config,
        args,
        {"train": train_list, "val": val_list, "test": test_list},
    )
    trainloader, train_sampler = make_loader(
        train_list, config, True, rank, world_size, "train"
    )
    valloader, val_sampler = make_loader(
        val_list, config, False, rank, world_size, "val"
    )
    train_device_loader = pl.MpDeviceLoader(trainloader, device)
    val_device_loader = pl.MpDeviceLoader(valloader, device)

    model = get_model(config["hparams"]["model"]).to(device)
    master_log_event(
        f"Created XLA model with {count_params(model)} params on {device}; "
        f"world_size={world_size}, train_batches={len(trainloader)}, "
        f"val_batches={len(valloader)}.",
        config,
        xm,
    )

    if config["hparams"]["l_smooth"]:
        criterion: Criterion = LabelSmoothingLoss(
            num_classes=config["hparams"]["model"]["num_classes"],
            smoothing=config["hparams"]["l_smooth"],
        )
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    optimizer = get_optimizer(model, config["hparams"]["optimizer"])
    schedulers, optimizer_steps_per_epoch, total_iters = build_schedulers(
        optimizer, trainloader, config
    )
    start_epoch = 0
    start_step = 0
    best_acc = 0.0
    best_saved = False
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        start_step = int(checkpoint.get("step", 0))
        best_acc = float(checkpoint.get("best_acc", checkpoint.get("val_acc", 0.0)))
        best_saved = os.path.exists(checkpoint_path(config["exp"]["save_dir"], "best"))
        if checkpoint.get("scheduler_state_dict") is None:
            completed_steps = start_epoch * optimizer_steps_per_epoch
            fast_forward_scheduler(
                schedulers["scheduler"], completed_steps, total_iters
            )
        else:
            for name, state in checkpoint["scheduler_state_dict"].items():
                if schedulers.get(name) is not None and state is not None:
                    schedulers[name].load_state_dict(state)
        xm.mark_step()

    grad_accum_steps = int(config["hparams"].get("grad_accum_steps", 1))
    precision = str(config["hparams"]["precision"])
    num_classes = int(config["hparams"]["model"]["num_classes"])
    step = start_step
    model.train()
    optimizer.zero_grad(set_to_none=True)
    master_log_event(
        f"XLA training started: epochs={config['hparams']['n_epochs']}, "
        f"precision={precision}, grad_accum_steps={grad_accum_steps}.",
        config,
        xm,
    )

    for epoch in range(start_epoch, config["hparams"]["n_epochs"]):
        t0 = time.perf_counter()
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        local_loss = torch.zeros((), device=device)
        local_correct = torch.zeros((), device=device)
        local_count = torch.zeros((), device=device)
        n_batches = len(trainloader)

        for batch_index, (data, targets) in enumerate(train_device_loader):
            active_scheduler = None
            if (
                schedulers["warmup"] is not None
                and epoch < config["hparams"]["scheduler"]["n_warmup"]
            ):
                active_scheduler = schedulers["warmup"]
            elif schedulers["scheduler"] is not None:
                active_scheduler = schedulers["scheduler"]

            loss_tensor, correct_tensor, count_tensor = train_one_batch(
                model,
                data,
                targets,
                criterion,
                device,
                loss_scale=_accumulation_group_size(
                    batch_index, n_batches, grad_accum_steps
                ),
                num_classes=num_classes,
                precision=precision,
            )
            local_loss = local_loss + loss_tensor
            local_correct = local_correct + correct_tensor
            local_count = local_count + count_tensor

            should_step_optimizer = (batch_index + 1) % grad_accum_steps == 0 or (
                batch_index + 1
            ) == n_batches
            if should_step_optimizer:
                xm.optimizer_step(optimizer, barrier=True)
                optimizer.zero_grad(set_to_none=True)
                if active_scheduler is not None:
                    active_scheduler.step()
            else:
                xm.mark_step()

            if step % config["exp"]["log_freq"] == 0:
                master_log(
                    {
                        "epoch": epoch,
                        "loss": float(loss_tensor.item()),
                        "grad_accum_steps": grad_accum_steps,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step,
                    config,
                    xm,
                )
            step += 1

        train_correct = reduce_sum(float(local_correct.item()), device, xm)
        train_count = reduce_sum(float(local_count.item()), device, xm)
        train_loss = reduce_mean(
            float((local_loss / max(n_batches, 1)).item()), device, xm
        )
        master_log(
            {
                "epoch": epoch,
                "time_per_epoch": time.perf_counter() - t0,
                "train_acc": train_correct / max(train_count, 1.0),
                "avg_loss_per_ep": train_loss,
            },
            step,
            config,
            xm,
        )

        if epoch % config["exp"]["val_freq"] == 0:
            val_acc, val_loss = evaluate_xla(
                model, criterion, valloader, val_device_loader, device, config, xm
            )
            master_log(
                {"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc},
                step,
                config,
                xm,
            )
            if val_acc > best_acc or not best_saved:
                best_acc = val_acc
                best_saved = True
                save_xla_checkpoint(
                    "best",
                    epoch,
                    val_acc,
                    model,
                    optimizer,
                    schedulers,
                    step,
                    best_acc,
                    config,
                    xm,
                )

    val_acc, val_loss = evaluate_xla(
        model, criterion, valloader, val_device_loader, device, config, xm
    )
    master_log(
        {
            "epoch": config["hparams"]["n_epochs"] - 1,
            "val_loss": val_loss,
            "val_acc": val_acc,
        },
        step,
        config,
        xm,
    )
    save_xla_checkpoint(
        "last",
        config["hparams"]["n_epochs"] - 1,
        val_acc,
        model,
        optimizer,
        schedulers,
        step,
        best_acc,
        config,
        xm,
    )
    rendezvous_if_distributed("final_last_saved", xm)

    testloader, test_sampler = make_loader(
        test_list, config, False, rank, world_size, "test"
    )
    test_sampler.set_epoch(0)
    test_device_loader = pl.MpDeviceLoader(testloader, device)

    test_acc, test_loss = evaluate_xla(
        model, criterion, testloader, test_device_loader, device, config, xm
    )
    master_log(
        {"test_loss_last": test_loss, "test_acc_last": test_acc},
        step,
        config,
        xm,
    )

    best_path = checkpoint_path(config["exp"]["save_dir"], "best")
    if os.path.exists(best_path):
        best_checkpoint = load_checkpoint(best_path, map_location="cpu")
        model.load_state_dict(best_checkpoint["model_state_dict"])
        xm.mark_step()
        testloader, test_sampler = make_loader(
            test_list, config, False, rank, world_size, "test"
        )
        test_sampler.set_epoch(0)
        test_device_loader = pl.MpDeviceLoader(testloader, device)
        test_acc, test_loss = evaluate_xla(
            model, criterion, testloader, test_device_loader, device, config, xm
        )
        master_log(
            {"test_loss_best": test_loss, "test_acc_best": test_acc},
            step,
            config,
            xm,
        )
    else:
        master_log_event("No best checkpoint was available for final test.", config, xm)

    if trackio_active:
        import trackio

        trackio.finish()


def main(args: Namespace) -> None:
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    config = load_config(args.conf, args.precision)
    ensure_data_lists(config)
    validate_label_map(config)
    validate_feature_config(config)
    if args.nprocs == 1:
        train_xla_worker(0, args, config)
        return

    _, _, xmp = import_xla()
    xmp.spawn(train_xla_worker, args=(args, config), nprocs=args.nprocs)


if __name__ == "__main__":
    parser = ArgumentParser("XLA TPU training driver.")
    parser.add_argument("--conf", type=str, required=True, help="Path to config YAML.")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a safetensors checkpoint to resume from.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Override precision. TPU should generally use bfloat16.",
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=1,
        help=(
            "Number of XLA worker processes. Use 1 for a single TPU chip. "
            "Set above 1 only when the TPU runtime exposes matching local workers."
        ),
    )
    main(parser.parse_args())
