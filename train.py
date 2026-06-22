import math
import os
import json
import time
import warnings
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from typing import Any

import trackio
import yaml
from config_parser import get_config
from torch import nn

from utils.checkpoint import checkpoint_path, load_checkpoint
from utils.dataset import get_loader, get_train_val_test_split, warm_loader_cache
from utils.loss import LabelSmoothingLoss
from utils.misc import (
    calc_step,
    count_params,
    get_model,
    log,
    log_event,
    seed_everything,
)
from utils.opt import get_optimizer
from utils.scheduler import WarmUpLR, get_scheduler
from utils.trainer import evaluate, train
from utils.types import Config


def ensure_data_lists(config: Config) -> None:
    data_root = config["data_root"]
    train_file = config["train_list_file"]
    val_file = config["val_list_file"]
    test_file = config["test_list_file"]
    label_map_file = config["label_map"]

    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"Dataset root {data_root!r} does not exist. "
            f"Download Speech Commands V2 first with: ./download_gspeech_v2.sh {data_root}"
        )

    if not os.path.exists(val_file) or not os.path.exists(test_file):
        raise FileNotFoundError(
            "Missing split files. Expected validation/test split files at "
            f"{val_file!r} and {test_file!r}."
        )

    should_generate = not (
        os.path.exists(train_file) and os.path.exists(label_map_file)
    )
    if not should_generate:
        try:
            validate_label_map(config)
            return
        except ValueError as exc:
            print(f"Existing derived data lists are invalid: {exc}")
            should_generate = True

    if should_generate:
        print("Generating derived data lists and label map.")
    train_list, val_list, test_list, label_map = get_train_val_test_split(
        data_root, val_file, test_file
    )

    for path in (train_file, val_file, test_file, label_map_file):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(train_file, "w+") as f:
        f.write("\n".join(train_list))
    with open(val_file, "w+") as f:
        f.write("\n".join(val_list))
    with open(test_file, "w+") as f:
        f.write("\n".join(test_list))
    with open(label_map_file, "w+") as f:
        json.dump(label_map, f)


def validate_label_map(config: Config) -> None:
    label_map_file = config["label_map"]
    num_classes = int(config["hparams"]["model"]["num_classes"])

    with open(label_map_file, "r") as f:
        label_map = json.load(f)

    invalid_keys = [key for key in label_map if not str(key).isdigit()]
    if invalid_keys:
        raise ValueError(
            f"{label_map_file} must use integer-like string keys; "
            f"found invalid keys: {invalid_keys[:10]}."
        )

    label_indices = sorted(int(key) for key in label_map)
    if not label_indices:
        raise ValueError(f"{label_map_file} is empty.")

    out_of_range = [idx for idx in label_indices if idx < 0 or idx >= num_classes]
    if out_of_range:
        bad_labels = {str(idx): label_map[str(idx)] for idx in out_of_range[:10]}
        raise ValueError(
            f"Label map has indices outside model num_classes={num_classes}: "
            f"{bad_labels}. Set hparams.model.num_classes to at least "
            f"{max(label_indices) + 1}, or regenerate {label_map_file} from the "
            "intended dataset root."
        )

    expected = list(range(num_classes))
    if label_indices != expected:
        raise ValueError(
            f"Label map indices must be contiguous 0..{num_classes - 1}; "
            f"found min={label_indices[0]}, max={label_indices[-1]}, "
            f"count={len(label_indices)}."
        )


def load_schedulers_state(
    schedulers: dict[str, WarmUpLR | Any | None],
    scheduler_state_dict: dict[str, Any] | None,
    config: Config,
) -> None:
    if scheduler_state_dict is None:
        return

    for name, state in scheduler_state_dict.items():
        scheduler = schedulers.get(name)
        if scheduler is not None and state is not None:
            state = reconcile_scheduler_state(scheduler, state, name, config)
            scheduler.load_state_dict(state)


def reconcile_scheduler_state(
    scheduler: WarmUpLR | Any,
    state: dict[str, Any],
    name: str,
    config: Config,
) -> dict[str, Any]:
    current_state = scheduler.state_dict()
    if (
        config["hparams"]["scheduler"].get("scheduler_type") == "one_cycle_lr"
        and "total_steps" in state
        and "total_steps" in current_state
        and state["total_steps"] != current_state["total_steps"]
    ):
        state = dict(state)
        old_total_steps = state["total_steps"]
        state["total_steps"] = current_state["total_steps"]
        state["_schedule_phases"] = current_state["_schedule_phases"]
        log_event(
            f"Adjusted resumed {name} OneCycleLR total_steps from "
            f"{old_total_steps} to {state['total_steps']} to match current "
            "scheduler.max_epochs.",
            config,
        )
    return state


def infer_completed_optimizer_steps(
    start_epoch: int,
    start_step: int,
    optimizer_steps_per_epoch: int,
    grad_accum_steps: int,
) -> int:
    if start_step > 0:
        return math.ceil(start_step / grad_accum_steps)
    return start_epoch * optimizer_steps_per_epoch


def fast_forward_scheduler(
    scheduler: WarmUpLR | Any | None,
    completed_steps: int,
    total_iters: int,
) -> int:
    if scheduler is None or completed_steps <= 0:
        return 0

    steps_to_advance = min(completed_steps, max(0, total_iters - 1))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Detected call of `lr_scheduler.step\\(\\)` before "
            "`optimizer.step\\(\\)`",
        )
        for _ in range(steps_to_advance):
            scheduler.step()
    return steps_to_advance


def training_pipeline(
    config: Config, fine_tune: bool = False, resume_path: str | None = None
) -> None:
    """Initiates and executes all the steps involved with model training.

    Args:
        config (dict) - Dict containing various settings for the training run.
    """
    ensure_data_lists(config)
    validate_label_map(config)

    config["exp"]["save_dir"] = os.path.join(
        config["exp"]["exp_dir"], config["exp"]["exp_name"]
    )
    os.makedirs(config["exp"]["save_dir"], exist_ok=True)
    log_event(f"Run directory: {config['exp']['save_dir']}", config)

    ######################################
    # save hyperparameters for current run
    ######################################

    config_str = yaml.dump(config)
    print("Using settings:\n", config_str)

    with open(os.path.join(config["exp"]["save_dir"], "settings.txt"), "w+") as f:
        f.write(config_str)
    log_event("Saved resolved settings.", config)

    #####################################
    # initialize training items
    #####################################

    # data
    t0 = time.perf_counter()
    log_event("Loading train/validation file lists.", config)
    with open(config["train_list_file"], "r") as f:
        train_list = [line for line in f.read().rstrip().split("\n") if line]

    with open(config["val_list_file"], "r") as f:
        val_list = [line for line in f.read().rstrip().split("\n") if line]
    log_event(
        f"Loaded file lists: train={len(train_list)}, val={len(val_list)} "
        f"in {time.perf_counter() - t0:.2f}s.",
        config,
    )

    t0 = time.perf_counter()
    log_event("Building train dataloader.", config)
    trainloader = get_loader(train_list, config, train=True, split_name="train")
    log_event(
        f"Built train dataloader: batches={len(trainloader)} "
        f"in {time.perf_counter() - t0:.2f}s.",
        config,
    )

    t0 = time.perf_counter()
    log_event("Building validation dataloader.", config)
    valloader = get_loader(val_list, config, train=False, split_name="val")
    log_event(
        f"Built validation dataloader: batches={len(valloader)} "
        f"in {time.perf_counter() - t0:.2f}s.",
        config,
    )

    if config["exp"].get("warm_cache", False):
        log_event("Warming train and validation dataloaders.", config)
        warm_loader_cache(trainloader, "train")
        warm_loader_cache(valloader, "val")
        log_event("Finished dataloader warmup.", config)

    if fine_tune and resume_path is not None:
        raise ValueError("fine_tune and resume_path cannot be used together.")

    resume_ckpt: dict[str, Any] | None = None
    start_epoch = 0
    start_step = 0
    best_acc = 0.0
    if resume_path is not None:
        log_event(f"Loading resume checkpoint: {resume_path}", config)
        resume_ckpt = load_checkpoint(resume_path, map_location="cpu")
        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        start_step = int(resume_ckpt.get("step", 0))
        checkpoint_best_acc = resume_ckpt.get("best_acc")
        if checkpoint_best_acc is None:
            checkpoint_best_acc = resume_ckpt.get("val_acc", 0.0)
        best_acc = float(checkpoint_best_acc)
        log_event(
            f"Resume checkpoint loaded: checkpoint_epoch={resume_ckpt.get('epoch')}, "
            f"start_epoch={start_epoch}, step={start_step}, "
            f"best_acc={best_acc:.6f}.",
            config,
        )

    # model
    log_event("Creating model.", config)
    model = get_model(config["hparams"]["model"])

    if fine_tune:
        model = get_model({"name": "kwt-1"})
        ckpt = load_checkpoint(config["ckpt_path"], map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.mlp_head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 3))

        print("Loaded model from checkpoint.")

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"])
        log_event("Restored model state from resume checkpoint.", config)

    model = model.to(config["hparams"]["device"])

    log_event(
        f"Created model with {count_params(model)} parameters on "
        f"{config['hparams']['device']}.",
        config,
    )

    # loss
    if config["hparams"]["l_smooth"]:
        criterion = LabelSmoothingLoss(
            num_classes=config["hparams"]["model"]["num_classes"],
            smoothing=config["hparams"]["l_smooth"],
        )
    else:
        criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = get_optimizer(model, config["hparams"]["optimizer"])
    log_event(
        f"Created optimizer: {config['hparams']['optimizer']['opt_type']}.", config
    )
    if resume_ckpt is not None and resume_ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        log_event("Restored optimizer state from resume checkpoint.", config)

    # lr scheduler
    schedulers = {"warmup": None, "scheduler": None}
    scheduler_config = config["hparams"]["scheduler"]
    scheduler_type = scheduler_config["scheduler_type"]
    grad_accum_steps = int(config["hparams"].get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("hparams.grad_accum_steps must be >= 1.")
    optimizer_steps_per_epoch = math.ceil(len(trainloader) / grad_accum_steps)

    if scheduler_type == "one_cycle_lr" and scheduler_config.get("n_warmup", 0):
        log_event(
            "Ignoring scheduler.n_warmup because one_cycle_lr includes warmup "
            "via scheduler.pct_start.",
            config,
        )
    elif scheduler_config["n_warmup"]:
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
                (scheduler_config["max_epochs"] - scheduler_config["n_warmup"]),
            )
        schedulers["scheduler"] = get_scheduler(
            optimizer,
            scheduler_config,
            total_iters,
        )
    log_event(
        "Schedulers configured: "
        f"warmup={schedulers['warmup'] is not None}, "
        f"scheduler={scheduler_type}.",
        config,
    )
    if resume_ckpt is not None:
        load_schedulers_state(
            schedulers,
            resume_ckpt.get("scheduler_state_dict"),
            config,
        )
        if resume_ckpt.get("scheduler_state_dict") is not None:
            log_event("Restored scheduler state from resume checkpoint.", config)
        else:
            completed_steps = infer_completed_optimizer_steps(
                start_epoch,
                start_step,
                optimizer_steps_per_epoch,
                grad_accum_steps,
            )
            advanced_steps = fast_forward_scheduler(
                schedulers["scheduler"],
                completed_steps,
                total_iters,
            )
            log_event(
                "Resume checkpoint has no scheduler state; inferred "
                f"{completed_steps} completed optimizer steps and advanced scheduler "
                f"by {advanced_steps} steps.",
                config,
            )

    #####################################
    # Training Run
    #####################################

    log_event(
        f"Initiating training for {config['hparams']['n_epochs']} epochs "
        f"({len(trainloader)} batches/epoch).",
        config,
    )
    train(
        model,
        optimizer,
        criterion,
        trainloader,
        valloader,
        schedulers,
        config,
        start_epoch=start_epoch,
        start_step=start_step,
        best_acc=best_acc,
        amp_scaler_state_dict=(
            resume_ckpt.get("amp_scaler_state_dict")
            if resume_ckpt is not None
            else None
        ),
    )
    log_event("Training loop finished.", config)

    #####################################
    # Final Test
    #####################################

    with open(config["test_list_file"], "r") as f:
        test_list = [line for line in f.read().rstrip().split("\n") if line]

    t0 = time.perf_counter()
    log_event(f"Building test dataloader for {len(test_list)} files.", config)
    testloader = get_loader(test_list, config, train=False, split_name="test")
    log_event(
        f"Built test dataloader: batches={len(testloader)} "
        f"in {time.perf_counter() - t0:.2f}s.",
        config,
    )
    if config["exp"].get("warm_cache", False):
        warm_loader_cache(testloader, "test")

    final_step = calc_step(
        config["hparams"]["n_epochs"] + 1, len(trainloader), len(trainloader) - 1
    )

    # evaluating the final state (last.safetensors)
    log_event("Evaluating final model state on test split.", config)
    test_acc, test_loss = evaluate(
        model,
        criterion,
        testloader,
        config["hparams"]["device"],
        num_classes=config["hparams"]["model"]["num_classes"],
        precision=config["hparams"]["precision"],
    )
    log_dict = {"test_loss_last": test_loss, "test_acc_last": test_acc}
    log(log_dict, final_step, config)

    # evaluating the best validation state (best.safetensors)
    log_event("Loading best validation checkpoint for test evaluation.", config)
    ckpt = load_checkpoint(
        checkpoint_path(config["exp"]["save_dir"], "best"),
        map_location=config["hparams"]["device"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    log_event("Best checkpoint loaded.", config)

    log_event("Evaluating best validation checkpoint on test split.", config)
    test_acc, test_loss = evaluate(
        model,
        criterion,
        testloader,
        config["hparams"]["device"],
        num_classes=config["hparams"]["model"]["num_classes"],
        precision=config["hparams"]["precision"],
    )
    log_dict = {"test_loss_best": test_loss, "test_acc_best": test_acc}
    log(log_dict, final_step, config)


def run_with_trackio(config: Config, fn: Callable[[], None]) -> None:
    if not config["exp"].get("trackio", False):
        fn()
        return

    trackio.init(
        project=config["exp"]["proj_name"],
        name=config["exp"]["exp_name"],
        space_id=config["exp"].get("trackio_space_id"),
        server_url=config["exp"].get("trackio_server_url"),
        config=config["hparams"],
    )
    try:
        fn()
    finally:
        trackio.finish()


def main(args: Namespace) -> None:
    """
    Main function to initiate training.
    """

    config = get_config(args.conf, device=args.device, precision=args.precision)
    seed_everything(config["hparams"]["seed"])
    try:
        run_with_trackio(
            config, lambda: training_pipeline(config, resume_path=args.resume)
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument(
        "--conf", type=str, required=True, help="Path to config.yaml file."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override config device. One of auto, cpu, cuda, or mps.",
    )
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
        help=(
            "Override config precision. One of float32, fp32, bfloat16, "
            "bf16, float16, fp16, or half."
        ),
    )
    args = parser.parse_args()

    main(args)
