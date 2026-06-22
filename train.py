import math
import os
import json
import time
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

    if os.path.exists(train_file) and os.path.exists(label_map_file):
        return

    if not os.path.exists(val_file) or not os.path.exists(test_file):
        raise FileNotFoundError(
            "Missing split files. Expected validation/test split files at "
            f"{val_file!r} and {test_file!r}."
        )

    print("Missing derived training split or label map; generating data lists.")
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


def load_schedulers_state(
    schedulers: dict[str, WarmUpLR | Any | None],
    scheduler_state_dict: dict[str, Any] | None,
) -> None:
    if scheduler_state_dict is None:
        return

    for name, state in scheduler_state_dict.items():
        scheduler = schedulers.get(name)
        if scheduler is not None and state is not None:
            scheduler.load_state_dict(state)


def training_pipeline(
    config: Config, fine_tune: bool = False, resume_path: str | None = None
) -> None:
    """Initiates and executes all the steps involved with model training.

    Args:
        config (dict) - Dict containing various settings for the training run.
    """
    ensure_data_lists(config)

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
        load_schedulers_state(schedulers, resume_ckpt.get("scheduler_state_dict"))
        if resume_ckpt.get("scheduler_state_dict") is not None:
            log_event("Restored scheduler state from resume checkpoint.", config)
        else:
            log_event(
                "Resume checkpoint has no scheduler state; continuing with newly "
                "initialized schedulers.",
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
        model, criterion, testloader, config["hparams"]["device"]
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
        model, criterion, testloader, config["hparams"]["device"]
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

    config = get_config(args.conf, device=args.device)
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
    args = parser.parse_args()

    main(args)
