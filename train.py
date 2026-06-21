import math
import os
from argparse import ArgumentParser, Namespace
from collections.abc import Callable

import trackio
import torch
import yaml
from config_parser import get_config
from torch import nn

from utils.dataset import get_loader, warm_loader_cache
from utils.loss import LabelSmoothingLoss
from utils.misc import calc_step, count_params, get_model, log, seed_everything
from utils.opt import get_optimizer
from utils.scheduler import WarmUpLR, get_scheduler
from utils.trainer import evaluate, train
from utils.types import Config


def training_pipeline(config: Config, fine_tune: bool = False) -> None:
    """Initiates and executes all the steps involved with model training.

    Args:
        config (dict) - Dict containing various settings for the training run.
    """
    config["exp"]["save_dir"] = os.path.join(
        config["exp"]["exp_dir"], config["exp"]["exp_name"]
    )
    os.makedirs(config["exp"]["save_dir"], exist_ok=True)

    ######################################
    # save hyperparameters for current run
    ######################################

    config_str = yaml.dump(config)
    print("Using settings:\n", config_str)

    with open(os.path.join(config["exp"]["save_dir"], "settings.txt"), "w+") as f:
        f.write(config_str)

    #####################################
    # initialize training items
    #####################################

    # data
    with open(config["train_list_file"], "r") as f:
        train_list = f.read().rstrip().split("\n")

    with open(config["val_list_file"], "r") as f:
        val_list = f.read().rstrip().split("\n")

    trainloader = get_loader(train_list, config, train=True)
    valloader = get_loader(val_list, config, train=False)

    if config["exp"].get("warm_cache", False):
        warm_loader_cache(trainloader, "train")
        warm_loader_cache(valloader, "val")

    # model
    model = get_model(config["hparams"]["model"])

    if fine_tune:
        model = get_model({"name": "kwt-1"})
        ckpt = torch.load(config["ckpt_path"], map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.mlp_head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 3))

        print("Loaded model from checkpoint.")

    model = model.to(config["hparams"]["device"])

    print(f"Created model with {count_params(model)} parameters.")

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

    # lr scheduler
    schedulers = {"warmup": None, "scheduler": None}
    grad_accum_steps = int(config["hparams"].get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("hparams.grad_accum_steps must be >= 1.")
    optimizer_steps_per_epoch = math.ceil(len(trainloader) / grad_accum_steps)

    if config["hparams"]["scheduler"]["n_warmup"]:
        schedulers["warmup"] = WarmUpLR(
            optimizer,
            total_iters=optimizer_steps_per_epoch
            * config["hparams"]["scheduler"]["n_warmup"],
        )

    if config["hparams"]["scheduler"]["scheduler_type"] is not None:
        total_iters = optimizer_steps_per_epoch * max(
            1,
            (
                config["hparams"]["scheduler"]["max_epochs"]
                - config["hparams"]["scheduler"]["n_warmup"]
            ),
        )
        schedulers["scheduler"] = get_scheduler(
            optimizer, config["hparams"]["scheduler"]["scheduler_type"], total_iters
        )

    #####################################
    # Training Run
    #####################################

    print("Initiating training.")
    train(model, optimizer, criterion, trainloader, valloader, schedulers, config)

    #####################################
    # Final Test
    #####################################

    with open(config["test_list_file"], "r") as f:
        test_list = f.read().rstrip().split("\n")

    testloader = get_loader(test_list, config, train=False)
    if config["exp"].get("warm_cache", False):
        warm_loader_cache(testloader, "test")

    final_step = calc_step(
        config["hparams"]["n_epochs"] + 1, len(trainloader), len(trainloader) - 1
    )

    # evaluating the final state (last.pth)
    test_acc, test_loss = evaluate(
        model, criterion, testloader, config["hparams"]["device"]
    )
    log_dict = {"test_loss_last": test_loss, "test_acc_last": test_acc}
    log(log_dict, final_step, config)

    # evaluating the best validation state (best.pth)
    ckpt = torch.load(
        os.path.join(config["exp"]["save_dir"], "best.pth"),
        map_location=config["hparams"]["device"],
        weights_only=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    print("Best ckpt loaded.")

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
    run_with_trackio(config, lambda: training_pipeline(config))


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
    args = parser.parse_args()

    main(args)
