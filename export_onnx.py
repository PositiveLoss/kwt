"""Export a KWT model checkpoint to ONNX."""

import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, cast

import torch

from config_parser import get_config
from utils.checkpoint import load_checkpoint
from utils.misc import get_model
from utils.types import Config


def load_model_checkpoint(
    model: torch.nn.Module, ckpt_path: str, device: torch.device
) -> None:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])


def get_input_shape(config: Config, batch_size: int) -> tuple[int, int, int, int]:
    model_config = config["hparams"]["model"]
    input_res = model_config.get("input_res")
    if input_res is None:
        raise ValueError(
            "Model config must include input_res, or pass a config with explicit model dimensions."
        )

    channels = model_config.get("channels", 1)
    return batch_size, channels, input_res[0], input_res[1]


def slim_onnx_model(
    output_path: Path, input_shape: tuple[int, int, int, int], verify: bool
) -> None:
    import onnxslim

    input_shape_arg = f"input:{','.join(str(dim) for dim in input_shape)}"
    slim = cast(Any, onnxslim.slim)
    slim(
        str(output_path),
        str(output_path),
        model_check=verify,
        model_check_inputs=[input_shape_arg] if verify else None,
    )


def export_onnx(args: Any) -> None:
    config = get_config(args.conf, device=args.device)
    device = config["hparams"]["device"]

    model = get_model(config["hparams"]["model"]).to(device)
    model.eval()

    if args.ckpt is not None:
        load_model_checkpoint(model, args.ckpt, device)

    input_shape = get_input_shape(config, args.batch_size)
    example_input = torch.randn(input_shape, device=device)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (example_input,),
        output_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamo=True,
        external_data=args.external_data,
        verify=args.verify,
    )

    if args.slim:
        slim_onnx_model(output_path, input_shape, args.verify)

    print(f"Exported ONNX model to {output_path}")


def main(args: Any) -> None:
    export_onnx(args)


if __name__ == "__main__":
    parser = ArgumentParser("Export KWT to ONNX.")
    parser.add_argument("--conf", type=str, required=True, help="Path to config file.")
    parser.add_argument("--out", type=str, required=True, help="Output .onnx path.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional checkpoint path. If omitted, exports initialized weights.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Static batch size for the exported graph.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for export. Defaults to cpu.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="Optional ONNX opset version. Defaults to PyTorch's recommended opset.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Ask PyTorch to verify the exported model with ONNX Runtime.",
    )
    parser.add_argument(
        "--slim",
        action="store_true",
        help="Run onnxslim after export.",
    )
    parser.add_argument(
        "--external-data",
        action="store_true",
        help="Store ONNX weights as external data instead of one self-contained file.",
    )
    args = parser.parse_args()

    if args.ckpt is not None and not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Could not find checkpoint {args.ckpt}")

    main(args)
