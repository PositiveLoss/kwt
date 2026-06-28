"""Run ONNX Runtime inference on short ~1s clips."""

import glob
import json
import os
from argparse import ArgumentParser, Namespace
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_parser import get_config
from utils.dataset import GoogleSpeechDataset


def get_providers(device: str) -> list[str]:
    """Resolve an ONNX Runtime execution provider list from a device string."""

    available = ort.get_available_providers()

    if device == "auto":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    if device == "cpu":
        return ["CPUExecutionProvider"]

    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDA was requested, but this ONNX Runtime installation does not "
                "provide CUDAExecutionProvider. Available providers: "
                f"{', '.join(available)}."
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    raise ValueError("Unsupported device. Expected one of: auto, cpu, cuda.")


def dtype_from_onnx_type(input_type: str) -> Any:
    if input_type == "tensor(float)":
        return np.float32
    if input_type == "tensor(float16)":
        return np.float16
    if input_type == "tensor(double)":
        return np.float64
    if input_type == "tensor(bfloat16)":
        try:
            import ml_dtypes
        except ImportError as exc:
            raise RuntimeError(
                "The ONNX model expects bfloat16 input, but ml_dtypes is not "
                "installed. Install ml_dtypes or export the model with "
                "--dtype float32 or --dtype float16."
            ) from exc
        return ml_dtypes.bfloat16

    raise ValueError(f"Unsupported ONNX model input type: {input_type}.")


def numpy_batch(data: torch.Tensor, dtype: Any) -> np.ndarray:
    batch = data.detach().cpu().numpy()
    if batch.dtype != dtype:
        batch = batch.astype(dtype)
    return np.ascontiguousarray(batch)


def static_batch_size(input_shape: list[Any]) -> int | None:
    if not input_shape:
        return None

    batch_size = input_shape[0]
    if isinstance(batch_size, int):
        return batch_size
    return None


def pad_to_static_batch(
    batch: np.ndarray, expected_batch_size: int | None
) -> np.ndarray:
    if expected_batch_size is None or batch.shape[0] == expected_batch_size:
        return batch

    if batch.shape[0] > expected_batch_size:
        raise ValueError(
            "ONNX model has a static batch size of "
            f"{expected_batch_size}, but received a batch with {batch.shape[0]} "
            "items. Lower --batch_size or re-export the model with a larger "
            "--batch-size."
        )

    padding = np.zeros(
        (expected_batch_size - batch.shape[0], *batch.shape[1:]),
        dtype=batch.dtype,
    )
    return np.concatenate([batch, padding], axis=0)


def create_session(model_path: str, device: str) -> ort.InferenceSession:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find ONNX model {model_path}")

    providers = get_providers(device)
    return ort.InferenceSession(model_path, providers=providers)


def get_preds(
    session: ort.InferenceSession,
    dataloader: DataLoader,
) -> list[int]:
    """Performs ONNX Runtime inference."""

    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"Expected a single ONNX input, found {len(inputs)}.")

    input_info = inputs[0]
    input_name = input_info.name
    input_dtype = dtype_from_onnx_type(input_info.type)
    expected_batch_size = static_batch_size(input_info.shape)
    output_names = [output.name for output in session.get_outputs()]

    preds_list: list[int] = []
    for data in tqdm(dataloader):
        batch = numpy_batch(data, input_dtype)
        item_count = batch.shape[0]
        batch = pad_to_static_batch(batch, expected_batch_size)
        outputs = session.run(output_names, {input_name: batch})
        logits = np.asarray(outputs[0])[:item_count]
        preds = logits.argmax(1).ravel().tolist()
        preds_list.extend(int(pred) for pred in preds)

    return preds_list


def main(args: Namespace) -> None:
    ######################
    # load config
    ######################
    config = get_config(args.conf)

    ######################
    # setup data
    ######################
    if os.path.isdir(args.inp):
        data_list = glob.glob(os.path.join(args.inp, "*.wav"))
    elif os.path.isfile(args.inp):
        data_list = [args.inp]
    else:
        raise FileNotFoundError(f"Could not find input {args.inp}")

    dataset = GoogleSpeechDataset(
        data_list=data_list,
        label_map=None,
        audio_settings=config["hparams"]["audio"],
        aug_settings=None,
        cache=0,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    ######################
    # run inference
    ######################
    session = create_session(args.onnx, args.device)
    preds = get_preds(session, dataloader)

    ######################
    # save predictions
    ######################
    if args.lmap:
        with open(args.lmap, "r") as f:
            label_map = json.load(f)
        preds = [label_map[str(pred)] for pred in preds]

    pred_dict = dict(zip(data_list, preds))

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "preds_onnx.json")

    with open(out_path, "w+") as f:
        json.dump(pred_dict, f)

    print(f"Saved preds to {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--conf",
        type=str,
        required=True,
        help="Path to config file. Used to process audio features.",
    )
    parser.add_argument(
        "--onnx", type=str, required=True, help="Path to exported .onnx model."
    )
    parser.add_argument(
        "--inp",
        type=str,
        required=True,
        help="Path to input. Can be a path to a .wav file, or a path to a folder containing .wav files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./",
        help="Path to output folder. Predictions will be stored in {out}/preds_onnx.json.",
    )
    parser.add_argument(
        "--lmap",
        type=str,
        default=None,
        help="Path to label_map.json. If not provided, will save predictions as class indices instead of class names.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="ONNX Runtime device. One of auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for batch inference."
    )

    args = parser.parse_args()

    main(args)
