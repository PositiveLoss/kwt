"""Safetensors checkpoint helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

CHECKPOINT_EXTENSION = ".safetensors"

_METADATA_KEY = "torch_kwt_checkpoint"
_OPTIMIZER_PREFIX = "__optimizer__."
_TENSOR_REF_KEY = "__tensor_key__"
_TUPLE_KEY = "__tuple__"
_FORMAT_VERSION = 2


def checkpoint_path(save_dir: str, name: str) -> str:
    """Return the canonical safetensors checkpoint path for a run artifact."""
    return str(Path(save_dir) / f"{name}{CHECKPOINT_EXTENSION}")


def _to_json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return {_TUPLE_KEY: [_to_json_value(item) for item in value]}
    if isinstance(value, list):
        return [_to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"Cannot serialize checkpoint metadata value of type {type(value)}")


def _from_json_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_from_json_value(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {_TUPLE_KEY}:
            return tuple(_from_json_value(item) for item in value[_TUPLE_KEY])
        return {key: _from_json_value(item) for key, item in value.items()}
    return value


def _pack_model_state(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            raise TypeError(f"Model state entry {key!r} is not a tensor")
        tensors[key] = tensor.detach().cpu().contiguous()
    return tensors


def _pack_optimizer_state(
    optimizer_state: dict[str, Any] | None,
    tensors: dict[str, torch.Tensor],
) -> dict[str, Any] | None:
    if optimizer_state is None:
        return None

    packed_state: dict[str, dict[str, Any]] = {}
    for param_id, state_values in optimizer_state.get("state", {}).items():
        param_key = str(param_id)
        packed_values: dict[str, Any] = {}
        for state_key, value in state_values.items():
            if torch.is_tensor(value):
                tensor_key = f"{_OPTIMIZER_PREFIX}state.{param_key}.{state_key}"
                tensors[tensor_key] = value.detach().cpu().contiguous()
                packed_values[state_key] = {_TENSOR_REF_KEY: tensor_key}
            else:
                packed_values[state_key] = _to_json_value(value)
        packed_state[param_key] = packed_values

    return {
        "state": packed_state,
        "param_groups": _to_json_value(optimizer_state.get("param_groups", [])),
    }


def _unpack_optimizer_state(
    optimizer_data: dict[str, Any] | None,
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, Any] | None:
    if optimizer_data is None:
        return None

    state: dict[int, dict[str, Any]] = {}
    for param_key, packed_values in optimizer_data.get("state", {}).items():
        values: dict[str, Any] = {}
        for state_key, value in packed_values.items():
            if isinstance(value, dict) and set(value) == {_TENSOR_REF_KEY}:
                values[state_key] = tensors[value[_TENSOR_REF_KEY]]
            else:
                values[state_key] = _from_json_value(value)
        state[int(param_key)] = values

    return {
        "state": state,
        "param_groups": _from_json_value(optimizer_data.get("param_groups", [])),
    }


def save_checkpoint(
    path: str,
    epoch: int,
    val_acc: float,
    model_state_dict: Mapping[str, torch.Tensor],
    optimizer_state_dict: dict[str, Any] | None = None,
    scheduler_state_dict: dict[str, Any] | None = None,
    step: int = 0,
    best_acc: float | None = None,
) -> None:
    """Save a training checkpoint as a safetensors file."""
    tensors = _pack_model_state(model_state_dict)
    optimizer_data = _pack_optimizer_state(optimizer_state_dict, tensors)
    metadata = {
        _METADATA_KEY: json.dumps(
            {
                "format_version": _FORMAT_VERSION,
                "epoch": epoch,
                "step": step,
                "val_acc": val_acc,
                "best_acc": val_acc if best_acc is None else best_acc,
                "optimizer_state_dict": optimizer_data,
                "scheduler_state_dict": _to_json_value(scheduler_state_dict),
            }
        )
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path, metadata=metadata)


def _load_safetensors_checkpoint(path: str) -> dict[str, Any]:
    with safe_open(path, framework="pt", device="cpu") as checkpoint_file:
        metadata = checkpoint_file.metadata() or {}
        tensors = {
            key: checkpoint_file.get_tensor(key) for key in checkpoint_file.keys()
        }

    checkpoint_metadata = json.loads(metadata.get(_METADATA_KEY, "{}"))
    optimizer_data = checkpoint_metadata.get("optimizer_state_dict")

    return {
        "epoch": checkpoint_metadata.get("epoch", 0),
        "step": checkpoint_metadata.get("step", 0),
        "val_acc": checkpoint_metadata.get("val_acc", 0.0),
        "best_acc": checkpoint_metadata.get(
            "best_acc", checkpoint_metadata.get("val_acc", 0.0)
        ),
        "model_state_dict": {
            key: tensor
            for key, tensor in tensors.items()
            if not key.startswith(_OPTIMIZER_PREFIX)
        },
        "optimizer_state_dict": _unpack_optimizer_state(optimizer_data, tensors),
        "scheduler_state_dict": _from_json_value(
            checkpoint_metadata.get("scheduler_state_dict")
        ),
    }


def load_checkpoint(
    path: str, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load a safetensors checkpoint, with legacy torch checkpoint fallback."""
    if Path(path).suffix == CHECKPOINT_EXTENSION:
        return _load_safetensors_checkpoint(path)
    return torch.load(path, map_location=map_location, weights_only=True)
