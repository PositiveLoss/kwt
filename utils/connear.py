"""CoNNear cochlea feature extraction.

The architecture mirrors the PyTorch CoNNear port published by
PositiveLoss/CoNNear_cochlea-PyTorch.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils.audio import resample_audio
from utils.device import resolve_device

CONNEAR_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/PositiveLoss/"
    "CoNNear_cochlea-PyTorch/main/connear/Gmodel.pt"
)
CONTEXT_SAMPLES = 256
KERNEL_SIZE = 64
STRIDE = 2
HIDDEN_CHANNELS = 128
OUTPUT_CHANNELS = 201


def _same_pad_1d(x: Tensor, kernel_size: int, stride: int) -> Tensor:
    input_length = x.shape[-1]
    output_length = (input_length + stride - 1) // stride
    pad_total = max((output_length - 1) * stride + kernel_size - input_length, 0)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return F.pad(x, (pad_left, pad_right))


class KerasSameConv1d(nn.Conv1d):
    """Conv1d with TensorFlow/Keras same-padding semantics."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False,
        )

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(_same_pad_1d(input, self.kernel_size[0], self.stride[0]))


class CoNNear(nn.Module):
    """Pretrained CoNNear architecture for basilar-membrane displacement."""

    def __init__(self, output_channels: int = OUTPUT_CHANNELS) -> None:
        super().__init__()
        self.enc1 = KerasSameConv1d(1, HIDDEN_CHANNELS, KERNEL_SIZE, STRIDE)
        self.enc2 = KerasSameConv1d(
            HIDDEN_CHANNELS, HIDDEN_CHANNELS, KERNEL_SIZE, STRIDE
        )
        self.enc3 = KerasSameConv1d(
            HIDDEN_CHANNELS, HIDDEN_CHANNELS, KERNEL_SIZE, STRIDE
        )
        self.enc4 = KerasSameConv1d(
            HIDDEN_CHANNELS, HIDDEN_CHANNELS, KERNEL_SIZE, STRIDE
        )

        self.dec1 = nn.ConvTranspose1d(
            HIDDEN_CHANNELS,
            HIDDEN_CHANNELS,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=31,
            bias=False,
        )
        self.dec2 = nn.ConvTranspose1d(
            HIDDEN_CHANNELS * 2,
            HIDDEN_CHANNELS,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=31,
            bias=False,
        )
        self.dec3 = nn.ConvTranspose1d(
            HIDDEN_CHANNELS * 2,
            HIDDEN_CHANNELS,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=31,
            bias=False,
        )
        self.dec4 = nn.ConvTranspose1d(
            HIDDEN_CHANNELS * 2,
            output_channels,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=31,
            bias=False,
        )

    def forward(self, x: Tensor, channels_last: bool = True) -> Tensor:
        if channels_last:
            x = x.transpose(1, 2)

        c1 = self.enc1(x)
        a1 = torch.tanh(c1)
        c2 = self.enc2(a1)
        a2 = torch.tanh(c2)
        c3 = self.enc3(a2)
        a3 = torch.tanh(c3)
        c4 = self.enc4(a3)
        x = torch.tanh(c4)

        x = torch.tanh(self.dec1(x))
        x = torch.cat([x, c3], dim=1)
        x = torch.tanh(self.dec2(x))
        x = torch.cat([x, c2], dim=1)
        x = torch.tanh(self.dec3(x))
        x = torch.cat([x, c1], dim=1)
        x = self.dec4(x)
        x = x[..., CONTEXT_SAMPLES:-CONTEXT_SAMPLES]

        if channels_last:
            x = x.transpose(1, 2)
        return x


def ensure_connear_weights(weights_path: str | Path, auto_download: bool) -> Path:
    path = Path(weights_path)
    if path.exists():
        return path
    if not auto_download:
        raise FileNotFoundError(
            f"CoNNear weights not found at {path}. Download them with "
            "`uv run python download_connear_model.py` or set "
            "hparams.audio.connear_auto_download: True."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if path.exists():
                return path
            time.sleep(0.25)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        urlretrieve(CONNEAR_WEIGHTS_URL, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
        if lock_path.exists():
            lock_path.unlink()

    return path


@lru_cache(maxsize=4)
def load_connear(weights_path: str, device: str = "cpu") -> CoNNear:
    device = str(resolve_device(device))
    model = CoNNear()
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(torch.device(device))
    model.eval()
    return model


def _pad_to_stride_multiple(x: np.ndarray, multiple: int = 16) -> np.ndarray:
    remainder = x.shape[0] % multiple
    if remainder == 0:
        return x
    return np.pad(x, (0, multiple - remainder))


def prepare_connear_waveform(
    x: np.ndarray,
    sr: int,
    model_sr: int = 20_000,
    input_scale: float = 1.0,
) -> np.ndarray:
    """Resample, pad, and scale a waveform for CoNNear inference."""
    if sr != model_sr:
        x = resample_audio(x, sr, model_sr)
    x = _pad_to_stride_multiple(x.astype(np.float32, copy=False))
    return x * np.float32(input_scale)


@torch.no_grad()
def extract_connear_channels_batch(
    waveforms: list[np.ndarray],
    sr: int,
    weights_path: str | Path,
    device: str = "cpu",
    auto_download: bool = False,
    model_sr: int = 20_000,
    input_scale: float = 1.0,
) -> np.ndarray:
    """Run CoNNear on a waveform batch and return ``(batch, channels, time)``."""
    path = ensure_connear_weights(weights_path, auto_download)
    prepared = [
        prepare_connear_waveform(
            waveform, sr=sr, model_sr=model_sr, input_scale=input_scale
        )
        for waveform in waveforms
    ]
    lengths = {waveform.shape[0] for waveform in prepared}
    if len(lengths) != 1:
        raise ValueError("Batched CoNNear extraction requires equal waveform lengths.")

    target_device = resolve_device(device)
    model = load_connear(str(path), device)
    batch = torch.from_numpy(np.stack(prepared)).to(target_device).float().unsqueeze(-1)
    output = model(batch, channels_last=True)
    return output.transpose(1, 2).cpu().numpy().astype(np.float32, copy=False)


def compress_connear_features(
    features: Tensor,
    output_channels: int = 40,
    output_time_bins: int = 98,
    log_scale: float = 1_000_000.0,
    normalize: bool = True,
) -> Tensor:
    """Convert dense BM displacement into KWT-ready envelope features.

    CoNNear returns sample-rate displacement traces. For keyword spotting we want
    a lower-rate per-channel envelope, so the temporal axis is rectified and
    average-pooled instead of directly interpolated.
    """
    if features.ndim != 3:
        raise ValueError("CoNNear features must have shape (batch, channels, time).")
    if output_channels <= 0:
        raise ValueError("output_channels must be positive.")
    if output_time_bins <= 0:
        raise ValueError("output_time_bins must be positive.")

    features = features.abs()
    features = F.adaptive_avg_pool1d(features, output_time_bins)
    features = torch.log1p(features * log_scale)

    if features.shape[1] != output_channels:
        features = F.interpolate(
            features.unsqueeze(1),
            size=(output_channels, output_time_bins),
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)

    if normalize:
        mean = features.mean(dim=(1, 2), keepdim=True)
        std = features.std(dim=(1, 2), keepdim=True)
        features = (features - mean) / (std + 1e-6)

    return features


@torch.no_grad()
def extract_connear_features_batch(
    waveforms: list[np.ndarray],
    sr: int,
    weights_path: str | Path,
    device: str = "cpu",
    auto_download: bool = False,
    model_sr: int = 20_000,
    input_scale: float = 1.0,
    output_channels: int = 40,
    output_time_bins: int = 98,
    log_scale: float = 1_000_000.0,
    normalize: bool = True,
) -> np.ndarray:
    """Run CoNNear and return compressed KWT-ready features.

    This keeps the large ``201 x time`` CoNNear output on the target device and
    transfers only the compact ``output_channels x output_time_bins`` result.
    """
    path = ensure_connear_weights(weights_path, auto_download)
    prepared = [
        prepare_connear_waveform(
            waveform, sr=sr, model_sr=model_sr, input_scale=input_scale
        )
        for waveform in waveforms
    ]
    lengths = {waveform.shape[0] for waveform in prepared}
    if len(lengths) != 1:
        raise ValueError("Batched CoNNear extraction requires equal waveform lengths.")

    target_device = resolve_device(device)
    model = load_connear(str(path), device)
    batch = torch.from_numpy(np.stack(prepared)).to(target_device).float().unsqueeze(-1)
    features = model(batch, channels_last=True).transpose(1, 2)
    features = compress_connear_features(
        features,
        output_channels=output_channels,
        output_time_bins=output_time_bins,
        log_scale=log_scale,
        normalize=normalize,
    )

    return features.cpu().numpy().astype(np.float32, copy=False)


@torch.no_grad()
def extract_connear_channels(
    x: np.ndarray,
    sr: int,
    weights_path: str | Path,
    device: str = "cpu",
    auto_download: bool = False,
    model_sr: int = 20_000,
    input_scale: float = 1.0,
) -> np.ndarray:
    """Run CoNNear and return channels-first BM displacement features."""
    return extract_connear_channels_batch(
        [x],
        sr=sr,
        weights_path=weights_path,
        device=device,
        auto_download=auto_download,
        model_sr=model_sr,
        input_scale=input_scale,
    )[0]
