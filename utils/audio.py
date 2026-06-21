from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: str | Path, sr: int) -> np.ndarray:
    audio, source_sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if source_sr != sr:
        audio = resample_audio(audio, source_sr, sr)
    return np.asarray(audio, dtype=np.float32)


def fix_length(x: np.ndarray, size: int) -> np.ndarray:
    if x.shape[0] > size:
        return x[:size]
    if x.shape[0] < size:
        return np.pad(x, (0, size - x.shape[0]))
    return x


def resample_audio(x: np.ndarray, orig_sr: float, target_sr: float) -> np.ndarray:
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive.")
    if x.size == 0 or orig_sr == target_sr:
        return x.astype(np.float32, copy=False)

    target_len = max(1, int(round(x.shape[0] * target_sr / orig_sr)))
    old_positions = np.linspace(0.0, x.shape[0] - 1, num=x.shape[0])
    new_positions = np.linspace(0.0, x.shape[0] - 1, num=target_len)
    resampled = np.interp(new_positions, old_positions, x)
    return np.asarray(resampled, dtype=np.float32)
