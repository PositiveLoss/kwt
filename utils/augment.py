import glob
import os

import numpy as np
import numba as nb

from utils.audio import fix_length, load_audio, resample_audio


class BackgroundNoiseAdder:
    def __init__(
        self,
        sounds_path: str,
        min_snr_db: float = 3.0,
        max_snr_db: float = 30.0,
    ) -> None:
        self.sound_paths = glob.glob(os.path.join(sounds_path, "*.wav"))
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

    def __call__(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.sound_paths:
            return samples

        noise = load_audio(np.random.choice(self.sound_paths), sr=sample_rate)
        if noise.shape[0] < samples.shape[0]:
            repeats = int(np.ceil(samples.shape[0] / noise.shape[0]))
            noise = np.tile(noise, repeats)
        if noise.shape[0] > samples.shape[0]:
            start = np.random.randint(0, noise.shape[0] - samples.shape[0] + 1)
            noise = noise[start : start + samples.shape[0]]
        noise = fix_length(noise, samples.shape[0])

        signal_rms = np.sqrt(np.mean(samples * samples))
        noise_rms = np.sqrt(np.mean(noise * noise))
        if signal_rms == 0.0 or noise_rms == 0.0:
            return samples

        snr_db = np.random.uniform(self.min_snr_db, self.max_snr_db)
        target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
        mixed = samples + noise * (target_noise_rms / noise_rms)
        return mixed.astype(np.float32)


@nb.jit(nopython=True, cache=True)
def time_shift(wav: np.ndarray, sr: int, s_min: float, s_max: float) -> np.ndarray:
    """Time shift augmentation.
    Refer to https://www.kaggle.com/haqishen/augmentation-methods-for-audio#1.-Time-shifting.
    Changed np.r_ to np.hstack for numba support.

    Args:
        wav (np.ndarray): Waveform array of shape (n_samples,).
        sr (int): Sampling rate.
        s_min (float): Minimum fraction of a second by which to shift.
        s_max (float): Maximum fraction of a second by which to shift.

    Returns:
        wav_time_shift (np.ndarray): Time-shifted waveform array.
    """

    start = int(np.random.uniform(sr * s_min, sr * s_max))
    if start >= 0:
        wav_time_shift = np.hstack(
            (wav[start:], np.random.uniform(-0.001, 0.001, start))
        )
    else:
        wav_time_shift = np.hstack(
            (np.random.uniform(-0.001, 0.001, -start), wav[:start])
        )

    return wav_time_shift


def resample(
    x: np.ndarray, sr: int, r_min: float, r_max: float
) -> tuple[np.ndarray, float]:
    """Resamples waveform.

    Args:
        x (np.ndarray): Input waveform, array of shape (n_samples, ).
        sr (int): Sampling rate.
        r_min (float): Minimum percentage of resampling.
        r_max (float): Maximum percentage of resampling.
    """

    sr_new = sr * np.random.uniform(r_min, r_max)
    x = resample_audio(x, orig_sr=sr, target_sr=sr_new)
    return x, sr_new


@nb.jit(nopython=True, cache=True)
def spec_augment(
    mel_spec: np.ndarray,
    n_time_masks: int,
    time_mask_width: int,
    n_freq_masks: int,
    freq_mask_width: int,
):
    """Numpy implementation of spectral augmentation.

    Args:
        mel_spec (np.ndarray): Mel spectrogram, array of shape (n_mels, T).
        n_time_masks (int): Number of time bands.
        time_mask_width (int): Max width of each time band.
        n_freq_masks (int): Number of frequency bands.
        freq_mask_width (int): Max width of each frequency band.

    Returns:
        mel_spec (np.ndarray): Spectrogram with random time bands and freq bands masked out.
    """

    offset, begin = 0, 0

    for _ in range(n_time_masks):
        offset = np.random.randint(0, time_mask_width)
        begin = np.random.randint(0, mel_spec.shape[1] - offset)
        mel_spec[:, begin : begin + offset] = 0.0

    for _ in range(n_freq_masks):
        offset = np.random.randint(0, freq_mask_width)
        begin = np.random.randint(0, mel_spec.shape[0] - offset)
        mel_spec[begin : begin + offset, :] = 0.0

    return mel_spec
