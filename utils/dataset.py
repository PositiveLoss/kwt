import functools
import glob
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import spafe
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.audio import fix_length, load_audio
from utils.augment import BackgroundNoiseAdder, time_shift, resample, spec_augment
from utils.connear import (
    CONNEAR_WEIGHTS_URL,
    extract_connear_features_batch,
)
from utils.misc import log_event
from utils.types import Config

FEATURE_CACHE_VERSION = 5


def get_train_val_test_split(
    root: str, val_file: str, test_file: str
) -> tuple[list[str], list[str], list[str], dict[int, str]]:
    """Creates train, val, and test split according to provided val and test files.

    Args:
        root (str): Path to base directory of the dataset.
        val_file (str): Path to file containing list of validation data files.
        test_file (str): Path to file containing list of test data files.

    Returns:
        train_list (list): List of paths to training data items.
        val_list (list): List of paths to validation data items.
        test_list (list): List of paths to test data items.
        label_map (dict): Mapping of indices to label classes.
    """

    ####################
    # Labels
    ####################

    label_list = [
        label
        for label in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, label)) and label[0] != "_"
    ]
    label_map = {idx: label for idx, label in enumerate(label_list)}

    ###################
    # Split
    ###################

    all_files_set = set()
    for label in label_list:
        all_files_set.update(glob.glob(os.path.join(root, label, "*.wav")))

    with open(val_file, "r") as f:
        val_files_set = {
            os.path.join(root, path)
            for path in f.read().rstrip("\n").split("\n")
            if path
        }

    with open(test_file, "r") as f:
        test_files_set = {
            os.path.join(root, path)
            for path in f.read().rstrip("\n").split("\n")
            if path
        }

    if val_files_set.intersection(test_files_set):
        raise ValueError("No files should be common between val and test.")

    all_files_set -= val_files_set
    all_files_set -= test_files_set

    train_list, val_list, test_list = (
        list(all_files_set),
        list(val_files_set),
        list(test_files_set),
    )

    print(f"Number of training samples: {len(train_list)}")
    print(f"Number of validation samples: {len(val_list)}")
    print(f"Number of test samples: {len(test_list)}")

    return train_list, val_list, test_list, label_map


class GoogleSpeechDataset(Dataset):
    """Dataset wrapper for Google Speech Commands V2."""

    def __init__(
        self,
        data_list: list[str],
        audio_settings: dict[str, Any],
        label_map: dict[str, str] | None = None,
        aug_settings: dict[str, Any] | None = None,
        cache: int = 0,
        feature_cache_dir: str | None = None,
    ) -> None:
        super().__init__()

        self.audio_settings = audio_settings
        self.aug_settings = aug_settings
        self.cache = cache
        self.feature_cache_dir = feature_cache_dir

        self.data_list: list[str] | list[np.ndarray]
        if cache:
            print("Caching dataset into memory.")
            self.data_list = init_cache(
                data_list,
                audio_settings["sr"],
                cache,
                audio_settings,
                feature_cache_dir,
            )
        else:
            self.data_list = data_list

        # labels: if no label map is provided, will not load labels. (Use for inference)
        self.label_list: list[int] | None
        if label_map is not None:
            self.label_list = []
            label_2_idx = {v: int(k) for k, v in label_map.items()}
            for path in data_list:
                self.label_list.append(label_2_idx[path.split("/")[-2]])
        else:
            self.label_list = None

        if aug_settings is not None:
            if "bg_noise" in aug_settings:
                self.bg_adder = BackgroundNoiseAdder(
                    sounds_path=aug_settings["bg_noise"]["bg_folder"]
                )

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, int]:
        features_extracted = self.cache >= 2
        if self.cache:
            x = cast(list[np.ndarray], self.data_list)[index]
            if self.aug_settings is not None:
                x = x.copy()
        elif self.feature_cache_dir is not None:
            path = cast(list[str], self.data_list)[index]
            x = load_feature_from_disk_cache(
                path,
                sr=self.audio_settings["sr"],
                audio_settings=self.audio_settings,
                cache_dir=self.feature_cache_dir,
            )
            features_extracted = True
        else:
            path = cast(list[str], self.data_list)[index]
            x = load_audio(path, sr=self.audio_settings["sr"])

        x = self.transform(x, features_extracted=features_extracted)

        if self.label_list is not None:
            label = self.label_list[index]
            return x, label
        else:
            return x

    def transform(
        self, x: np.ndarray, features_extracted: bool = False
    ) -> torch.Tensor:
        """Applies necessary preprocessing to audio.

        Args:
            x (np.ndarray) - Input waveform; array of shape (n_samples, ).

        Returns:
            x (torch.FloatTensor) - MFCC matrix of shape (n_mfcc, T).
        """

        sr = self.audio_settings["sr"]

        ###################
        # Waveform
        ###################

        if not features_extracted:
            if self.aug_settings is not None:
                if "bg_noise" in self.aug_settings:
                    x = self.bg_adder(samples=x, sample_rate=sr)

                if "time_shift" in self.aug_settings:
                    x = time_shift(x, sr, **self.aug_settings["time_shift"])

                if "resample" in self.aug_settings:
                    x, _ = resample(x, sr, **self.aug_settings["resample"])

            x = extract_features(x, self.audio_settings)

        if self.aug_settings is not None:
            if "spec_aug" in self.aug_settings:
                x = spec_augment(x, **self.aug_settings["spec_aug"])

        tensor = torch.from_numpy(x).float().unsqueeze(0)
        return tensor


def extract_features(x: np.ndarray, audio_settings: dict[str, Any]) -> np.ndarray:
    sr = audio_settings["sr"]
    x = fix_length(x, size=sr)
    feature_type = get_feature_type(audio_settings)
    if feature_type == "mfcc":
        features = extract_mfcc_spafe(x, audio_settings)
    elif feature_type == "cochleagram":
        features = extract_cochleagram_spafe(x, audio_settings)
    elif feature_type == "connear":
        features = extract_connear(x, audio_settings)
    else:
        raise ValueError(
            "hparams.audio.feature_type must be one of mfcc, cochleagram, or connear."
        )
    return resize_feature_time(features, audio_settings.get("feature_time_bins"))


def get_feature_type(audio_settings: dict[str, Any]) -> str:
    return str(audio_settings.get("feature_type", "mfcc")).lower()


def extract_mfcc_spafe(x: np.ndarray, audio_settings: dict[str, Any]) -> np.ndarray:
    sr = audio_settings["sr"]
    opts = spafe.FeatureOptions(
        fs=sr,
        num_ceps=audio_settings["n_mels"],
        nfilts=audio_settings["n_mels"],
        nfft=audio_settings["n_fft"],
        win_len=audio_settings["win_length"] / sr,
        win_hop=audio_settings["hop_length"] / sr,
        pre_emph=audio_settings.get("pre_emph", False),
    )
    features = np.asarray(
        spafe.mfcc(x.astype(np.float64).tolist(), opts), dtype=np.float32
    )
    return features.T


def extract_cochleagram_spafe(
    x: np.ndarray, audio_settings: dict[str, Any]
) -> np.ndarray:
    sr = audio_settings["sr"]
    configured_high_freq = audio_settings.get("coch_high_freq")
    configured_filter_n = audio_settings.get("coch_filter_n")
    high_lim = min(float(configured_high_freq or sr / 2), sr / 2)
    filter_n = int(configured_filter_n or audio_settings["n_mels"] - 2)
    opts = spafe.CochleagramOptions(
        signal_size=x.shape[0],
        sr=sr,
        env_sr=int(audio_settings.get("coch_env_sr", 100)),
        filter_n=filter_n,
        low_lim=float(audio_settings.get("coch_low_freq", 50.0)),
        high_lim=high_lim,
        sample_factor=int(audio_settings.get("coch_sample_factor", 1)),
    )
    output = spafe.cochleagram(x.astype(np.float64).tolist(), opts)
    return np.asarray(output.cochleagram, dtype=np.float32)


def extract_connear(x: np.ndarray, audio_settings: dict[str, Any]) -> np.ndarray:
    return extract_connear_batch([x], audio_settings)[0]


def extract_connear_batch(
    waveforms: list[np.ndarray], audio_settings: dict[str, Any]
) -> list[np.ndarray]:
    target_channels = audio_settings.get("connear_n_channels")
    if target_channels is None:
        target_channels = audio_settings["n_mels"]

    features = extract_connear_features_batch(
        waveforms,
        sr=int(audio_settings["sr"]),
        weights_path=str(
            audio_settings.get("connear_weights_path", "./data/connear/Gmodel.pt")
        ),
        device=str(audio_settings.get("connear_device", "cpu")),
        auto_download=bool(audio_settings.get("connear_auto_download", False)),
        model_sr=int(audio_settings.get("connear_sr", 20_000)),
        input_scale=float(audio_settings.get("connear_input_scale", 1.0)),
        output_channels=int(target_channels),
        output_time_bins=int(audio_settings.get("feature_time_bins", 98)),
        log_scale=float(audio_settings.get("connear_log_scale", 1_000_000.0)),
        normalize=bool(audio_settings.get("connear_normalize", True)),
    )
    return [feature for feature in features]


def resize_feature_time(
    features: np.ndarray, feature_time_bins: Any | None
) -> np.ndarray:
    if feature_time_bins is None:
        return features

    target_size = int(feature_time_bins)
    if target_size <= 0 or features.shape[1] == target_size:
        return features

    old_positions = np.linspace(0.0, 1.0, num=features.shape[1])
    new_positions = np.linspace(0.0, 1.0, num=target_size)
    resized = [np.interp(new_positions, old_positions, row) for row in features]
    return np.asarray(resized, dtype=np.float32)


def get_feature_cache_settings(audio_settings: dict[str, Any]) -> dict[str, Any]:
    cache_settings = dict(audio_settings)
    feature_type = get_feature_type(audio_settings)
    if feature_type == "connear":
        cache_settings["connear_weights_url"] = CONNEAR_WEIGHTS_URL
    cache_settings["feature_backend"] = (
        "connear-pytorch" if feature_type == "connear" else f"spafe-rs:{feature_type}"
    )
    return cache_settings


def get_feature_cache_path(
    path: str, audio_settings: dict[str, Any], cache_dir: str
) -> Path:
    source = Path(path).resolve()
    stat = source.stat()
    cache_payload = {
        "version": FEATURE_CACHE_VERSION,
        "path": str(source),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "audio_settings": get_feature_cache_settings(audio_settings),
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return Path(cache_dir) / cache_key[:2] / f"{cache_key}.npy"


def load_feature_from_disk_cache(
    path: str,
    sr: int,
    audio_settings: dict[str, Any],
    cache_dir: str,
) -> np.ndarray:
    cache_path = get_feature_cache_path(path, audio_settings, cache_dir)
    if cache_path.exists():
        return np.load(cache_path, allow_pickle=False)

    x = load_audio(path, sr=sr)
    features = extract_features(x, audio_settings)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "wb") as f:
        np.save(f, features, allow_pickle=False)
    os.replace(tmp_path, cache_path)
    return features


def save_feature_to_disk_cache(
    cache_path: Path,
    features: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "wb") as f:
        np.save(f, features, allow_pickle=False)
    os.replace(tmp_path, cache_path)


def cache_item_loader(
    path: str,
    sr: int,
    cache_level: int,
    audio_settings: dict[str, Any],
    feature_cache_dir: str | None = None,
) -> np.ndarray:
    if cache_level == 2:
        if feature_cache_dir is not None:
            return load_feature_from_disk_cache(
                path, sr, audio_settings, feature_cache_dir
            )
        x = load_audio(path, sr=sr)
        return extract_features(x, audio_settings)
    return load_audio(path, sr=sr)


def init_connear_feature_cache(
    data_list: list[str],
    sr: int,
    audio_settings: dict[str, Any],
    feature_cache_dir: str | None = None,
) -> list[np.ndarray]:
    """Batched CoNNear cache builder.

    CoNNear is a neural model and is expensive one sample at a time. This path
    intentionally avoids multiprocessing, especially for MPS, and batches the
    model forward pass instead.
    """

    batch_size = int(audio_settings.get("connear_batch_size", 4))
    if batch_size < 1:
        raise ValueError("hparams.audio.connear_batch_size must be >= 1.")

    cache: list[np.ndarray | None] = [None] * len(data_list)
    pending_indices: list[int] = []
    pending_waveforms: list[np.ndarray] = []

    def flush_pending(progress: tqdm) -> None:
        nonlocal pending_indices, pending_waveforms
        if not pending_waveforms:
            return

        features_batch = extract_connear_batch(pending_waveforms, audio_settings)
        for index, features in zip(pending_indices, features_batch, strict=True):
            if feature_cache_dir is not None:
                cache_path = get_feature_cache_path(
                    data_list[index], audio_settings, feature_cache_dir
                )
                save_feature_to_disk_cache(cache_path, features)
            cache[index] = features
            progress.update(1)

        pending_indices = []
        pending_waveforms = []

    with tqdm(total=len(data_list), desc="cache connear features") as progress:
        for index, path in enumerate(data_list):
            if feature_cache_dir is not None:
                cache_path = get_feature_cache_path(
                    path, audio_settings, feature_cache_dir
                )
                if cache_path.exists():
                    cache[index] = np.load(cache_path, allow_pickle=False)
                    progress.update(1)
                    continue

            pending_indices.append(index)
            pending_waveforms.append(fix_length(load_audio(path, sr=sr), size=sr))
            if len(pending_waveforms) >= batch_size:
                flush_pending(progress)

        flush_pending(progress)

    return cast(list[np.ndarray], cache)


def init_cache(
    data_list: list[str],
    sr: int,
    cache_level: int,
    audio_settings: dict[str, Any],
    feature_cache_dir: str | None = None,
    n_cache_workers: int = 4,
) -> list[np.ndarray]:
    """Loads entire dataset into memory for later use.

    Args:
        data_list (list): List of data items.
        sr (int): Sampling rate.
        cache_level (int): Cache levels, one of (1, 2), caching wavs and spectrograms respectively.
        n_cache_workers (int, optional): Number of workers. Defaults to 4.

    Returns:
        cache (list): List of data items.
    """

    cache: list[np.ndarray] = []

    if cache_level == 2 and get_feature_type(audio_settings) == "connear":
        return init_connear_feature_cache(
            data_list,
            sr=sr,
            audio_settings=audio_settings,
            feature_cache_dir=feature_cache_dir,
        )

    loader_fn = functools.partial(
        cache_item_loader,
        sr=sr,
        cache_level=cache_level,
        audio_settings=audio_settings,
        feature_cache_dir=feature_cache_dir,
    )

    with mp.Pool(n_cache_workers) as pool:
        for audio in tqdm(
            pool.imap(func=loader_fn, iterable=data_list), total=len(data_list)
        ):
            cache.append(audio)

    return cache


def get_loader(
    data_list: list[str],
    config: Config,
    train: bool = True,
    split_name: str | None = None,
) -> DataLoader:
    """Creates dataloaders for training, validation and testing.

    Args:
        config (dict): Dict containing various settings for the training run.
        train (bool): Training or evaluation mode.

    Returns:
        dataloader (DataLoader): DataLoader wrapper for training/validation/test data.
    """

    split = split_name or ("train" if train else "eval")
    audio_settings = config["hparams"]["audio"]
    feature_type = get_feature_type(audio_settings)
    feature_cache_dir = (
        config["exp"].get("feature_cache_dir")
        if config["exp"].get("feature_cache", False)
        else None
    )
    cache_message = (
        f"Preparing {split} dataset: samples={len(data_list)}, "
        f"feature_type={feature_type}, cache={config['exp']['cache']}, "
        f"feature_cache={bool(feature_cache_dir)}, "
        f"num_workers={config['exp']['n_workers']}, "
        f"batch_size={config['hparams']['batch_size']}."
    )
    if feature_type == "connear":
        cache_message += (
            f" connear_device={audio_settings.get('connear_device', 'cpu')}, "
            f"connear_batch_size={audio_settings.get('connear_batch_size', 4)}."
        )
    log_event(cache_message, config)

    with open(config["label_map"], "r") as f:
        label_map = json.load(f)

    dataset = GoogleSpeechDataset(
        data_list=data_list,
        label_map=label_map,
        audio_settings=audio_settings,
        aug_settings=config["hparams"]["augment"] if train else None,
        cache=config["exp"]["cache"],
        feature_cache_dir=feature_cache_dir,
    )
    log_event(f"Prepared {split} dataset with {len(dataset)} samples.", config)

    dataloader = DataLoader(
        dataset,
        batch_size=config["hparams"]["batch_size"],
        num_workers=config["exp"]["n_workers"],
        pin_memory=config["exp"]["pin_memory"],
        shuffle=train,
    )

    return dataloader


def warm_loader_cache(dataloader: DataLoader, split_name: str) -> None:
    """Iterate a dataloader once to warm dataset, worker, and preprocessing caches."""

    print(f"Warming {split_name} cache.")
    for _ in tqdm(dataloader, desc=f"warm {split_name}"):
        pass
