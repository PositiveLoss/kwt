"""
uv run python carfac/prepare_features.py \
  --data-root ./data \
  --out-dir ./data/.feature_cache_carfac \
  --config config.yaml
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from make_data_list import main as make_data_lists  # noqa: E402
from utils.audio import fix_length, load_audio  # noqa: E402
from utils.dataset import (  # noqa: E402
    get_feature_cache_path,
    resize_feature_frequency,
    resize_feature_time,
    save_feature_to_disk_cache,
)


DEFAULT_AUDIO_SETTINGS: dict[str, Any] = {
    "feature_type": "carfac",
    "feature_time_bins": 98,
    "sr": 16_000,
    "carfac_frame_length": 160,
    "carfac_log_scale": 1.0,
    "carfac_output_channels": None,
}


def load_audio_settings(config_path: Path | None) -> dict[str, Any]:
    settings = dict(DEFAULT_AUDIO_SETTINGS)
    if config_path is not None:
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        settings.update(config.get("hparams", {}).get("audio", {}))
    settings["feature_type"] = "carfac"
    settings.setdefault(
        "feature_time_bins", DEFAULT_AUDIO_SETTINGS["feature_time_bins"]
    )
    settings.setdefault("sr", DEFAULT_AUDIO_SETTINGS["sr"])
    settings.setdefault("carfac_frame_length", int(settings["sr"]) // 100)
    settings.setdefault("carfac_log_scale", DEFAULT_AUDIO_SETTINGS["carfac_log_scale"])
    settings.setdefault("carfac_output_channels", None)
    return settings


def import_jax_carfac(require_tpu: bool) -> tuple[Any, Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
        from carfac.jax import carfac as carfac_jax
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "CARFAC TPU extraction needs JAX. Run `uv sync` after the new "
            "`jax[tpu]` dependency is resolved, or install JAX for your TPU "
            "runtime before running this script."
        ) from exc

    devices = jax.devices()
    tpu_devices = [device for device in devices if device.platform == "tpu"]
    if require_tpu and not tpu_devices:
        device_summary = ", ".join(f"{d.platform}:{d.id}" for d in devices) or "none"
        raise SystemExit(
            "No TPU device is visible to JAX. Found devices: "
            f"{device_summary}. Pass `--allow-non-tpu` to run anyway."
        )
    return jax, jnp, carfac_jax


def discover_wavs(data_root: Path) -> list[str]:
    wavs = [
        path
        for path in data_root.glob("*/*.wav")
        if path.parent.name and not path.parent.name.startswith("_")
    ]
    return [str(path) for path in sorted(wavs)]


def read_list(path: Path, data_root: Path) -> list[str]:
    files: list[str] = []
    with path.open("r") as f:
        for line in f:
            item = line.strip()
            if not item:
                continue
            candidate = Path(item)
            files.append(
                str(candidate if candidate.is_absolute() else data_root / item)
            )
    return files


def collect_dataset_files(data_root: Path, lists: list[Path]) -> list[str]:
    if not lists:
        return discover_wavs(data_root)

    files: list[str] = []
    for list_path in lists:
        files.extend(read_list(list_path, data_root))
    return sorted(set(files))


def ensure_lists(data_root: Path) -> None:
    required = [
        data_root / "training_list.txt",
        data_root / "validation_list.txt",
        data_root / "testing_list.txt",
        data_root / "label_map.json",
    ]
    if all(path.exists() for path in required):
        return

    args = argparse.Namespace(
        val_list_file=str(data_root / "validation_list.txt"),
        test_list_file=str(data_root / "testing_list.txt"),
        data_root=str(data_root),
        out_dir=str(data_root),
    )
    make_data_lists(args)


def make_runner(carfac_jax: Any, jax: Any, audio_settings: dict[str, Any]) -> Any:
    params = carfac_jax.CarfacDesignParameters.with_n_ears(
        n_ears=1,
        fs=float(audio_settings["sr"]),
    )
    hypers, weights, initial_state = carfac_jax.design_and_init_carfac(params)
    segment_runner = jax.jit(
        functools.partial(carfac_jax.run_segment, hypers=hypers),
        static_argnames=("open_loop",),
    )
    return weights, initial_state, segment_runner


def extract_carfac_features(
    path: str,
    audio_settings: dict[str, Any],
    weights: Any,
    initial_state: Any,
    segment_runner: Any,
    jnp: Any,
) -> np.ndarray:
    sr = int(audio_settings["sr"])
    waveform = fix_length(load_audio(path, sr=sr), size=sr)
    input_waves = jnp.asarray(waveform, dtype=jnp.float32).reshape(-1, 1)
    naps, _, _, _, _, _ = segment_runner(
        input_waves,
        weights=weights,
        state=initial_state,
        open_loop=bool(audio_settings.get("carfac_open_loop", False)),
    )
    nap = np.asarray(naps[:, :, 0], dtype=np.float32)

    frame_length = max(1, int(audio_settings.get("carfac_frame_length", sr // 100)))
    num_frames = max(1, nap.shape[0] // frame_length)
    trim = num_frames * frame_length
    if trim > nap.shape[0]:
        nap = np.pad(nap, ((0, trim - nap.shape[0]), (0, 0)))
    else:
        nap = nap[:trim]

    features = nap.reshape(num_frames, frame_length, nap.shape[1]).mean(axis=1).T
    features = np.log1p(
        np.maximum(features, 0.0) * float(audio_settings.get("carfac_log_scale", 1.0))
    )

    output_channels = audio_settings.get("carfac_output_channels")
    if output_channels is not None and int(output_channels) > 0:
        features = resize_feature_frequency(features, int(output_channels))
    return resize_feature_time(
        np.asarray(features, dtype=np.float32),
        audio_settings.get("feature_time_bins"),
    )


def write_metadata(
    out_dir: Path,
    audio_settings: dict[str, Any],
    files: list[str],
    elapsed: float,
    hits: int,
    built: int,
) -> None:
    metadata = {
        "feature_type": "carfac",
        "feature_backend": "carfac-jax:nap",
        "audio_settings": audio_settings,
        "files": len(files),
        "cache_hits": hits,
        "cache_built": built,
        "elapsed_seconds": elapsed,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute CARFAC NAP features for Speech Commands V2."
    )
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("./data/.feature_cache_carfac")
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--list",
        dest="lists",
        type=Path,
        action="append",
        default=[],
        help="Dataset list file. May be passed multiple times.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-non-tpu", action="store_true")
    parser.add_argument("--make-lists", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    out_dir = args.out_dir.resolve()
    if args.make_lists:
        ensure_lists(data_root)

    audio_settings = load_audio_settings(args.config)
    files = collect_dataset_files(data_root, args.lists)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No .wav files found under {data_root}.")

    jax, jnp, carfac_jax = import_jax_carfac(require_tpu=not args.allow_non_tpu)
    print("JAX devices:", ", ".join(str(device) for device in jax.devices()))
    print(f"Preparing {len(files)} files into {out_dir}")
    print(f"Audio settings: {audio_settings}")

    weights, initial_state, segment_runner = make_runner(
        carfac_jax, jax, audio_settings
    )
    warmup = jnp.zeros((int(audio_settings["sr"]), 1), dtype=jnp.float32)
    segment_runner(
        warmup,
        weights=weights,
        state=initial_state,
        open_loop=bool(audio_settings.get("carfac_open_loop", False)),
    )[0].block_until_ready()

    start = time.perf_counter()
    hits = 0
    built = 0
    for path in tqdm(files, desc="carfac features"):
        cache_path = get_feature_cache_path(path, audio_settings, str(out_dir))
        if cache_path.exists() and not args.overwrite:
            hits += 1
            continue
        features = extract_carfac_features(
            path,
            audio_settings,
            weights,
            initial_state,
            segment_runner,
            jnp,
        )
        save_feature_to_disk_cache(cache_path, features)
        built += 1

    elapsed = time.perf_counter() - start
    write_metadata(out_dir, audio_settings, files, elapsed, hits, built)
    print(
        f"Done: files={len(files)}, cache_hits={hits}, built={built}, "
        f"elapsed={elapsed:.1f}s, output={out_dir}"
    )


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    main()
