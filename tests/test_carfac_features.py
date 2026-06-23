import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from utils.dataset import (
    extract_features,
    get_feature_cache_path,
    get_feature_cache_settings,
    validate_feature_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(name: str) -> dict:
    with (PROJECT_ROOT / name).open("r") as f:
        return yaml.safe_load(f)


class CarfacFeatureTests(unittest.TestCase):
    def test_carfac_smallest_shape_is_finite_float32(self) -> None:
        config = load_config("config_carfac_smallest.yaml")
        audio = dict(config["hparams"]["audio"])
        audio["carfac_backend"] = "np"
        x = np.zeros(int(audio["sr"]), dtype=np.float32)

        features = extract_features(x, audio)

        self.assertEqual(features.shape, (65, 98))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())

    def test_carfac_base_shape_is_finite_float32(self) -> None:
        config = load_config("config_carfac_base.yaml")
        audio = dict(config["hparams"]["audio"])
        audio["carfac_backend"] = "np"
        x = np.zeros(int(audio["sr"]), dtype=np.float32)

        features = extract_features(x, audio)

        self.assertEqual(features.shape, (65, 128))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())

    def test_carfac_output_channels_resizes_frequency_axis(self) -> None:
        config = load_config("config_carfac_smallest.yaml")
        audio = dict(config["hparams"]["audio"])
        audio["carfac_backend"] = "np"
        audio["carfac_output_channels"] = 40
        x = np.zeros(int(audio["sr"]), dtype=np.float32)

        features = extract_features(x, audio)

        self.assertEqual(features.shape, (40, 98))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())

    def test_carfac_np_and_jax_cache_paths_are_distinct(self) -> None:
        config = load_config("config_carfac_smallest.yaml")
        audio_np = dict(config["hparams"]["audio"])
        audio_jax = dict(audio_np)
        audio_np["carfac_backend"] = "np"
        audio_jax["carfac_backend"] = "jax"

        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "sample.wav"
            source.write_bytes(b"not-a-real-wav")

            path_np = get_feature_cache_path(str(source), audio_np, tmp_dir)
            path_jax = get_feature_cache_path(str(source), audio_jax, tmp_dir)

        self.assertNotEqual(path_np, path_jax)
        self.assertEqual(
            get_feature_cache_settings(audio_np)["feature_backend"], "carfac-np:nap"
        )
        self.assertEqual(
            get_feature_cache_settings(audio_jax)["feature_backend"], "carfac-jax:nap"
        )

    def test_non_carfac_cache_backend_is_unchanged(self) -> None:
        settings = {
            "feature_type": "mfcc",
            "sr": 16000,
            "n_mels": 40,
            "n_fft": 480,
            "win_length": 480,
            "hop_length": 160,
        }

        cache_settings = get_feature_cache_settings(settings)

        self.assertEqual(cache_settings["feature_backend"], "spafe-rs:mfcc")
        self.assertNotIn("carfac_backend", cache_settings)

    def test_matching_carfac_config_validation_passes(self) -> None:
        config = load_config("config_carfac_smallest.yaml")

        validate_feature_config(config)

    def test_mismatched_carfac_input_res_validation_fails(self) -> None:
        config = load_config("config_carfac_smallest.yaml")
        bad_config = copy.deepcopy(config)
        bad_config["hparams"]["model"]["input_res"] = [65, 100]

        with self.assertRaisesRegex(ValueError, "CARFAC feature shape"):
            validate_feature_config(bad_config)


if __name__ == "__main__":
    unittest.main()
