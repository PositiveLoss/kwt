import sys

import yaml

from utils.device import resolve_device
from utils.precision import resolve_precision
from utils.types import Config


def get_config(
    config_file: str, device: str | None = None, precision: str | None = None
) -> Config:
    """Reads settings from config file.

    Args:
        config_file (str): YAML config file.

    Returns:
        dict: Dict containing settings.
    """

    with open(config_file, "r") as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)

    resolved_device = resolve_device(device or base_config["exp"].get("device", "auto"))
    resolved_precision = resolve_precision(
        precision or base_config["hparams"].get("precision", "float32")
    )
    base_config["exp"]["device"] = resolved_device
    base_config["hparams"]["device"] = resolved_device
    base_config["hparams"]["precision"] = resolved_precision

    return base_config


if __name__ == "__main__":
    config = get_config(sys.argv[1])
    print("Using settings:\n", yaml.dump(config))
