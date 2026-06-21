import torch


def is_mps_available() -> bool:
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device

    if device in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if resolved.type == "mps" and not is_mps_available():
        raise RuntimeError("MPS was requested but is not available.")

    return resolved
