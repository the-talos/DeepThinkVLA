"""Device selection for the inference entry points (XPU, then CUDA, then CPU).

An explicit device always wins over detection and is never swapped for another backend.
"""

from typing import Optional, Union

import torch

DeviceLike = Union[str, torch.device, None]
_ACCELERATORS = ("xpu", "cuda")


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in this torch build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def default_device_type() -> str:
    """Best available backend: ``xpu``, then ``cuda``, then ``cpu``."""
    for device_type in _ACCELERATORS:
        if is_available(device_type):
            return device_type
    return "cpu"


def default_device(index: int = 0) -> torch.device:
    """Device of the best available backend (``cuda:0`` on a CUDA host, as before)."""
    device_type = default_device_type()
    return torch.device("cpu") if device_type == "cpu" else torch.device(device_type, index)


def resolve_device(device: DeviceLike = None) -> torch.device:
    """Resolve ``None`` to the detected default; an explicit device is honoured, a bare ``xpu`` or ``cuda`` becomes index 0."""
    if device is None or device == "":
        return default_device()
    device = torch.device(device)
    if device.type in _ACCELERATORS and device.index is None:
        return torch.device(device.type, 0)
    return device


def manual_seed_all(seed: int, device_type: Optional[str] = None) -> None:
    """Seed the accelerator RNGs of the given (or detected) backend; no-op on CPU."""
    device_type = device_type or default_device_type()
    if device_type in _ACCELERATORS and is_available(device_type):
        getattr(torch, device_type).manual_seed_all(seed)
