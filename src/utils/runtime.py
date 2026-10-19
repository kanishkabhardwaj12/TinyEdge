import json
import platform
import random
import sys
from datetime import datetime, timezone

import numpy as np
import torch


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(config):
    threads = config.get("runtime", {}).get("cpu_threads")
    if threads:
        torch.set_num_threads(int(threads))


def available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            devices.append("mps")
    return devices


def resolve_device(requested):
    if requested == "auto":
        return torch.device(available_devices()[-1])
    device = torch.device(requested)
    if requested != "cpu" and requested not in available_devices():
        raise RuntimeError(
            f"Requested device '{requested}' is unavailable. "
            f"Available devices: {available_devices()}"
        )
    return device


def environment_metadata(config, split_sizes, device):
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": config["training"]["seed"],
        "config": config,
        "dataset_split_sizes": split_sizes,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "torchvision": _package_version("torchvision"),
            "numpy": np.__version__,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "execution_device": str(device),
        "available_devices": available_devices(),
        "cpu_threads": torch.get_num_threads(),
    }


def _package_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unavailable"


def save_json(data, path):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, default=str)
