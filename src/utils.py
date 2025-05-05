import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def _default(o):
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    raise TypeError(f"{type(o)} is not JSON serializable")


def device():
    if not _HAS_TORCH:
        return None
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def running_mean(x, window: int = 50):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < window:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[window:] - c[:-window]) / window


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
