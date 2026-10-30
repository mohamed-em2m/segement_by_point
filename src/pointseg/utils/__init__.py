"""Small shared utilities: seeding, logging and run metadata."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and torch (including all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # cuDNN determinism is best-effort; some kernels remain non-deterministic.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker a distinct, reproducible seed."""
    base = torch.initial_seed() % (2**32)
    seed = (base + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str = "pointseg", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    # Do not let the record bubble up to a root handler (e.g. Jupyter's), which
    # would print every line twice.
    logger.propagate = False
    logger.setLevel(level)
    return logger


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # pragma: no cover - only when git is unavailable
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return False


def run_metadata() -> dict[str, Any]:
    """Library versions and environment information stored with each run."""
    import cv2
    import matplotlib

    try:
        import segmentation_models_pytorch as smp

        smp_version = smp.__version__
    except Exception:  # pragma: no cover
        smp_version = "unknown"

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _module_version("torchvision"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "matplotlib": matplotlib.__version__,
        "segmentation_models_pytorch": smp_version,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }


def _module_version(name: str) -> str:
    try:
        module = __import__(name)
        return getattr(module, "__version__", "unknown")
    except Exception:  # pragma: no cover
        return "unknown"


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")
