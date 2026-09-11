from __future__ import annotations

import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
RUNS_DIR = ROOT / "runs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_run_name(prefix: str) -> str:
    return f"{prefix}_{timestamp_tag()}"


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_std(values: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = arr.copy()
    out[out < floor] = 1.0
    return out


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true_centered = y_true - y_true.mean()
    y_pred_centered = y_pred - y_pred.mean()
    denom = math.sqrt(float((y_true_centered**2).sum()) * float((y_pred_centered**2).sum()))
    return 0.0 if denom < 1e-12 else float((y_true_centered * y_pred_centered).sum() / denom)


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_rank = pd.Series(np.asarray(y_true)).rank(method="average").to_numpy(dtype=np.float64)
    pred_rank = pd.Series(np.asarray(y_pred)).rank(method="average").to_numpy(dtype=np.float64)
    return pearson_r(true_rank, pred_rank)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "pearson_r": pearson_r(y_true, y_pred),
        "spearman_rho": spearman_rho(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }
