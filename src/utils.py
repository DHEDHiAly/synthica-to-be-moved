"""
utils.py — Utilities: logging, seeding, GradientReversalLayer, checkpoint I/O, metric helpers.
"""

import os
import random
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Configure root logger with console (and optional file) handler."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers: list = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, log_level.upper()), format=fmt, handlers=handlers)
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set all RNG seeds for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (Ganin et al., JMLR 2016)
# ---------------------------------------------------------------------------

class _GRLFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (alpha,) = ctx.saved_tensors
        return -alpha.item() * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps the GRL function; alpha controls reversal strength."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRLFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = alpha


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    cfg: Optional[Dict[str, Any]] = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "cfg": cfg or {},
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC, returning 0.5 when only one class is present."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute PR-AUC via average precision, returning nan when undefined."""
    from sklearn.metrics import average_precision_score

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


def compute_all_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score

    y_pred = (y_score >= threshold).astype(int)
    return {
        "auroc": safe_auroc(y_true, y_score),
        "auprc": safe_auprc(y_true, y_score),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "ece": calibration_error(y_true, y_score),
    }


# ---------------------------------------------------------------------------
# JSON serialisation (handles numpy types)
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):  # type: ignore[override]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 10, mode: str = "max", delta: float = 1e-4) -> None:
        assert mode in ("max", "min")
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Returns True if this is a new best, False otherwise."""
        if self.best is None:
            self.best = value
            return True
        improved = (
            value > self.best + self.delta
            if self.mode == "max"
            else value < self.best - self.delta
        )
        if improved:
            self.best = value
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False
