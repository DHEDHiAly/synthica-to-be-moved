"""
Main training script for the ICU trajectory disentangled model.

Run via:
    python src/train.py                         # default pipeline
    python src/train.py --mode full_experiment  # full experiment + reporting

Features:
  - Loads data/eicu_final_sequences_for_modeling.csv (hard rule)
  - Infers & validates schema programmatically
  - Baseline tuning harness: grid over hidden_dim, lr, dropout
    (+ n_layers, n_heads for transformer baselines)
  - Main model training with early stopping on validation AUROC
  - Multi-seed evaluation (seeds 42/43/44) for main model + best baseline
  - Ablation suite with standardized paper-ready names
  - s_t predictive sufficiency probe
  - Semi-grounded counterfactual sanity check
  - Domain generalization gap + relative drop metrics
  - Counterfactual proxy evaluation
  - Calibration (ECE) mandatory
  - Competitiveness flagging (ΔAUROC)
  - Loss grouped under: invariance / disentanglement / predictive modules
  - All results written to outputs/results.json
  - Publication-ready plots and tables via --mode full_experiment
train.py — Full training pipeline for the DisentangledICUModel and baselines.

Entry point
-----------
    python src/train.py [--config <path>] [--mode <main|baselines|all>]

Training procedure
------------------
1. Load & preprocess data.
2. Train DisentangledICUModel with adversarial + contrastive objectives.
   - Discriminators (hospital / treatment adversaries) trained separately
     on their own objective (no gradient reversal for discriminator step).
   - Encoder trained with gradient reversal to fool discriminators.
3. Train each baseline model.
4. Evaluate all models (in-distribution, out-of-hospital, counterfactual).
5. Run ablation studies.
6. Save full report to outputs/results.json.
"""

from __future__ import annotations

import argparse
import csv
import copy
import itertools
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

print("[DEBUG] CWD:", os.getcwd())

# Absolute base directory for repo-relative output paths
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add src/ to path when running from repo root
sys.path.insert(0, os.path.dirname(__file__))

from data import (
    load_raw,
    infer_schema,
    build_sequences,
    split_random,
    split_out_of_hospital,
    make_loaders,
    ICUDatasetConfig,
)
from model import DisentangledICUModel, EtOnlyModel
from baselines import build_baseline, BASELINE_REGISTRY
from losses import TotalLoss, reconstruction_loss, outcome_loss, irm_penalty
from eval import (
    evaluate_model,
    flag_competitiveness,
    run_counterfactual_proxy_evaluation,
    disentanglement_check,
    compute_ece,
    evaluate_st_predictive_probe,
    grounded_counterfactual_sanity,
    compute_domain_generalization_metrics,
)
from utils import set_seed, get_logger, EarlyStopping, save_checkpoint, SEED

logger = get_logger("synthica.train")
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------------------------------------------------------------------
# Ensure src/ is on the path when invoked as `python src/train.py`
# ---------------------------------------------------------------------------
_SRC = str(Path(__file__).parent.resolve())
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from data import load_and_prepare
from model import DisentangledICUModel, build_model
from baselines import BASELINE_REGISTRY, build_baseline
from losses import TotalLoss, AdversarialLoss, irm_penalty
from utils import (
    set_seed,
    setup_logging,
    get_device,
    save_checkpoint,
    load_checkpoint,
    EarlyStopping,
    safe_auroc,
    save_json,
)
from eval import (
    evaluate_in_distribution,
    evaluate_out_of_hospital,
    evaluate_counterfactual,
    monitor_disentanglement,
    generate_report,
    compute_st_probe_auroc,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CFG: Dict[str, Any] = {
    # Data
    "data_path": os.path.join(_BASE_DIR, "data", "eicu_final_sequences_for_modeling.csv"),
    "seq_len": 24,
    "max_seq_len": 48,
    "min_seq_len": 3,
    "batch_size": 64,
    "split_mode": "out_of_hospital",
    "seed": 42,
    # Model
    "hidden_dim": 128,
    "s_dim": 64,
    "e_dim": 32,
    "num_layers": 2,
    "enc_layers": 2,
    "enc_dropout": 0.1,
    "dropout": 0.1,
    "grl_alpha": 1.0,
    "grl_anneal": True,
    "grl_alpha_max": 1.0,
    "num_trt_classes": 2,
    "proj_dim": 64,
    # Training
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "max_epochs": 50,
    "patience": 10,
    "grad_clip": 1.0,
    # Losses
    "lambda_outcome": 1.0,
    "lambda_hospital_adv": 0.1,
    "lambda_hosp_adv": 0.1,
    "lambda_treatment_adv": 0.1,
    "lambda_trt_adv": 0.1,
    "lambda_contrastive": 0.05,
    "lambda_irm": 0.01,
    "disc_lr_multiplier": 2.0,
    "contrastive_noise_scale": 0.05,
    # Baseline tuning
    "run_baseline_tuning": True,
    "run_baselines": True,
    "baseline_tune_epochs": 20,
    "baseline_epochs": 30,
    "baseline_tune_patience": 5,
    "baseline_patience": 10,
    "baseline_hparam_search": False,
    "baseline_tune_trials": 3,
    "baseline_hidden_dim": 128,
    "baseline_lr": 3e-4,
    "baseline_hparam_hidden_dims": [64, 128, 256],
    "baseline_hparam_lrs": [1e-3, 3e-4, 1e-4],
    "baselines_to_run": [
        "erm", "dann", "domain_confusion", "irm",
        "crn", "gnet", "rmsn", "dcrn",
        "causal_transformer", "g_transformer", "mamba_cdsp",
    ],
    # Multi-seed evaluation
    "eval_seeds": [42, 43, 44],
    # Ablations
    "run_ablations": True,
    "ablation_epochs": 20,
    "monitor_disentanglement_interval": 10,
    # Counterfactual proxy eval
    "run_cf_eval": True,
    "cf_n_samples": 16,
    # Outputs (absolute paths)
    "output_dir": os.path.join(_BASE_DIR, "outputs"),
    "checkpoint_dir": os.path.join(_BASE_DIR, "outputs", "checkpoints"),
    "log_file": os.path.join(_BASE_DIR, "outputs", "train.log"),
    "results_path": os.path.join(_BASE_DIR, "outputs", "results.json"),
}

# Base hyperparameter search grid for all baselines
TUNE_GRID = {
    "hidden_dim": [64, 128, 256],
    "lr": [1e-3, 3e-4, 1e-4],
    "dropout": [0.0, 0.2, 0.5],
}

# Additional grid dimensions for transformer-based baselines
TRANSFORMER_BASELINES = {"causal_transformer", "g_transformer"}
TRANSFORMER_EXTRA_GRID = {
    "n_layers": [2, 4],
    "n_heads": [2, 4],
}

# Loss module grouping labels for structured logging
LOSS_GROUPS = {
    "predictive": ["recon", "outcome"],
    "invariance": ["hospital_adv", "irm"],
    "disentanglement": ["treatment_adv", "contrastive"],
}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: TotalLoss,
    device: torch.device,
    cfg: Dict,
    n_hospitals: int,
    monitor_collapse: bool = True,
) -> Dict[str, float]:
    model.train()
    total_losses: Dict[str, List[float]] = {}
    collapse_stats: List[Dict] = []

    for batch in loader:
        x = batch["x"].to(device)
        u = batch["u"].to(device)
        y_true = batch["outcome"].to(device)
        env_ids = batch["env_id"].to(device)

        optimizer.zero_grad()
        out = model(x, u, env_ids=env_ids)

        hosp_logits = out.get("hosp_logits")
        treat_logits = out.get("treat_logits")
        s_proj = out.get("s_proj")

        if treat_logits is not None:
            treat_labels = u.mean(dim=1).argmax(dim=-1).clamp(0, treat_logits.shape[-1] - 1)
        else:
            treat_labels = None

        total, losses = loss_fn(
            x_pred=out["x_pred"],
            x_true=out["x_true"],
            y_pred=out["y_pred"],
            y_true=y_true,
            hosp_logits=hosp_logits,
            hosp_labels=env_ids if hosp_logits is not None else None,
            treat_logits=treat_logits,
            treat_labels=treat_labels,
            s_i=s_proj,
            s_j=s_proj.roll(1, 0) if s_proj is not None else None,
            irm_logits=out["y_pred"] if cfg.get("lambda_irm", 0) > 0 else None,
            irm_labels=y_true if cfg.get("lambda_irm", 0) > 0 else None,
            irm_env_ids=env_ids if cfg.get("lambda_irm", 0) > 0 else None,
        )

        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
        optimizer.step()

        for k, v in losses.items():
            total_losses.setdefault(k, []).append(v.item() if isinstance(v, torch.Tensor) else v)

        if monitor_collapse and hasattr(model, "monitor_collapse"):
            with torch.no_grad():
                s_seq = out.get("s_seq")
                if s_seq is not None:
                    stats = model.monitor_collapse(s_seq.detach())
                    collapse_stats.append(stats)

    avg = {k: float(np.mean(v)) for k, v in total_losses.items()}

    if collapse_stats:
        avg["s_t_mean_var"] = float(np.mean([s["s_t_mean_var"] for s in collapse_stats]))
        avg["s_t_min_var"] = float(np.mean([s["s_t_min_var"] for s in collapse_stats]))
        if avg["s_t_mean_var"] < 0.01:
            logger.warning(
                "!!! COLLAPSE WARNING: s_t mean variance = %.6f !!!",
                avg["s_t_mean_var"],
            )

    _log_loss_groups(avg)
    return avg

# ---------------------------------------------------------------------------
# Paper-ready ablation name mapping
# ---------------------------------------------------------------------------

ABLATION_PAPER_NAMES: Dict[str, str] = {
    "no_adversaries": "w/o Adversarial",
    "no_contrastive": "w/o Contrastive",
    "minimal_e":      "w/o e_t",
    "no_invariant_s": "e_t only",
    "no_hosp_adv":    "w/o Hosp Adv",
    "no_trt_adv":     "w/o Trt Adv",
}


# ---------------------------------------------------------------------------
# GRL alpha schedule
# ---------------------------------------------------------------------------

def compute_grl_alpha(epoch: int, max_epochs: int, alpha_max: float = 1.0) -> float:
    """Anneal GRL alpha from 0 → alpha_max using a sigmoid schedule."""
    progress = epoch / max(max_epochs, 1)
    return float(alpha_max * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0))


# ---------------------------------------------------------------------------
# Discriminator update step
# ---------------------------------------------------------------------------

def update_discriminators(
    model: DisentangledICUModel,
    batch: Dict[str, torch.Tensor],
    disc_optim: torch.optim.Optimizer,
    adv_loss_fn: AdversarialLoss,
    device: torch.device,
    num_trt_classes: int,
) -> Dict[str, float]:
    """
    Separate discriminator update: train hospital and treatment adversaries
    with their own optimiser (no gradient reversal here — we detach s).
    """
    model.eval()  # freeze encoder norms
    disc_optim.zero_grad()

    x = batch["x"].to(device)
    u = batch["u"].to(device)
    mask = batch["mask"].to(device)
    hospital_id = batch["hospital_id"].to(device)

    with torch.no_grad():
        out = model(x, u, mask)
        s = out["s"]                                   # (B, T, s_dim)

    # Flatten time dimension
    B, T, _ = s.shape
    if mask is not None:
        valid = mask.view(-1)
        s_flat = s.view(B * T, -1)[valid].detach()
        h_flat = hospital_id.unsqueeze(1).expand(B, T).reshape(B * T)[valid]
        u_flat = u.view(B * T, -1)[valid]
    else:
        s_flat = s.view(B * T, -1).detach()
        h_flat = hospital_id.unsqueeze(1).expand(B, T).reshape(B * T)
        u_flat = u.view(B * T, -1)

    # Hospital adversary (no GRL — discriminator wants to classify correctly)
    hosp_logits = model.hosp_adversary.net(s_flat)
    l_hosp = adv_loss_fn(hosp_logits, h_flat)

    # Treatment adversary — binary: any treatment vs. no treatment
    trt_label = (u_flat.sum(dim=-1) > 0).long()
    trt_logits = model.trt_adversary.net(s_flat)
    l_trt = adv_loss_fn(trt_logits, trt_label)

    disc_loss = l_hosp + l_trt
    disc_loss.backward()
    disc_optim.step()
    model.train()

    return {
        "disc_hosp": l_hosp.item(),
        "disc_trt": l_trt.item(),
    }


# ---------------------------------------------------------------------------
# Main model training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: DisentangledICUModel,
    loader,
    optimizer: torch.optim.Optimizer,
    disc_optim: torch.optim.Optimizer,
    loss_fn: TotalLoss,
    adv_loss_fn: AdversarialLoss,
    device: torch.device,
    cfg: Dict[str, Any],
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_stats: Dict[str, float] = {}
    total_losses: Dict[str, List[float]] = {}
    collapse_stats: List[Dict] = []
    monitor_collapse = True
    n_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        u = batch["u"].to(device)
        y_true = batch["outcome"].to(device)
        env_ids = batch["env_id"].to(device)

        optimizer.zero_grad()
        out = model(x, u, env_ids=env_ids)

        hosp_logits = out.get("hosp_logits")
        treat_logits = out.get("treat_logits")
        s_proj = out.get("s_proj")

        # Build treatment labels for adversary (majority treatment per patient)
        if treat_logits is not None:
            treat_labels = u.mean(dim=1).argmax(dim=-1).clamp(0, treat_logits.shape[-1] - 1)
        else:
            treat_labels = None

        total, losses = loss_fn(
            x_pred=out["x_pred"],
            x_true=out["x_true"],
            y_pred=out["y_pred"],
            y_true=y_true,
            hosp_logits=hosp_logits,
            hosp_labels=env_ids if hosp_logits is not None else None,
            treat_logits=treat_logits,
            treat_labels=treat_labels,
            s_i=s_proj,
            s_j=s_proj.roll(1, 0) if s_proj is not None else None,
            irm_logits=out["y_pred"] if cfg.get("lambda_irm", 0) > 0 else None,
            irm_labels=y_true if cfg.get("lambda_irm", 0) > 0 else None,
            irm_env_ids=env_ids if cfg.get("lambda_irm", 0) > 0 else None,
        )

        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k, v in losses.items():
            total_losses.setdefault(k, []).append(v.item() if isinstance(v, torch.Tensor) else v)

        # Collapse monitoring
        if monitor_collapse and hasattr(model, "monitor_collapse"):
            with torch.no_grad():
                stats = model.monitor_collapse(out["s_seq"].detach())
                collapse_stats.append(stats)

    avg = {k: float(np.mean(v)) for k, v in total_losses.items()}

    if collapse_stats:
        avg["s_t_mean_var"] = float(np.mean([s["s_t_mean_var"] for s in collapse_stats]))
        avg["s_t_min_var"] = float(np.mean([s["s_t_min_var"] for s in collapse_stats]))
        if avg["s_t_mean_var"] < 0.01:
            logger.warning(
                "!!! COLLAPSE WARNING: s_t mean variance = %.6f — "
                "invariant latent may be collapsing to trivial representation !!!",
                avg["s_t_mean_var"],
            )

    # Log losses grouped by module for interpretability
    _log_loss_groups(avg)

    return avg


def _log_loss_groups(avg_losses: Dict[str, float]) -> None:
    """Log training losses grouped under invariance / disentanglement / predictive modules."""
    groups: Dict[str, Dict[str, float]] = {g: {} for g in LOSS_GROUPS}
    for key, val in avg_losses.items():
        for group, prefixes in LOSS_GROUPS.items():
            if any(key.startswith(p) for p in prefixes):
                groups[group][key] = val
                break

    for group, losses in groups.items():
        if losses:
            loss_str = "  ".join(f"{k}={v:.4f}" for k, v in losses.items())
            logger.debug("[%s module] %s", group.capitalize(), loss_str)


def _train_baseline_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Simple epoch for baselines (recon + outcome loss only)."""
    model.train()
    losses = []
    for batch in loader:
        x = batch["x"].to(device)
        u = batch["u"].to(device)
        y_true = batch["outcome"].to(device)

        optimizer.zero_grad()
        out = model(x, u)

        recon = reconstruction_loss(out["x_pred"], out["x_true"])
        out_loss = outcome_loss(out["y_pred"].squeeze(-1), y_true)
        loss = recon + out_loss

        # IRM baseline: add IRM penalty
        if hasattr(model, "_is_irm") or type(model).__name__ == "IRMBaseline":
            env_ids = batch["env_id"].to(device)
            irm_pen = irm_penalty(out["y_pred"], y_true, env_ids)
            loss = loss + 0.01 * irm_pen

        # DANN/DomainConfusion: domain loss
        if "domain_logits" in out:
            env_ids = batch["env_id"].to(device)
            n_domains = out.get("n_domains", out["domain_logits"].shape[-1])
            domain_loss = F.cross_entropy(out["domain_logits"], env_ids.clamp(0, n_domains - 1))
            loss = loss + 0.1 * domain_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Baseline tuning harness
# ---------------------------------------------------------------------------

def tune_baseline(
    name: str,
    input_dim: int,
    treatment_dim: int,
    n_hospitals: int,
    train_seqs: List[dict],
    val_seqs: List[dict],
    batch_size: int,
    device: torch.device,
    max_epochs: int = 20,
    patience: int = 5,
    k_trials: int = 3,
) -> Tuple[Dict, float, nn.Module]:
    """
    Grid search over hidden_dim x lr x dropout for a single baseline.
    Transformer baselines (CausalTransformer, GTransformer) additionally
    search over n_layers ∈ {2,4} and n_heads ∈ {2,4}.
    Runs k_trials with different seeds per config; reports best validation AUROC config.

    Returns: (best_config, best_val_auroc, best_model)
    """
    is_transformer = name.lower() in TRANSFORMER_BASELINES

    if is_transformer:
        base_configs = list(itertools.product(
            TUNE_GRID["hidden_dim"],
            TUNE_GRID["lr"],
            TUNE_GRID["dropout"],
            TRANSFORMER_EXTRA_GRID["n_layers"],
            TRANSFORMER_EXTRA_GRID["n_heads"],
        ))
        n_configs = len(base_configs)
    else:
        base_configs = list(itertools.product(
            TUNE_GRID["hidden_dim"],
            TUNE_GRID["lr"],
            TUNE_GRID["dropout"],
        ))
        n_configs = len(base_configs)

    logger.info("[Baseline Tune] %s — grid search over %d configs × %d trials",
                name, n_configs, k_trials)

    best_cfg: Optional[Dict] = None
    best_auroc: float = -1.0
    best_model_state: Optional[Dict] = None

    for combo in base_configs:
        if is_transformer:
            hidden_dim, lr, dropout, n_layers, n_heads = combo
            extra_kwargs: Dict[str, Any] = {"n_layers": n_layers, "n_heads": n_heads}
        else:
            hidden_dim, lr, dropout = combo
            extra_kwargs = {}

        for trial in range(k_trials):
            seed = SEED + trial * 1000 + abs(hash((name, hidden_dim, lr, dropout,
                                                     extra_kwargs.get("n_layers", 0),
                                                     extra_kwargs.get("n_heads", 0)))) % 1000
            set_seed(seed)

            try:
                model = build_baseline(name, input_dim, treatment_dim, n_hospitals,
                                       hidden_dim=hidden_dim, dropout=dropout, **extra_kwargs)
                model = model.to(device)
            except Exception as ex:
                logger.warning("[Baseline Tune] %s build failed: %s", name, ex)
                continue

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            train_loader, val_loader, _ = make_loaders(train_seqs, val_seqs, val_seqs,
                                                        batch_size=batch_size)

            stopper = EarlyStopping(patience=patience)
            for epoch in range(max_epochs):
                _train_baseline_epoch(model, train_loader, optimizer, device)
                val_metrics = evaluate_model(model, val_loader, device, tag=f"{name}_val")
                val_auroc = val_metrics["auroc"]
                if stopper.step(val_auroc, model.state_dict()):
                    break

            trial_auroc = stopper.best_score
            if trial_auroc > best_auroc:
                best_auroc = trial_auroc
                best_cfg = {"hidden_dim": hidden_dim, "lr": lr, "dropout": dropout,
                            "trial_seed": seed, **extra_kwargs}
                stopper.restore_best(model)
                best_model_state = copy.deepcopy(model.state_dict())

    logger.info("[Baseline Tune] %s — best config: %s  val_AUROC=%.4f", name, best_cfg, best_auroc)

    # Rebuild best model
    rebuild_kwargs = {}
    if best_cfg and is_transformer:
        rebuild_kwargs = {k: best_cfg[k] for k in ("n_layers", "n_heads") if k in best_cfg}
    best_model = build_baseline(name, input_dim, treatment_dim, n_hospitals,
                                 hidden_dim=best_cfg["hidden_dim"] if best_cfg else 128,
                                 dropout=best_cfg["dropout"] if best_cfg else 0.0,
                                 **rebuild_kwargs)
    best_model = best_model.to(device)
    if best_model_state:
        best_model.load_state_dict(best_model_state)

    return best_cfg or {}, best_auroc, best_model


# ---------------------------------------------------------------------------
# Main model training
# ---------------------------------------------------------------------------

def train_main_model(
    model: DisentangledICUModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict,
    device: torch.device,
    n_hospitals: int,
) -> Tuple[float, Dict]:
    """Train main model; return best val AUROC and per-epoch log."""
    loss_fn = TotalLoss(
        lambda_outcome=cfg["lambda_outcome"],
        lambda_hospital_adv=cfg["lambda_hospital_adv"],
        lambda_treatment_adv=cfg["lambda_treatment_adv"],
        lambda_contrastive=cfg["lambda_contrastive"],
        lambda_irm=cfg["lambda_irm"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    stopper = EarlyStopping(patience=cfg["patience"])
    epoch_log: Dict[str, List] = {"train_loss": [], "val_auroc": []}

    for epoch in range(cfg["max_epochs"]):
        t0 = time.time()
        train_stats = _train_one_epoch(model, train_loader, optimizer, loss_fn, device, cfg,
                                        n_hospitals=n_hospitals, monitor_collapse=True)
        val_metrics = evaluate_model(model, val_loader, device, tag="val")
        val_auroc = val_metrics["auroc"]

        scheduler.step(-val_auroc)  # negate because scheduler minimizes
        epoch_log["train_loss"].append(train_stats.get("total", 0.0))
        epoch_log["val_auroc"].append(val_auroc)

        logger.info(
            "[Epoch %02d/%.0f] loss=%.4f  val_AUROC=%.4f  s_var=%.4f  (%.1fs)",
            epoch + 1, cfg["max_epochs"],
            train_stats.get("total", 0.0),
            val_auroc,
            train_stats.get("s_t_mean_var", float("nan")),
            time.time() - t0,
        )

        if stopper.step(val_auroc, model.state_dict()):
            logger.info("Early stopping at epoch %d (best AUROC=%.4f)", epoch + 1, stopper.best_score)
            break

    stopper.restore_best(model)
    save_checkpoint(model, os.path.join(cfg["checkpoint_dir"], "main_model_best.pt"))
    return stopper.best_score, epoch_log


# ---------------------------------------------------------------------------
# Multi-seed evaluation
# ---------------------------------------------------------------------------

def run_multiseed_evaluation(
    input_dim: int,
    treatment_dim: int,
    n_hospitals: int,
    train_seqs: List[dict],
    val_seqs: List[dict],
    test_seqs: List[dict],
    cfg: Dict,
    device: torch.device,
    seeds: Optional[List[int]] = None,
    schema_hospital_enabled: bool = True,
) -> Dict[str, Any]:
    """
    Train main model with multiple seeds and report mean ± std AUROC.
    Required for statistical credibility at ICML.
    """
    if seeds is None:
        seeds = cfg.get("eval_seeds", [42, 43, 44])

    logger.info("=== MULTI-SEED EVALUATION (seeds=%s) ===", seeds)
    train_loader, val_loader, test_loader = make_loaders(
        train_seqs, val_seqs, test_seqs, batch_size=cfg["batch_size"]
    )

    auroc_scores: List[float] = []
    for seed in seeds:
        set_seed(seed)
        model = DisentangledICUModel(
            input_dim=input_dim,
            treatment_dim=treatment_dim,
            n_hospitals=max(2, n_hospitals),
            hidden_dim=cfg["hidden_dim"],
            s_dim=cfg["s_dim"],
            e_dim=cfg["e_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            grl_alpha=cfg["grl_alpha"],
            use_hospital_adv=schema_hospital_enabled,
            use_treatment_adv=True,
            use_contrastive=True,
            use_irm=schema_hospital_enabled,
        ).to(device)

        best_val_auroc, _ = train_main_model(model, train_loader, val_loader, cfg, device, n_hospitals)
        test_metrics = evaluate_model(model, test_loader, device, tag=f"seed{seed}_test")
        auroc_scores.append(test_metrics["auroc"])
        logger.info("[Multi-seed] seed=%d  test_AUROC=%.4f", seed, test_metrics["auroc"])

    mean_auroc = float(np.mean(auroc_scores))
    std_auroc = float(np.std(auroc_scores))
    logger.info(
        "[Multi-seed] AUROC = %.4f ± %.4f  (seeds=%s)",
        mean_auroc, std_auroc, seeds,
    )
    return {
        "seeds": seeds,
        "per_seed_auroc": {str(s): round(a, 4) for s, a in zip(seeds, auroc_scores)},
        "mean_auroc": round(mean_auroc, 4),
        "std_auroc": round(std_auroc, 4),
    }


# ---------------------------------------------------------------------------
# Ablation training  (standardized paper-ready names)
# ---------------------------------------------------------------------------

# Maps internal ablation key → paper display name
ABLATION_PAPER_NAMES: Dict[str, str] = {
    "no_adv_heads":    "w/o Adversarial",
    "no_hosp_adv":     "w/o Hospital Adversary",
    "no_treat_adv":    "w/o Treatment Adversary",
    "no_contrastive":  "w/o Contrastive",
    "no_irm":          "w/o IRM",
    "no_s_t_et_only":  "e_t only (no s_t)",
}


def run_ablations(
    input_dim: int,
    treatment_dim: int,
    n_hospitals: int,
    train_seqs: List[dict],
    val_seqs: List[dict],
    test_seqs: List[dict],
    cfg: Dict,
    device: torch.device,
) -> Dict[str, Dict]:
    """
    Run ablation experiments with paper-ready names:
      - w/o Adversarial           (both hospital + treatment adversary removed)
      - w/o Hospital Adversary
      - w/o Treatment Adversary
      - w/o Contrastive
      - w/o IRM
      - e_t only (no s_t)         ← KEY ABLATION: validates s_t signal
    """
    logger.info("=== ABLATION SUITE ===")
    results = {}

    # Ablation configs — key is standardised for table output
    ablation_configs = [
        ("no_adv_heads",   {"use_hospital_adv": False, "use_treatment_adv": False}),
        ("no_hosp_adv",    {"use_hospital_adv": False}),
        ("no_treat_adv",   {"use_treatment_adv": False}),
        ("no_contrastive", {"use_contrastive": False}),
        ("no_irm",         {"lambda_irm": 0.0}),
    ]

    train_loader, val_loader, test_loader = make_loaders(train_seqs, val_seqs, test_seqs,
                                                          batch_size=cfg["batch_size"])

    for abl_name, overrides in ablation_configs:
        display = ABLATION_PAPER_NAMES.get(abl_name, abl_name)
        logger.info("--- Ablation: %s (%s) ---", display, abl_name)
        abl_cfg = {**cfg, **overrides}
        set_seed(SEED)
        model = DisentangledICUModel(
            input_dim=input_dim,
            treatment_dim=treatment_dim,
            n_hospitals=max(2, n_hospitals),
            hidden_dim=abl_cfg.get("hidden_dim", cfg["hidden_dim"]),
            s_dim=cfg["s_dim"],
            e_dim=cfg["e_dim"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            grl_alpha=cfg["grl_alpha"],
            use_hospital_adv=abl_cfg.get("use_hospital_adv", True),
            use_treatment_adv=abl_cfg.get("use_treatment_adv", True),
            use_contrastive=abl_cfg.get("use_contrastive", True),
            use_irm=abl_cfg.get("lambda_irm", cfg["lambda_irm"]) > 0,
        ).to(device)

        best_val_auroc, _ = train_main_model(model, train_loader, val_loader, abl_cfg, device, n_hospitals)
        test_metrics = evaluate_model(model, test_loader, device, tag=f"ablation_{abl_name}_test")
        results[abl_name] = {
            "display_name": display,
            "val_auroc": best_val_auroc,
            **test_metrics,
        }

    # KEY ABLATION: no s_t / e_t-only model
    display_et = ABLATION_PAPER_NAMES.get("no_s_t_et_only", "e_t only (no s_t)")
    logger.info("--- KEY ABLATION: %s ---", display_et)
    set_seed(SEED)
    et_model = EtOnlyModel(
        input_dim=input_dim,
        treatment_dim=treatment_dim,
        hidden_dim=cfg["hidden_dim"],
        e_dim=cfg["e_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(et_model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    stopper = EarlyStopping(patience=cfg["patience"])

    for epoch in range(cfg["max_epochs"]):
        _train_baseline_epoch(et_model, train_loader, optimizer, device)
        val_m = evaluate_model(et_model, val_loader, device, tag="et_only_val")
        if stopper.step(val_m["auroc"], et_model.state_dict()):
            break

    stopper.restore_best(et_model)
    test_m = evaluate_model(et_model, test_loader, device, tag="no_s_t_test")
    results["no_s_t_et_only"] = {
        "display_name": display_et,
        "val_auroc": stopper.best_score,
        **test_m,
        "note": "KEY ABLATION: tests if s_t carries signal — s_t removed entirely",
    }
    logger.info("KEY ABLATION %s: test_AUROC=%.4f", display_et, test_m["auroc"])
    return results


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: DisentangledICUModel,
    loader,
    loss_fn: TotalLoss,
    device: torch.device,
    cfg: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """Returns (val_auroc, metrics_dict)."""
    from eval import evaluate_model

    result = evaluate_model(model, loader, device, is_disentangled=True)
    return result["metrics"]["auroc"], result["metrics"]


# ---------------------------------------------------------------------------
# Main model training loop
# ---------------------------------------------------------------------------

def train_model(
    model: DisentangledICUModel,
    train_loader,
    val_loader,
    cfg: Dict[str, Any],
    device: torch.device,
    max_epochs: int = 100,
    checkpoint_path: str = "outputs/checkpoints/best_main.pt",
) -> Dict[str, float]:
    """Full training loop for the disentangled model."""
    optimizer = AdamW(
        model.parameters(), lr=cfg.get("lr", 3e-4), weight_decay=cfg.get("weight_decay", 1e-5)
    )
    # Discriminator optimiser — only adversary parameters
    disc_params = (
        list(model.hosp_adversary.net.parameters())
        + list(model.trt_adversary.net.parameters())
    )
    disc_optim = AdamW(disc_params, lr=cfg.get("lr", 3e-4) * cfg.get("disc_lr_multiplier", 2.0))

    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(
        patience=cfg.get("patience", 15), mode="max", delta=1e-4
    )

    loss_fn = TotalLoss(
        lambda_outcome=cfg.get("lambda_outcome", 1.0),
        lambda_hosp_adv=cfg.get("lambda_hosp_adv", 0.1),
        lambda_trt_adv=cfg.get("lambda_trt_adv", 0.1),
        lambda_contrastive=cfg.get("lambda_contrastive", 0.05),
        lambda_irm=cfg.get("lambda_irm", 0.0),
    )
    adv_loss_fn = AdversarialLoss()

    best_metrics: Dict[str, float] = {}

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        # Anneal GRL alpha
        if cfg.get("grl_anneal", True):
            alpha = compute_grl_alpha(epoch, max_epochs, cfg.get("grl_alpha_max", 1.0))
            model.set_grl_alpha(alpha)

        train_stats = train_epoch(
            model, train_loader, optimizer, disc_optim, loss_fn, adv_loss_fn,
            device, cfg, epoch,
        )
        val_auroc, val_metrics = validate(model, val_loader, loss_fn, device, cfg)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d  [%.1fs]  train_loss=%.4f  val_AUROC=%.4f  val_AUPRC=%.4f",
            epoch, max_epochs, elapsed,
            train_stats.get("total", 0.0), val_auroc,
            val_metrics.get("auprc", float("nan")),
        )

        is_best = early_stopping.step(val_auroc)
        if is_best:
            best_metrics = val_metrics
            save_checkpoint(checkpoint_path, model, optimizer, epoch, val_metrics, cfg)
            logger.info("  [BEST] New best  AUROC=%.4f  saved to %s", val_auroc, checkpoint_path)

        if early_stopping.should_stop:
            logger.info("Early stopping triggered at epoch %d.", epoch)
            break

        # Disentanglement monitor (run every N epochs)
        monitor_interval = cfg.get("monitor_disentanglement_interval", 10)
        if monitor_interval > 0 and epoch % monitor_interval == 0:
            try:
                disent_stats = monitor_disentanglement(model, val_loader, device)
                best_metrics["disentanglement"] = disent_stats
            except Exception as exc:
                logger.debug("Disentanglement monitor skipped: %s", exc)

    # Reload best weights
    if Path(checkpoint_path).exists():
        load_checkpoint(checkpoint_path, model, device=device)
        logger.info("Loaded best checkpoint from %s", checkpoint_path)

    return best_metrics


# ---------------------------------------------------------------------------
# Baseline training loop
# ---------------------------------------------------------------------------

def train_baseline(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: Dict[str, Any],
    device: torch.device,
    name: str,
    max_epochs: int = 30,
    checkpoint_path: str = "outputs/checkpoints/best_baseline.pt",
    lr: Optional[float] = None,
) -> Dict[str, float]:
    """Generic training loop for baselines (outcome + reconstruction losses)."""
    effective_lr = lr if lr is not None else cfg.get("lr", 3e-4)
    optimizer = AdamW(model.parameters(), lr=effective_lr,
                      weight_decay=cfg.get("weight_decay", 1e-5))
    early_stopping = EarlyStopping(patience=cfg.get("baseline_patience", 10), mode="max")
    best_metrics: Dict[str, float] = {}

    is_irm = name == "irm"

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            u = batch["u"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y"].to(device)
            hospital_id = batch["hospital_id"].to(device)

            optimizer.zero_grad()
            out = model(x, u, mask)
            logit = out["outcome_logit"]
            x_pred = out["x_pred"]

            l_outcome = F.binary_cross_entropy_with_logits(logit, y.float())
            l_recon = F.mse_loss(x_pred[mask], x[mask]) if mask.any() else torch.tensor(0.0)
            loss = l_outcome + 0.1 * l_recon

            # Domain adversary losses for DANN / DomainConfusion
            if "domain_logit" in out and hospital_id is not None:
                l_domain = F.cross_entropy(out["domain_logit"], hospital_id)
                loss = loss + 0.1 * l_domain
            if "confusion_loss" in out:
                loss = loss + 0.1 * out["confusion_loss"]

            # IRM penalty
            if is_irm and cfg.get("lambda_irm", 1.0) > 0:
                if hasattr(model, "irm_scale") and "scaled_logit" in out:
                    irm_losses = []
                    for hid in hospital_id.unique():
                        h_mask = hospital_id == hid
                        if h_mask.sum() < 2:
                            continue
                        pen = irm_penalty(
                            out["scaled_logit"][h_mask], y[h_mask], model.irm_scale
                        )
                        irm_losses.append(pen)
                    if irm_losses:
                        loss = loss + cfg.get("lambda_irm", 1.0) * torch.stack(irm_losses).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                xv = batch["x"].to(device)
                uv = batch["u"].to(device)
                mv = batch["mask"].to(device)
                yv = batch["y"].to(device)
                out_v = model(xv, uv, mv)
                all_logits.append(out_v["outcome_logit"].cpu())
                all_labels.append(yv.cpu())

        logits_np = torch.cat(all_logits).numpy()
        labels_np = torch.cat(all_labels).numpy()
        import numpy as np
        probs = torch.sigmoid(torch.tensor(logits_np)).numpy()
        val_auroc = safe_auroc(labels_np, probs)

        logger.info("  [Baseline %s] Epoch %d  val_AUROC=%.4f", name, epoch, val_auroc)

        is_best = early_stopping.step(val_auroc)
        if is_best:
            best_metrics = {"auroc": val_auroc}
            save_checkpoint(checkpoint_path, model, optimizer, epoch, best_metrics, cfg)

        if early_stopping.should_stop:
            break

    if Path(checkpoint_path).exists():
        load_checkpoint(checkpoint_path, model, device=device)

    return best_metrics


# ---------------------------------------------------------------------------
# Baseline hyperparameter search
# ---------------------------------------------------------------------------

def tune_baseline_hparams(
    bl_name: str,
    train_loader,
    val_loader,
    cfg: Dict[str, Any],
    device: torch.device,
    checkpoint_dir: str,
) -> Tuple[nn.Module, Dict[str, float], Dict[str, Any]]:
    """
    Grid search over hidden_dim × lr for a single baseline.

    Grid (when baseline_hparam_search=True):
        hidden_dim ∈ cfg["baseline_hparam_hidden_dims"]  (default [64, 128, 256])
        lr         ∈ cfg["baseline_hparam_lrs"]          (default [1e-3, 3e-4, 1e-4])

    When baseline_hparam_search=False, uses the single values
    cfg["baseline_hidden_dim"] and cfg["baseline_lr"] (faster, for quick runs).

    Returns the best model, its val metrics, and the best config dict.
    """
    do_search = cfg.get("baseline_hparam_search", False)

    if do_search:
        hidden_dims = cfg.get("baseline_hparam_hidden_dims", [64, 128, 256])
        lrs = cfg.get("baseline_hparam_lrs", [1e-3, 3e-4, 1e-4])
    else:
        hidden_dims = [cfg.get("baseline_hidden_dim", 128)]
        lrs = [cfg.get("baseline_lr", 3e-4)]

    best_auroc = -1.0
    best_model = None
    best_metrics: Dict[str, float] = {}
    best_hparams: Dict[str, Any] = {}

    for hd in hidden_dims:
        for lr_val in lrs:
            trial_cfg = {**cfg, "hidden_dim": hd}
            trial_id = f"hd{hd}_lr{lr_val:.0e}"
            ckpt = f"{checkpoint_dir}/best_{bl_name}_{trial_id}.pt"

            try:
                model = build_baseline(
                    bl_name,
                    x_dim=cfg["num_features"],
                    u_dim=cfg["num_treatments"],
                    num_hospitals=cfg["num_hospitals"],
                    hidden_dim=hd,
                    device=device,
                )
                metrics = train_baseline(
                    model, train_loader, val_loader, trial_cfg, device,
                    name=bl_name,
                    max_epochs=cfg.get("baseline_epochs", 30),
                    checkpoint_path=ckpt,
                    lr=lr_val,
                )
                trial_auroc = metrics.get("auroc", 0.0)
                logger.info(
                    "  [Baseline %s] hd=%d lr=%.0e → AUROC=%.4f",
                    bl_name, hd, lr_val, trial_auroc,
                )
                if trial_auroc > best_auroc:
                    best_auroc = trial_auroc
                    best_model = model
                    best_metrics = {**metrics, "hidden_dim": hd, "lr": lr_val}
                    best_hparams = {"hidden_dim": hd, "lr": lr_val}
            except Exception as exc:
                logger.warning(
                    "  [Baseline %s] hd=%d lr=%.0e failed: %s", bl_name, hd, lr_val, exc
                )

    if best_model is None:
        raise RuntimeError(f"All hyperparameter trials failed for baseline {bl_name}")

    logger.info(
        "  [Baseline %s] Best config: %s  AUROC=%.4f",
        bl_name, best_hparams, best_auroc,
    )
    return best_model, best_metrics, best_hparams


# ---------------------------------------------------------------------------
# Ablation helper
# ---------------------------------------------------------------------------

def run_ablations_standalone(
    cfg: Dict[str, Any],
    train_seqs,
    val_seqs,
    test_seqs,
    hospital_id_map,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Train and evaluate ablated variants of the main model.

    Ablations
    ---------
    no_hosp_adv    : remove hospital adversary (lambda_hosp_adv=0)
    no_trt_adv     : remove treatment adversary (lambda_trt_adv=0)
    no_contrastive : remove contrastive loss (lambda_contrastive=0)
    no_adversaries : remove both adversarial heads
    minimal_e      : minimal e_dim=4 (near-constant environment branch)
    no_invariant_s : **KEY** ablation — outcome uses e_final only (not s_final).
                     Tests whether the invariant s_t branch carries real predictive
                     signal.  A large AUROC drop confirms s_t is non-trivial.
    """
    from data import get_dataloaders
    from eval import evaluate_in_distribution

    results = {}

    ablation_overrides = {
        "no_hosp_adv": {"lambda_hosp_adv": 0.0},
        "no_trt_adv": {"lambda_trt_adv": 0.0},
        "no_contrastive": {"lambda_contrastive": 0.0},
        "no_adversaries": {"lambda_hosp_adv": 0.0, "lambda_trt_adv": 0.0},
        "minimal_e": {"e_dim": 4},
        # Most critical ablation: route outcome through e_t only, bypassing s_t.
        # If performance drops, s_t is genuinely useful; if not, investigate collapse.
        "no_invariant_s": {
            "use_e_for_outcome": True,
            "lambda_hosp_adv": 0.0,
            "lambda_trt_adv": 0.0,
            "lambda_contrastive": 0.0,
        },
    }

    for abl_name, overrides in ablation_overrides.items():
        logger.info("=== Ablation: %s ===", abl_name)
        abl_cfg = {**cfg, **overrides}

        train_ld, val_ld, test_ld = get_dataloaders(
            train_seqs, val_seqs, test_seqs,
            batch_size=cfg["batch_size"],
            hospital_id_map=hospital_id_map,
        )

        model = build_model(abl_cfg, device)
        ckpt = f"{cfg['checkpoint_dir']}/ablation_{abl_name}_best.pt"
        train_model(
            model, train_ld, val_ld, abl_cfg, device,
            max_epochs=cfg.get("ablation_epochs", 20),
            checkpoint_path=ckpt,
        )
        res = evaluate_in_distribution(model, test_ld, device)
        results[abl_name] = res["metrics"]
        logger.info("Ablation %s results: %s", abl_name, results[abl_name])

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(cfg: Optional[Dict] = None) -> Dict[str, Any]:
    if cfg is None:
        cfg = copy.deepcopy(DEFAULT_CFG)

    set_seed(SEED)
    device = _get_device()
    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    logger.info("=== ICU Trajectory Modeling System ===")
    logger.info("Device: %s", device)

    # -----------------------------------------------------------------------
    # 1. Load and prepare data
    # -----------------------------------------------------------------------
    df = load_raw()
    schema = infer_schema(df)

    sequences, scaler = build_sequences(df, schema, seq_len=cfg["seq_len"])
    logger.info("Total sequences: %d", len(sequences))

    # Splits
    if schema.hospital_invariance_enabled:
        try:
            train_seqs, val_seqs, iid_test_seqs, ooh_test_seqs = split_out_of_hospital(sequences, df, schema)
            logger.info("Using out-of-hospital split: train=%d val=%d iid_test=%d ooh_test=%d",
                        len(train_seqs), len(val_seqs), len(iid_test_seqs), len(ooh_test_seqs))
        except Exception as ex:
            logger.warning("OOH split failed (%s) — falling back to random split.", ex)
            train_seqs, val_seqs, iid_test_seqs = split_random(sequences)
            ooh_test_seqs = []
    else:
        logger.warning("Hospital invariance disabled — using random split only.")
        train_seqs, val_seqs, iid_test_seqs = split_random(sequences)
        ooh_test_seqs = []

    train_loader, val_loader, test_loader = make_loaders(train_seqs, val_seqs, iid_test_seqs,
                                                          batch_size=cfg["batch_size"])
    ooh_loader = None
    if ooh_test_seqs:
        from data import ICUSequenceDataset
        ooh_loader = DataLoader(ICUSequenceDataset(ooh_test_seqs), batch_size=cfg["batch_size"], shuffle=False)

    # Derived dimensions
    input_dim = sequences[0]["x"].shape[-1]
    treatment_dim = sequences[0]["u"].shape[-1]
    n_hospitals = int(max(s["env_id"].item() for s in sequences)) + 1
    logger.info("input_dim=%d  treatment_dim=%d  n_hospitals=%d", input_dim, treatment_dim, n_hospitals)

    results: Dict[str, Any] = {
        "schema": {
            "input_dim": input_dim,
            "treatment_dim": treatment_dim,
            "n_hospitals": n_hospitals,
            "env_valid": schema.env_valid,
            "hospital_invariance_enabled": schema.hospital_invariance_enabled,
        }
    }

    # -----------------------------------------------------------------------
    # 2. Baseline tuning harness
    # -----------------------------------------------------------------------
    baseline_results: Dict[str, Dict] = {}
    best_baseline_configs: Dict[str, Dict] = {}

    if cfg.get("run_baseline_tuning", True):
        logger.info("=== BASELINE TUNING HARNESS ===")
        for bl_name in BASELINE_REGISTRY.keys():
            try:
                best_cfg_bl, best_val_auroc_bl, best_bl_model = tune_baseline(
                    name=bl_name,
                    input_dim=input_dim,
                    treatment_dim=treatment_dim,
                    n_hospitals=max(2, n_hospitals),
                    train_seqs=train_seqs,
                    val_seqs=val_seqs,
                    batch_size=cfg["batch_size"],
                    device=device,
                    max_epochs=cfg["baseline_tune_epochs"],
                    patience=cfg["baseline_tune_patience"],
                    k_trials=cfg["baseline_tune_trials"],
                )
                test_metrics_bl = evaluate_model(best_bl_model, test_loader, device,
                                                   tag=f"{bl_name}_test",
                                                   ece_save_path=os.path.join(
                                                       cfg["output_dir"], f"ece_{bl_name}.json"))
                baseline_results[bl_name] = {
                    "best_config": best_cfg_bl,
                    "val_auroc": round(best_val_auroc_bl, 4),
                    **test_metrics_bl,
                }
                best_baseline_configs[bl_name] = best_cfg_bl
                logger.info("[Baseline] %s: test_AUROC=%.4f  ECE=%.4f",
                            bl_name, test_metrics_bl["auroc"], test_metrics_bl["ece"])

                # OOH evaluation
                if ooh_loader:
                    ooh_metrics_bl = evaluate_model(best_bl_model, ooh_loader, device, tag=f"{bl_name}_ooh")
                    baseline_results[bl_name]["ooh_auroc"] = ooh_metrics_bl["auroc"]

            except Exception as ex:
                logger.warning("[Baseline] %s failed: %s", bl_name, ex)
                baseline_results[bl_name] = {"error": str(ex)}

        results["baselines"] = baseline_results
        results["best_baseline_configs"] = best_baseline_configs

        # Persist best configs
        cfg_path = os.path.join(cfg["output_dir"], "best_baseline_configs.json")
        with open(cfg_path, "w") as fh:
            json.dump(best_baseline_configs, fh, indent=2)
        logger.info("Best baseline configs saved → %s", cfg_path)
    else:
        logger.info("Baseline tuning skipped (run_baseline_tuning=False)")

    # -----------------------------------------------------------------------
    # 3. Train main model
    # -----------------------------------------------------------------------
    logger.info("=== TRAINING MAIN MODEL ===")
    set_seed(SEED)
    main_model = DisentangledICUModel(
        input_dim=input_dim,
        treatment_dim=treatment_dim,
        n_hospitals=max(2, n_hospitals),
        hidden_dim=cfg["hidden_dim"],
        s_dim=cfg["s_dim"],
        e_dim=cfg["e_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        grl_alpha=cfg["grl_alpha"],
        use_hospital_adv=schema.hospital_invariance_enabled,
        use_treatment_adv=True,
        use_contrastive=True,
        use_irm=schema.hospital_invariance_enabled,
    ).to(device)

    best_val_auroc, epoch_log = train_main_model(main_model, train_loader, val_loader, cfg, device, n_hospitals)

    # Test evaluation
    test_metrics = evaluate_model(
        main_model, test_loader, device, tag="main_iid_test",
        ece_save_path=os.path.join(cfg["output_dir"], "ece_main_model.json"),
    )
    results["main_model"] = {
        "val_auroc": round(best_val_auroc, 4),
        **test_metrics,
        "epoch_log": epoch_log,
    }

    # OOH test
    if ooh_loader:
        ooh_metrics = evaluate_model(main_model, ooh_loader, device, tag="main_ooh_test",
                                      ece_save_path=os.path.join(cfg["output_dir"], "ece_main_ooh.json"))
        results["main_model"]["ooh_metrics"] = ooh_metrics
        results["main_model"]["domain_shift_drop"] = round(
            test_metrics["auroc"] - ooh_metrics["auroc"], 4)

    # -----------------------------------------------------------------------
    # 4. Competitiveness flagging
    # -----------------------------------------------------------------------
    competitiveness = flag_competitiveness(test_metrics["auroc"], baseline_results)
    results["competitiveness"] = competitiveness

    # -----------------------------------------------------------------------
    # 5. Disentanglement check
    # -----------------------------------------------------------------------
    disent_result = disentanglement_check(main_model, test_loader, device, test_metrics)
    results["disentanglement_check"] = disent_result

    # -----------------------------------------------------------------------
    # 6. Counterfactual proxy evaluation
    # -----------------------------------------------------------------------
    if cfg.get("run_cf_eval", True):
        cf_results = run_counterfactual_proxy_evaluation(
            main_model, iid_test_seqs, device, n_samples=cfg["cf_n_samples"]
        )
        results["counterfactual_proxy_eval"] = cf_results

    # -----------------------------------------------------------------------
    # 7. Ablation suite
    # -----------------------------------------------------------------------
    if cfg.get("run_ablations", True):
        ablation_results = run_ablations(
            input_dim=input_dim,
            treatment_dim=treatment_dim,
            n_hospitals=n_hospitals,
            train_seqs=train_seqs,
            val_seqs=val_seqs,
            test_seqs=iid_test_seqs,
            cfg=cfg,
            device=device,
        )
        results["ablations"] = ablation_results

        # Log ablation summary with standardised names
        logger.info("=== ABLATION SUMMARY ===")
        main_auroc = test_metrics["auroc"]
        for abl_name, abl_res in ablation_results.items():
            display = abl_res.get("display_name", abl_name)
            delta = abl_res.get("auroc", 0.0) - main_auroc
            logger.info("  %-35s AUROC=%.4f  Δ=%.4f  %s",
                        display, abl_res.get("auroc", 0.0), delta,
                        abl_res.get("note", ""))

    # -----------------------------------------------------------------------
    # 8. s_t predictive sufficiency probe
    # -----------------------------------------------------------------------
    logger.info("=== s_t PREDICTIVE SUFFICIENCY PROBE ===")
    st_probe_result = evaluate_st_predictive_probe(
        main_model, test_loader, device, full_model_auroc=test_metrics["auroc"]
    )
    results["st_predictive_probe"] = st_probe_result

    # -----------------------------------------------------------------------
    # 9. Semi-grounded counterfactual sanity check
    # -----------------------------------------------------------------------
    if cfg.get("run_cf_eval", True):
        grounded_cf_result = grounded_counterfactual_sanity(
            main_model, iid_test_seqs, device, n_samples=cfg.get("cf_n_samples", 16) * 4
        )
        results.setdefault("counterfactual_proxy_eval", {})["grounded_sanity"] = grounded_cf_result

    # -----------------------------------------------------------------------
    # 10. Domain generalization gap metrics
    # -----------------------------------------------------------------------
    all_models_for_gen: Dict[str, Dict] = {}
    if "main_model" in results and results["main_model"].get("ooh_metrics"):
        all_models_for_gen["Full Model"] = {
            "auroc": results["main_model"]["auroc"],
            "ooh_auroc": results["main_model"]["ooh_metrics"]["auroc"],
        }
    for bl_name, bl_res in baseline_results.items():
        if isinstance(bl_res, dict) and "auroc" in bl_res and "ooh_auroc" in bl_res:
            all_models_for_gen[bl_name] = bl_res

    if all_models_for_gen:
        domain_gen_metrics = compute_domain_generalization_metrics(all_models_for_gen)
        results["domain_generalization"] = domain_gen_metrics

    # -----------------------------------------------------------------------
    # 11. Final summary
    # -----------------------------------------------------------------------
    logger.info("=== FINAL RESULTS SUMMARY ===")
    logger.info("Main model:  AUROC=%.4f  AUPRC=%.4f  Acc=%.4f  ECE=%.4f",
                test_metrics["auroc"], test_metrics["auprc"],
                test_metrics["accuracy"], test_metrics["ece"])
    if competitiveness.get("delta_auroc") is not None:
        logger.info("ΔAUROC vs best baseline (%s): %.4f",
                    competitiveness["best_baseline"], competitiveness["delta_auroc"])

    # Save results
    out_path = os.path.join(cfg["output_dir"], "results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Full results saved → %s", out_path)

    return results


# ---------------------------------------------------------------------------
# Full experiment pipeline (--mode full_experiment)
# ---------------------------------------------------------------------------

def full_experiment(cfg: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Full experiment pipeline: trains all models, runs all evaluations,
    runs multi-seed stability check, and generates publication-ready
    plots + tables.

    Run via:
        python src/train.py --mode full_experiment
    """
    if cfg is None:
        cfg = copy.deepcopy(DEFAULT_CFG)

    logger.info("=== FULL EXPERIMENT PIPELINE ===")

    # Step 1: Run main pipeline (baselines + main model + ablations + CF eval)
    results = main(cfg)

    # Step 2: Multi-seed stability evaluation
    # Reload data for re-training under different seeds
    logger.info("=== STEP 2: MULTI-SEED STABILITY ===")
    df = load_raw()
    schema = infer_schema(df)
    sequences, _ = build_sequences(df, schema, seq_len=cfg["seq_len"])

    if schema.hospital_invariance_enabled:
        try:
            train_seqs, val_seqs, iid_test_seqs, _ = split_out_of_hospital(sequences, df, schema)
        except Exception:
            train_seqs, val_seqs, iid_test_seqs = split_random(sequences)
    else:
        train_seqs, val_seqs, iid_test_seqs = split_random(sequences)

    input_dim = sequences[0]["x"].shape[-1]
    treatment_dim = sequences[0]["u"].shape[-1]
    n_hospitals = int(max(s["env_id"].item() for s in sequences)) + 1
    device = _get_device()

    stability = run_multiseed_evaluation(
        input_dim=input_dim,
        treatment_dim=treatment_dim,
        n_hospitals=n_hospitals,
        train_seqs=train_seqs,
        val_seqs=val_seqs,
        test_seqs=iid_test_seqs,
        cfg=cfg,
        device=device,
        seeds=cfg.get("eval_seeds", [42, 43, 44]),
        schema_hospital_enabled=schema.hospital_invariance_enabled,
    )
    results["stability"] = stability

    # Step 3: Generate plots + tables
    logger.info("=== STEP 3: GENERATING PLOTS + TABLES ===")
    from reporting import generate_all_reports
    generate_all_reports(
        results=results,
        output_dir=cfg["output_dir"],
    )

    # Re-save updated results with stability data
    out_path = os.path.join(cfg["output_dir"], "results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Updated results saved → %s", out_path)

    logger.info("=== FULL EXPERIMENT COMPLETE ===")
    logger.info("Outputs → %s/", cfg["output_dir"])
    logger.info("  results.json, metrics.csv")
    logger.info("  plots/  tables/  logs/")
    if results.get("stability"):
        logger.info("Stability: mean AUROC = %.4f ± %.4f",
                    results["stability"]["mean_auroc"],
                    results["stability"]["std_auroc"])

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICU Trajectory Modeling")
    parser.add_argument(
        "--mode",
        choices=["train", "full_experiment"],
        default="train",
        help=(
            "Execution mode: "
            "'train' runs the standard pipeline (default); "
            "'full_experiment' adds multi-seed stability + publication plots/tables."
        ),
    )
    args = parser.parse_args()

    if args.mode == "full_experiment":
        full_experiment()
    else:
        main()
def main(cfg: Optional[Dict[str, Any]] = None) -> None:
    if cfg is None:
        cfg = DEFAULT_CFG.copy()

    # Setup
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    setup_logging("INFO", cfg.get("log_file"))
    set_seed(cfg["seed"])
    device = get_device()
    logger.info("Device: %s", device)
    logger.info("Configuration: %s", cfg)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    logger.info("Loading data…")
    data = load_and_prepare(
        max_seq_len=cfg["max_seq_len"],
        min_seq_len=cfg["min_seq_len"],
        split_mode=cfg["split_mode"],
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
    )

    cfg["num_features"] = data["num_features"]
    cfg["num_treatments"] = data["num_treatments"]
    cfg["num_hospitals"] = data["num_hospitals"]
    cfg["hospital_id_map"] = data["hospital_id_map"]
    hospital_valid = data.get("hospital_valid", True)

    logger.info(
        "Data ready: %d features, %d treatments, %d hospitals (hospital_valid=%s)",
        cfg["num_features"], cfg["num_treatments"], cfg["num_hospitals"], hospital_valid,
    )
    if not hospital_valid:
        logger.warning(
            "Hospital column is not valid for cross-hospital generalisation. "
            "OOH experiments will run but results should be interpreted with caution."
        )
    save_json(
        {k: v for k, v in cfg.items() if k != "hospital_id_map"},
        f"{cfg['output_dir']}/config.json",
    )

    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    ooh_test_loader = data.get("ooh_test_loader")
    ooh_train_loader = data.get("ooh_train_loader")
    ooh_val_loader = data.get("ooh_val_loader")

    # ------------------------------------------------------------------
    # Main model
    # ------------------------------------------------------------------
    logger.info("Building DisentangledICUModel…")
    main_model = build_model(cfg, device)
    logger.info(
        "Model parameters: %d",
        sum(p.numel() for p in main_model.parameters() if p.requires_grad),
    )

    ckpt_main = f"{cfg['checkpoint_dir']}/best_main.pt"
    logger.info("Training main model…")
    train_model(
        main_model, train_loader, val_loader, cfg, device,
        max_epochs=cfg["max_epochs"],
        checkpoint_path=ckpt_main,
    )

    # Evaluate main model (in-distribution)
    indist = evaluate_in_distribution(main_model, test_loader, device)
    logger.info("Main model in-dist metrics: %s", indist["metrics"])

    # Evaluate main model (out-of-hospital)
    ooh_result = None
    if ooh_test_loader is not None:
        # Also train on OOH splits for comparison
        if ooh_train_loader is not None:
            logger.info("Training main model on OOH train split…")
            ooh_main_model = build_model(cfg, device)
            train_model(
                ooh_main_model, ooh_train_loader, ooh_val_loader, cfg, device,
                max_epochs=cfg["max_epochs"],
                checkpoint_path=f"{cfg['checkpoint_dir']}/best_main_ooh.pt",
            )
            ooh_result = evaluate_out_of_hospital(
                ooh_main_model, train_loader, ooh_test_loader, device
            )
        else:
            ooh_result = evaluate_out_of_hospital(
                main_model, test_loader, ooh_test_loader, device
            )
        logger.info("Main model OOH results: %s", ooh_result)

    # ------------------------------------------------------------------
    # Baselines — with optional hyperparameter search
    # ------------------------------------------------------------------
    trained_baselines: Dict[str, nn.Module] = {}
    baseline_best_hparams: Dict[str, Any] = {}

    if cfg.get("run_baselines", True):
        logger.info(
            "Training baselines (hparam_search=%s)…",
            cfg.get("baseline_hparam_search", False),
        )
        for bl_name in cfg.get("baselines_to_run", list(BASELINE_REGISTRY.keys())):
            logger.info("  → %s", bl_name)
            try:
                bl_model, bl_metrics, bl_hparams = tune_baseline_hparams(
                    bl_name,
                    train_loader,
                    val_loader,
                    cfg,
                    device,
                    cfg["checkpoint_dir"],
                )
                trained_baselines[bl_name] = bl_model
                baseline_best_hparams[bl_name] = bl_hparams
            except Exception as exc:
                logger.error("Failed to train baseline %s: %s", bl_name, exc, exc_info=True)

    # ------------------------------------------------------------------
    # Ablations
    # ------------------------------------------------------------------
    ablation_results: Dict[str, Any] = {}
    if cfg.get("run_ablations", True):
        logger.info("Running ablation studies…")
        ablation_results = run_ablations_standalone(
            cfg,
            data["train_seqs"],
            data["val_seqs"],
            data["test_seqs"],
            data["hospital_id_map"],
            device,
        )
        logger.info("Ablation results: %s", ablation_results)

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------
    logger.info("Generating full report…")
    report = generate_report(
        main_model=main_model,
        baselines=trained_baselines,
        test_loader=test_loader,
        ooh_test_loader=ooh_test_loader,
        device=device,
        output_path=cfg["results_path"],
    )

    # Augment with OOH training result, ablations and hparam search results
    if ooh_result:
        report["main_model_ooh_dedicated"] = ooh_result
    report["ablations"] = ablation_results
    report["baseline_best_hparams"] = baseline_best_hparams
    report["hospital_valid"] = hospital_valid

    save_json(report, cfg["results_path"])
    print(f"[DEBUG] Saved results to {cfg['results_path']}")
    logger.info("=" * 60)
    logger.info("DONE. Results saved to %s", cfg["results_path"])
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Generate plots and tables
    # ------------------------------------------------------------------
    try:
        generate_plots(report, main_model, trained_baselines, ablation_results,
                       test_loader, device, cfg["output_dir"])
        logger.info("Plots saved to %s/plots/", cfg["output_dir"])
        print(f"[DEBUG] Saved plots to {cfg['output_dir']}/plots/")
    except Exception as exc:
        logger.warning("Plot generation failed: %s", exc)

    try:
        generate_tables(report, ablation_results, cfg["output_dir"])
        logger.info("Tables saved to %s/tables/", cfg["output_dir"])
        print(f"[DEBUG] Saved tables to {cfg['output_dir']}/tables/")
    except Exception as exc:
        logger.warning("Table generation failed: %s", exc)

    try:
        generate_metrics_csv(report, ablation_results, cfg["output_dir"])
        metrics_path = os.path.join(cfg["output_dir"], "metrics.csv")
        logger.info("metrics.csv saved to %s/", cfg["output_dir"])
        print(f"[DEBUG] Saved metrics to {metrics_path}")
    except Exception as exc:
        logger.warning("metrics.csv generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    _print_final_summary(report, indist, ooh_result, ablation_results,
                         baseline_best_hparams)


def _print_final_summary(
    report: Dict[str, Any],
    indist: Dict[str, Any],
    ooh_result: Optional[Dict[str, Any]],
    ablation_results: Dict[str, Any],
    baseline_best_hparams: Dict[str, Any],
) -> None:
    """Print a concise final summary matching the required format."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    m = indist["metrics"]
    print(
        f"Main model (in-distribution):"
        f"  AUROC={m.get('auroc', float('nan')):.4f}"
        f"  AUPRC={m.get('auprc', float('nan')):.4f}"
        f"  ACC={m.get('accuracy', float('nan')):.4f}"
        f"  ECE={m.get('ece', float('nan')):.4f}"
    )
    if ooh_result:
        ooh_m = ooh_result.get("out_of_hospital", {})
        in_m = ooh_result.get("in_dist", {})
        gap = ooh_result.get("domain_shift", {}).get("auroc_drop", float("nan"))
        print(
            f"Main model (out-of-hospital):"
            f"  In-Dist AUROC={in_m.get('auroc', float('nan')):.4f}"
            f"  OOD AUROC={ooh_m.get('auroc', float('nan')):.4f}"
            f"  Generalization Gap={gap:.4f}"
            f"  ECE={ooh_m.get('ece', float('nan')):.4f}"
        )

    # ΔAUROC vs best baseline
    auroc_cmp = report.get("auroc_comparison", {})
    if auroc_cmp:
        status = auroc_cmp.get("competitive_status", "N/A")
        delta = auroc_cmp.get("delta_auroc", float("nan"))
        sign = "+" if (not math.isnan(delta) and delta >= 0) else ""
        print(
            f"\nAUROC competitiveness [{status}]:"
            f"  main={auroc_cmp.get('main_model_auroc', float('nan')):.4f}"
            f"  best_baseline={auroc_cmp.get('best_baseline_name')} "
            f"({auroc_cmp.get('best_baseline_auroc', float('nan')):.4f})"
            f"  ΔAUROC={sign}{delta:.4f}"
        )

    # s_t probe AUROC
    st_probe = report.get("st_probe_auroc", float("nan"))
    _ST_PROBE_THRESHOLD = 0.60
    st_status = "VALID" if (st_probe == st_probe and st_probe >= _ST_PROBE_THRESHOLD) else "LOW"
    print(f"\ns_t Probe AUROC: {st_probe:.4f} → {st_status}")

    print(f"\nBaselines:")
    for bname, bm in report.get("baselines", {}).items():
        if not bname.endswith("_ooh"):
            hparams = baseline_best_hparams.get(bname, {})
            print(
                f"  {bname:25s}: AUROC={bm.get('auroc', float('nan')):.4f}"
                f"  ECE={bm.get('ece', float('nan')):.4f}"
                + (f"  [hd={hparams.get('hidden_dim')} lr={hparams.get('lr'):.0e}]"
                   if hparams else "")
            )

    if ablation_results:
        print(f"\nAblations:")
        for aname, am in ablation_results.items():
            paper_name = ABLATION_PAPER_NAMES.get(aname, aname)
            flag = "← KEY: tests if s_t carries signal" if aname == "no_invariant_s" else ""
            print(
                f"  {paper_name:25s}: AUROC={am.get('auroc', float('nan')):.4f}"
                f"  ECE={am.get('ece', float('nan')):.4f}  {flag}"
            )

    disent = report.get("disentanglement_monitor", {})
    if disent:
        collapse = "[WARN] COLLAPSE DETECTED" if disent.get("collapse_warning") else "[OK]"
        print(
            f"\nDisentanglement monitor: s_var={disent.get('s_var_mean', float('nan')):.4e}"
            f"  recon_degradation={disent.get('recon_degradation_frac', float('nan')):.2%}"
            f"  {collapse}"
        )

    print("=" * 60)


# ---------------------------------------------------------------------------
# Plot generation (matplotlib only)
# ---------------------------------------------------------------------------

def generate_plots(
    report: Dict[str, Any],
    main_model: nn.Module,
    trained_baselines: Dict[str, nn.Module],
    ablation_results: Dict[str, Any],
    test_loader,
    device: torch.device,
    output_dir: str = "outputs",
) -> None:
    """Generate all required plots using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    from sklearn.calibration import calibration_curve as sklearn_cal_curve

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Collect main model probs/labels
    main_probs = np.array(report.get("_main_model_probs", []))
    main_labels = np.array(report.get("_main_model_labels", []))

    # Identify best baseline
    auroc_cmp = report.get("auroc_comparison", {})
    best_bl_name = auroc_cmp.get("best_baseline_name", "")
    best_bl_probs = np.array(report.get("_baseline_probs", {}).get(best_bl_name, []))

    # ------------------------------------------------------------------
    # 1. AUROC Comparison bar chart
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    names, aurocs, colors = [], [], []
    for bname, bm in report.get("baselines", {}).items():
        if "_ooh" in bname:
            continue
        names.append(bname)
        aurocs.append(bm.get("auroc", float("nan")))
        colors.append("#f4a261" if bname == best_bl_name else "#adb5bd")
    # Add main model
    main_auroc = report.get("main_model_indist", {}).get("auroc", float("nan"))
    names.append("DisentangledICU\n(Main)")
    aurocs.append(main_auroc)
    colors.append("#2a9d8f")

    bars = ax.barh(names, aurocs, color=colors)
    ax.set_xlabel("AUROC")
    ax.set_title("AUROC Comparison: Baselines vs Main Model")
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, label="Random")
    for bar, val in zip(bars, aurocs):
        if not math.isnan(val):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.4, 1.05)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2a9d8f", label="Main Model"),
        Patch(facecolor="#f4a261", label="Best Baseline"),
        Patch(facecolor="#adb5bd", label="Other Baselines"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    fig.tight_layout()
    fig.savefig(str(plots_dir / "auroc_comparison.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Precision-Recall Curve
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    if len(main_probs) > 0 and len(np.unique(main_labels)) > 1:
        prec, rec, _ = precision_recall_curve(main_labels, main_probs)
        main_auprc = report.get("main_model_indist", {}).get("auprc", float("nan"))
        ax.plot(rec, prec, color="#2a9d8f", lw=2,
                label=f"Main Model (AUPRC={main_auprc:.3f})")
    if len(best_bl_probs) > 0 and len(np.unique(main_labels)) > 1:
        try:
            prec_bl, rec_bl, _ = precision_recall_curve(main_labels, best_bl_probs)
            bl_auprc = report.get("baselines", {}).get(best_bl_name, {}).get("auprc", float("nan"))
            ax.plot(rec_bl, prec_bl, color="#f4a261", lw=2, linestyle="--",
                    label=f"{best_bl_name} (AUPRC={bl_auprc:.3f})")
        except Exception:
            pass
    # Baseline random
    pos_rate = main_labels.mean() if len(main_labels) > 0 else 0.5
    ax.axhline(y=pos_rate, color="gray", linestyle=":", lw=1, label=f"Random (={pos_rate:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(str(plots_dir / "pr_curve.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Calibration Curve
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    if len(main_probs) > 0 and len(np.unique(main_labels)) > 1:
        try:
            frac_pos, mean_pred = sklearn_cal_curve(main_labels, main_probs, n_bins=10)
            main_ece = report.get("main_model_indist", {}).get("ece", float("nan"))
            ax.plot(mean_pred, frac_pos, "s-", color="#2a9d8f", lw=2,
                    label=f"Main Model (ECE={main_ece:.3f})")
        except Exception:
            pass
    if len(best_bl_probs) > 0 and len(np.unique(main_labels)) > 1:
        try:
            frac_pos_bl, mean_pred_bl = sklearn_cal_curve(main_labels, best_bl_probs, n_bins=10)
            bl_ece = report.get("baselines", {}).get(best_bl_name, {}).get("ece", float("nan"))
            ax.plot(mean_pred_bl, frac_pos_bl, "o--", color="#f4a261", lw=2,
                    label=f"{best_bl_name} (ECE={bl_ece:.3f})")
        except Exception:
            pass
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(str(plots_dir / "calibration_curve.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Domain Generalization Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    model_names_dg, indist_aurocs, ood_aurocs, gaps = [], [], [], []

    main_ooh = report.get("main_model_ooh", {})
    if main_ooh:
        in_a = main_ooh.get("in_dist", {}).get("auroc", float("nan"))
        ood_a = main_ooh.get("out_of_hospital", {}).get("auroc", float("nan"))
        model_names_dg.append("Main Model")
        indist_aurocs.append(in_a)
        ood_aurocs.append(ood_a)
        gaps.append(in_a - ood_a if (not math.isnan(in_a) and not math.isnan(ood_a)) else float("nan"))

    for bname, bdata in report.get("baselines", {}).items():
        if "_ooh" not in bname:
            continue
        bl_base = bname.replace("_ooh", "")
        in_a = bdata.get("in_dist", {}).get("auroc", float("nan"))
        ood_a = bdata.get("out_of_hospital", {}).get("auroc", float("nan"))
        model_names_dg.append(bl_base)
        indist_aurocs.append(in_a)
        ood_aurocs.append(ood_a)
        gaps.append(in_a - ood_a if (not math.isnan(in_a) and not math.isnan(ood_a)) else float("nan"))

    if model_names_dg:
        x = np.arange(len(model_names_dg))
        width = 0.35
        bars_in = ax.bar(x - width / 2, indist_aurocs, width, label="In-Dist", color="#2a9d8f")
        bars_ood = ax.bar(x + width / 2, ood_aurocs, width, label="OOD", color="#e76f51")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names_dg, rotation=30, ha="right")
        ax.set_ylabel("AUROC")
        ax.set_title("Domain Generalization: In-Distribution vs Out-of-Hospital")
        ax.set_ylim(0.3, 1.05)
        ax.legend()
        # Annotate gaps
        for i, g in enumerate(gaps):
            if not math.isnan(g):  # not nan
                ax.annotate(f"Gap={g:.3f}", xy=(i, max(indist_aurocs[i], ood_aurocs[i]) + 0.01),
                            ha="center", fontsize=7, color="darkred")
        fig.tight_layout()
    else:
        ax.text(0.5, 0.5, "No OOH evaluation data available",
                ha="center", va="center", transform=ax.transAxes)
    fig.savefig(str(plots_dir / "domain_generalization.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Ablation Study Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    main_auroc_val = report.get("main_model_indist", {}).get("auroc", float("nan"))
    abl_names_plot = ["Full Model"]
    abl_aurocs_plot = [main_auroc_val]
    abl_colors_plot = ["#2a9d8f"]

    for abl_key, abl_m in ablation_results.items():
        paper_name = ABLATION_PAPER_NAMES.get(abl_key, abl_key)
        abl_names_plot.append(paper_name)
        abl_aurocs_plot.append(abl_m.get("auroc", float("nan")))
        abl_colors_plot.append("#e9c46a" if abl_key == "no_invariant_s" else "#adb5bd")

    bars = ax.barh(abl_names_plot, abl_aurocs_plot, color=abl_colors_plot)
    ax.set_xlabel("AUROC")
    ax.set_title("Ablation Study")
    if not math.isnan(main_auroc_val):
        ax.axvline(x=main_auroc_val, color="#2a9d8f", linestyle="--", lw=1.5,
                   label=f"Full Model ({main_auroc_val:.3f})")
    for bar, val in zip(bars, abl_aurocs_plot):
        if not math.isnan(val):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.3, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(plots_dir / "ablation_study.png"), dpi=120)
    plt.close(fig)

    logger.info("Plots saved: %s", [str(p) for p in plots_dir.glob("*.png")])


# ---------------------------------------------------------------------------
# Table generation (paper-ready CSV tables)
# ---------------------------------------------------------------------------

def generate_tables(
    report: Dict[str, Any],
    ablation_results: Dict[str, Any],
    output_dir: str = "outputs",
) -> None:
    """Generate paper-ready CSV tables."""
    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    auroc_cmp = report.get("auroc_comparison", {})
    main_m = report.get("main_model_indist", {})
    main_ooh = report.get("main_model_ooh", {})

    def _fmt(v: Any, fmt: str = ".4f") -> str:
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return "N/A"

    # ------------------------------------------------------------------
    # 1. Main Results Table
    # ------------------------------------------------------------------
    main_in_auroc = _fmt(main_m.get("auroc"))
    main_ooh_auroc = _fmt(main_ooh.get("out_of_hospital", {}).get("auroc") if main_ooh else None)
    main_gap = _fmt(main_ooh.get("domain_shift", {}).get("auroc_drop") if main_ooh else None)
    delta_auroc = auroc_cmp.get("delta_auroc", float("nan"))
    delta_sign = "+" if (not math.isnan(delta_auroc) and delta_auroc >= 0) else ""

    with open(str(tables_dir / "main_results_table.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "AUROC", "AUPRC", "ECE", "In-Dist AUROC",
                         "OOD AUROC", "Gen Gap", "ΔAUROC vs Best Baseline",
                         "Competitive Status", "s_t Probe AUROC"])
        st_probe = report.get("st_probe_auroc", float("nan"))
        writer.writerow([
            "DisentangledICU (Main)",
            main_in_auroc,
            _fmt(main_m.get("auprc")),
            _fmt(main_m.get("ece")),
            main_in_auroc,
            main_ooh_auroc,
            main_gap,
            f"{delta_sign}{_fmt(delta_auroc)}",
            auroc_cmp.get("competitive_status", "N/A"),
            _fmt(st_probe),
        ])
        writer.writerow([])
        writer.writerow([f"Best Baseline: {auroc_cmp.get('best_baseline_name', 'N/A')}",
                         _fmt(auroc_cmp.get("best_baseline_auroc"))])

    # ------------------------------------------------------------------
    # 2. Baseline Table
    # ------------------------------------------------------------------
    with open(str(tables_dir / "baseline_table.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "AUROC", "AUPRC", "Accuracy", "ECE", "OOD AUROC", "Gen Gap"])
        for bname, bm in report.get("baselines", {}).items():
            if "_ooh" in bname:
                continue
            ooh_key = f"{bname}_ooh"
            bl_ooh = report.get("baselines", {}).get(ooh_key, {})
            ood_auroc = _fmt(bl_ooh.get("out_of_hospital", {}).get("auroc") if bl_ooh else None)
            gen_gap = _fmt(bl_ooh.get("domain_shift", {}).get("auroc_drop") if bl_ooh else None)
            best_marker = " *" if bname == auroc_cmp.get("best_baseline_name") else ""
            writer.writerow([
                f"{bname}{best_marker}",
                _fmt(bm.get("auroc")),
                _fmt(bm.get("auprc")),
                _fmt(bm.get("accuracy")),
                _fmt(bm.get("ece")),
                ood_auroc,
                gen_gap,
            ])
        writer.writerow([])
        writer.writerow(["* = best baseline"])

    # ------------------------------------------------------------------
    # 3. Ablation Table
    # ------------------------------------------------------------------
    with open(str(tables_dir / "ablation_table.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "AUROC", "AUPRC", "Accuracy", "ECE"])
        # Full model first
        writer.writerow([
            "Full Model",
            _fmt(main_m.get("auroc")),
            _fmt(main_m.get("auprc")),
            _fmt(main_m.get("accuracy")),
            _fmt(main_m.get("ece")),
        ])
        # Ablations with paper names
        abl_order = ["no_adversaries", "no_contrastive", "minimal_e", "no_invariant_s",
                     "no_hosp_adv", "no_trt_adv"]
        for abl_key in abl_order:
            abl_m = ablation_results.get(abl_key)
            if abl_m is None:
                continue
            paper_name = ABLATION_PAPER_NAMES.get(abl_key, abl_key)
            writer.writerow([
                paper_name,
                _fmt(abl_m.get("auroc")),
                _fmt(abl_m.get("auprc")),
                _fmt(abl_m.get("accuracy")),
                _fmt(abl_m.get("ece")),
            ])
        # Any remaining ablations not in the standard order
        for abl_key, abl_m in ablation_results.items():
            if abl_key not in abl_order:
                paper_name = ABLATION_PAPER_NAMES.get(abl_key, abl_key)
                writer.writerow([
                    paper_name,
                    _fmt(abl_m.get("auroc")),
                    _fmt(abl_m.get("auprc")),
                    _fmt(abl_m.get("accuracy")),
                    _fmt(abl_m.get("ece")),
                ])

    logger.info("Tables saved to %s", tables_dir)


# ---------------------------------------------------------------------------
# Metrics CSV generation
# ---------------------------------------------------------------------------

def generate_metrics_csv(
    report: Dict[str, Any],
    ablation_results: Dict[str, Any],
    output_dir: str = "outputs",
) -> None:
    """Save all metrics in a flat metrics.csv file."""
    out_path = Path(output_dir) / "metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    main_m = report.get("main_model_indist", {})
    main_ooh = report.get("main_model_ooh", {})
    auroc_cmp = report.get("auroc_comparison", {})
    st_probe = report.get("st_probe_auroc", float("nan"))

    in_dist_auroc = main_m.get("auroc", float("nan"))
    ood_auroc = (main_ooh.get("out_of_hospital", {}).get("auroc", float("nan"))
                 if main_ooh else float("nan"))
    gen_gap = (main_ooh.get("domain_shift", {}).get("auroc_drop", float("nan"))
               if main_ooh else float("nan"))
    delta_auroc = auroc_cmp.get("delta_auroc", float("nan"))

    rows.append({
        "model": "main_model",
        "auroc": in_dist_auroc,
        "auprc": main_m.get("auprc", float("nan")),
        "accuracy": main_m.get("accuracy", float("nan")),
        "ece": main_m.get("ece", float("nan")),
        "in_dist_auroc": in_dist_auroc,
        "ood_auroc": ood_auroc,
        "generalization_gap": gen_gap,
        "delta_auroc_vs_best_baseline": delta_auroc,
        "competitive_status": auroc_cmp.get("competitive_status", "N/A"),
        "st_probe_auroc": st_probe,
    })

    for bname, bm in report.get("baselines", {}).items():
        if "_ooh" in bname:
            continue
        bl_ooh = report.get("baselines", {}).get(f"{bname}_ooh", {})
        b_ood = (bl_ooh.get("out_of_hospital", {}).get("auroc", float("nan"))
                 if bl_ooh else float("nan"))
        b_gap = (bl_ooh.get("domain_shift", {}).get("auroc_drop", float("nan"))
                 if bl_ooh else float("nan"))
        rows.append({
            "model": bname,
            "auroc": bm.get("auroc", float("nan")),
            "auprc": bm.get("auprc", float("nan")),
            "accuracy": bm.get("accuracy", float("nan")),
            "ece": bm.get("ece", float("nan")),
            "in_dist_auroc": bm.get("auroc", float("nan")),
            "ood_auroc": b_ood,
            "generalization_gap": b_gap,
            "delta_auroc_vs_best_baseline": float("nan"),
            "competitive_status": "N/A",
            "st_probe_auroc": float("nan"),
        })

    for abl_key, abl_m in ablation_results.items():
        paper_name = ABLATION_PAPER_NAMES.get(abl_key, abl_key)
        rows.append({
            "model": f"ablation_{abl_key}",
            "auroc": abl_m.get("auroc", float("nan")),
            "auprc": abl_m.get("auprc", float("nan")),
            "accuracy": abl_m.get("accuracy", float("nan")),
            "ece": abl_m.get("ece", float("nan")),
            "in_dist_auroc": abl_m.get("auroc", float("nan")),
            "ood_auroc": float("nan"),
            "generalization_gap": float("nan"),
            "delta_auroc_vs_best_baseline": float("nan"),
            "competitive_status": "N/A",
            "st_probe_auroc": float("nan"),
        })

    fieldnames = ["model", "auroc", "auprc", "accuracy", "ece",
                  "in_dist_auroc", "ood_auroc", "generalization_gap",
                  "delta_auroc_vs_best_baseline", "competitive_status", "st_probe_auroc"]
    with open(str(out_path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Metrics CSV saved to %s", out_path)


# ---------------------------------------------------------------------------
# Full experiment: multi-seed run
# ---------------------------------------------------------------------------

def run_full_experiment(cfg: Dict[str, Any]) -> None:
    """
    Run the full experiment pipeline with multi-seed evaluation.

    Procedure:
    1. Run the complete pipeline (baselines + main model + ablations) with seed=42.
    2. Run the main model only with seeds 43 and 44 for stability assessment.
    3. Report mean ± std of main model AUROC across seeds.
    4. Generate all output files (plots, tables, metrics.csv).
    """
    seeds = [42, 43, 44]

    logger.info("=" * 60)
    logger.info("FULL EXPERIMENT: multi-seed evaluation (seeds=%s)", seeds)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Seed 42 — full pipeline (baselines + main + ablations)
    # ------------------------------------------------------------------
    seed42_cfg = {**cfg, "seed": seeds[0]}
    logger.info("--- Seed %d: full pipeline ---", seeds[0])
    main(seed42_cfg)

    # ------------------------------------------------------------------
    # Seeds 43, 44 — main model only (no baselines, no ablations)
    # ------------------------------------------------------------------
    seed_aurocs: List[float] = []

    # Load seed-42 result to get seed-42 AUROC
    import json
    results_path = cfg.get("results_path", os.path.join(_BASE_DIR, "outputs", "results.json"))
    try:
        with open(results_path) as f:
            seed42_report = json.load(f)
        seed42_auroc = seed42_report.get("main_model_indist", {}).get("auroc", float("nan"))
        seed_aurocs.append(seed42_auroc)
        logger.info("Seed %d main model AUROC = %.4f", seeds[0], seed42_auroc)
    except Exception as exc:
        logger.warning("Could not load seed-42 results: %s", exc)
        seed42_auroc = float("nan")

    for seed in seeds[1:]:
        logger.info("--- Seed %d: main model only ---", seed)
        seed_cfg = {
            **cfg,
            "seed": seed,
            "run_baselines": False,
            "run_ablations": False,
            "results_path": f"{cfg['output_dir']}/results_seed{seed}.json",
            "checkpoint_dir": f"{cfg['checkpoint_dir']}/seed{seed}",
            "log_file": f"{cfg['output_dir']}/train_seed{seed}.log",
        }
        Path(seed_cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
        try:
            main(seed_cfg)
            with open(seed_cfg["results_path"]) as f:
                seed_report = json.load(f)
            s_auroc = seed_report.get("main_model_indist", {}).get("auroc", float("nan"))
            seed_aurocs.append(s_auroc)
            logger.info("Seed %d main model AUROC = %.4f", seed, s_auroc)
        except Exception as exc:
            logger.error("Seed %d run failed: %s", seed, exc, exc_info=True)

    # ------------------------------------------------------------------
    # Multi-seed summary
    # ------------------------------------------------------------------
    valid_aurocs = [a for a in seed_aurocs if not math.isnan(a)]  # filter nan
    if valid_aurocs:
        mean_auroc = float(np.mean(valid_aurocs))
        std_auroc = float(np.std(valid_aurocs))
        logger.info(
            "Multi-seed AUROC: %.4f ± %.4f  (seeds=%s, n=%d)",
            mean_auroc, std_auroc, seeds[:len(valid_aurocs)], len(valid_aurocs),
        )
        high_variance = std_auroc > 0.02
        if high_variance:
            logger.warning(
                "[WARN] High variance across seeds: std=%.4f > 0.02. "
                "Results may not be stable/publishable.",
                std_auroc,
            )
    else:
        mean_auroc, std_auroc = float("nan"), float("nan")

    # Save multi-seed summary
    seed_summary = {
        "seeds": seeds[:len(seed_aurocs)],
        "seed_aurocs": seed_aurocs,
        "mean_auroc": mean_auroc,
        "std_auroc": std_auroc,
        "high_variance_warning": bool(std_auroc > 0.02),
    }
    save_json(seed_summary, f"{cfg['output_dir']}/multi_seed_summary.json")
    print(f"[DEBUG] Saved multi-seed summary to {cfg['output_dir']}/multi_seed_summary.json")

    # Append to main results.json
    try:
        with open(results_path) as f:
            full_report = json.load(f)
        full_report["multi_seed"] = seed_summary
        save_json(full_report, results_path)
    except Exception as exc:
        logger.warning("Could not update results.json with multi-seed summary: %s", exc)

    # ------------------------------------------------------------------
    # Final summary print + directory dump (Step 5)
    # ------------------------------------------------------------------
    print("\n[DEBUG] Final output directory contents:")
    for root, dirs, files in os.walk(cfg["output_dir"]):
        for name in files:
            print(os.path.join(root, name))

    if not os.listdir(cfg["output_dir"]):
        raise RuntimeError("No output files were generated. Pipeline failed silently.")

    print("\n" + "=" * 60)
    print("FULL EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Seeds evaluated: {seeds[:len(seed_aurocs)]}")
    print(f"Main Model AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
    for s, a in zip(seeds, seed_aurocs):
        print(f"  Seed {s}: AUROC = {a:.4f}" if not math.isnan(a) else f"  Seed {s}: AUROC = N/A")
    if valid_aurocs and std_auroc > 0.02:
        print("[WARN] High variance detected — check model stability.")
    print(f"\nOutputs written to: {cfg['output_dir']}/")
    print(f"  results.json, metrics.csv, multi_seed_summary.json")
    print(f"  plots/: auroc_comparison, pr_curve, calibration_curve, "
          f"domain_generalization, ablation_study")
    print(f"  tables/: main_results_table, baseline_table, ablation_table")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train ICU trajectory model")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to JSON config file (overrides defaults)"
    )
    parser.add_argument(
        "--mode", choices=["main", "baselines", "all", "full_experiment"], default="all",
        help=(
            "What to train: 'main' trains only the main model (no baselines, no ablations); "
            "'baselines' trains baselines + main model (no ablations); "
            "'all' runs the complete single-seed pipeline (baselines + main model + ablations); "
            "'full_experiment' runs the full multi-seed pipeline (seeds=42,43,44) with "
            "all baselines + ablations on seed-42, main model on remaining seeds, and "
            "generates all output files (plots, tables, metrics.csv)."
        )
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split_mode", choices=["random", "out_of_hospital"], default=None)
    parser.add_argument(
        "--no_ablations", action="store_true",
        help="Skip ablation studies"
    )
    parser.add_argument(
        "--no_baselines", action="store_true",
        help="Skip baselines"
    )
    parser.add_argument(
        "--hparam_search", action="store_true",
        help="Enable baseline hyperparameter grid search (hidden_dim × lr)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"[DEBUG] Running mode: {args.mode}")
    cfg = DEFAULT_CFG.copy()

    if args.config:
        from utils import load_json
        loaded = load_json(args.config)
        cfg.update(loaded)

    # CLI overrides
    if args.epochs is not None:
        cfg["max_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.output_dir is not None:
        cfg["output_dir"] = os.path.abspath(args.output_dir)
        cfg["checkpoint_dir"] = os.path.join(cfg["output_dir"], "checkpoints")
        cfg["log_file"] = os.path.join(cfg["output_dir"], "train.log")
        cfg["results_path"] = os.path.join(cfg["output_dir"], "results.json")
    if args.split_mode is not None:
        cfg["split_mode"] = args.split_mode
    if args.no_ablations:
        cfg["run_ablations"] = False
    if args.no_baselines:
        cfg["run_baselines"] = False
    if args.hparam_search:
        cfg["baseline_hparam_search"] = True

    # Force output directories to exist before any I/O
    OUTPUT_DIR = cfg["output_dir"]
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    if args.mode == "full_experiment":
        run_full_experiment(cfg)
    elif args.mode == "main":
        cfg["run_baselines"] = False
        cfg["run_ablations"] = False
        main(cfg)
    elif args.mode == "baselines":
        cfg["run_ablations"] = False
        main(cfg)
    else:
        main(cfg)
