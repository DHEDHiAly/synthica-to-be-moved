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
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CFG: Dict[str, Any] = {
    # Data
    "seq_len": 24,
    "batch_size": 64,
    # Model
    "hidden_dim": 128,
    "s_dim": 64,
    "e_dim": 64,
    "num_layers": 2,
    "dropout": 0.1,
    "grl_alpha": 1.0,
    # Training
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "max_epochs": 50,
    "patience": 10,
    # Losses
    "lambda_outcome": 1.0,
    "lambda_hospital_adv": 0.1,
    "lambda_treatment_adv": 0.1,
    "lambda_contrastive": 0.05,
    "lambda_irm": 0.01,
    # Baseline tuning
    "run_baseline_tuning": True,
    "baseline_tune_epochs": 20,
    "baseline_tune_patience": 5,
    "baseline_tune_trials": 3,  # k random seeds per config
    # Multi-seed evaluation
    "eval_seeds": [42, 43, 44],
    # Ablations
    "run_ablations": True,
    # Counterfactual proxy eval
    "run_cf_eval": True,
    "cf_n_samples": 16,
    # Outputs
    "output_dir": "outputs",
    "checkpoint_dir": "outputs/checkpoints",
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
