"""
eval.py — Comprehensive evaluation for ICU trajectory models.

Functions
---------
evaluate_model          — Generic evaluation of any model on a DataLoader
evaluate_in_distribution — Evaluate on random test split
evaluate_out_of_hospital — Evaluate on held-out hospital test set
evaluate_counterfactual  — Measure trajectory divergence & outcome sensitivity
run_ablations            — Systematically ablate model components
generate_report          — Aggregate all results into a final JSON report
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import compute_all_metrics, safe_auroc, calibration_error, save_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_disentangled: bool = True,
) -> Dict[str, Any]:
    """
    Run `model` on all batches in `loader` and collect outcome predictions.

    Parameters
    ----------
    model          : Any model with a `forward(x, u, mask)` or
                     `get_outcome_logit(x, u, mask)` method.
    loader         : DataLoader yielding dicts with keys x, u, mask, y, hospital_id.
    device         : Computation device.
    is_disentangled: Whether to call the DisentangledICUModel API.

    Returns
    -------
    Dict with 'metrics' (AUROC, AUPRC, accuracy, ECE) and raw arrays.
    """
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    all_recon: List[float] = []
    all_hospital_ids: List[torch.Tensor] = []

    for batch in loader:
        x = batch["x"].to(device)
        u = batch["u"].to(device)
        mask = batch["mask"].to(device)
        y = batch["y"].to(device)
        hid = batch["hospital_id"].to(device)

        if is_disentangled:
            out = model(x, u, mask)
            logit = out["outcome_logit"]
            x_pred = out["x_pred"]
        else:
            # Baseline API
            if hasattr(model, "get_outcome_logit"):
                logit = model.get_outcome_logit(x, u, mask)
            else:
                out = model(x, u, mask)
                logit = out["outcome_logit"]
            x_pred = None

        all_logits.append(logit.cpu())
        all_labels.append(y.cpu())
        all_hospital_ids.append(hid.cpu())

        if x_pred is not None:
            recon = torch.nn.functional.mse_loss(
                x_pred[mask], x[mask], reduction="mean"
            ).item()
            all_recon.append(recon)

    logits_np = torch.cat(all_logits).numpy()
    labels_np = torch.cat(all_labels).numpy()
    probs_np = torch.sigmoid(torch.tensor(logits_np)).numpy()

    metrics = compute_all_metrics(labels_np, probs_np)
    if all_recon:
        metrics["recon_mse"] = float(np.mean(all_recon))

    return {
        "metrics": metrics,
        "logits": logits_np,
        "probs": probs_np,
        "labels": labels_np,
        "hospital_ids": torch.cat(all_hospital_ids).numpy(),
    }


# ---------------------------------------------------------------------------
# In-distribution evaluation
# ---------------------------------------------------------------------------

def evaluate_in_distribution(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    is_disentangled: bool = True,
) -> Dict[str, Any]:
    logger.info("=== In-distribution evaluation ===")
    result = evaluate_model(model, test_loader, device, is_disentangled)
    logger.info("In-distribution metrics: %s", result["metrics"])
    return result


# ---------------------------------------------------------------------------
# Out-of-hospital evaluation
# ---------------------------------------------------------------------------

def evaluate_out_of_hospital(
    model: nn.Module,
    indist_test_loader: DataLoader,
    ooh_test_loader: DataLoader,
    device: torch.device,
    is_disentangled: bool = True,
) -> Dict[str, Any]:
    logger.info("=== Out-of-hospital evaluation ===")
    indist = evaluate_model(model, indist_test_loader, device, is_disentangled)
    ooh = evaluate_model(model, ooh_test_loader, device, is_disentangled)

    # Domain shift degradation
    auroc_drop = indist["metrics"]["auroc"] - ooh["metrics"]["auroc"]
    auprc_drop = indist["metrics"]["auprc"] - ooh["metrics"]["auprc"]
    logger.info(
        "AUROC: in-dist=%.4f  OOH=%.4f  drop=%.4f",
        indist["metrics"]["auroc"],
        ooh["metrics"]["auroc"],
        auroc_drop,
    )
    return {
        "in_dist": indist["metrics"],
        "out_of_hospital": ooh["metrics"],
        "domain_shift": {
            "auroc_drop": float(auroc_drop),
            "auprc_drop": float(auprc_drop),
        },
    }


# ---------------------------------------------------------------------------
# Counterfactual evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_counterfactual(
    model: nn.Module,  # Must be DisentangledICUModel
    loader: DataLoader,
    device: torch.device,
    cf_strategy: str = "zero_treatment",
) -> Dict[str, Any]:
    """
    Simulate counterfactual trajectories and measure:
      - Trajectory divergence  (mean L2 distance between factual / CF trajectories)
      - Outcome sensitivity    (absolute change in predicted mortality)

    Parameters
    ----------
    cf_strategy : How to construct counterfactual treatment:
        "zero_treatment"  — set all treatments to zero
        "max_treatment"   — set all treatments to 1
        "flip_treatment"  — flip binary treatments
    """
    logger.info("=== Counterfactual evaluation (strategy=%s) ===", cf_strategy)
    model.eval()

    traj_divergences: List[float] = []
    outcome_sensitivities: List[float] = []

    for batch in loader:
        x = batch["x"].to(device)
        u = batch["u"].to(device)
        mask = batch["mask"].to(device)

        # Construct counterfactual treatment
        if cf_strategy == "zero_treatment":
            u_cf = torch.zeros_like(u)
        elif cf_strategy == "max_treatment":
            u_cf = torch.ones_like(u)
        elif cf_strategy == "flip_treatment":
            u_cf = 1.0 - u.clamp(0, 1)
        else:
            raise ValueError(f"Unknown cf_strategy: {cf_strategy}")

        # Factual prediction
        out_fact = model(x, u, mask)
        x_fact = out_fact["x_pred"]
        out_cf = model.simulate_counterfactual(x, u, u_cf, mask)
        x_cf = out_cf["x_cf_pred"]

        # Trajectory divergence (over valid steps)
        diff = (x_fact - x_cf) ** 2             # (B, T, Dx)
        if mask is not None:
            diff = diff * mask.unsqueeze(-1).float()
            seq_counts = mask.float().sum(dim=1).clamp(min=1)       # (B,)
            divergence = diff.sum(dim=(1, 2)) / (seq_counts * x_fact.size(-1) + 1e-8)
        else:
            divergence = diff.mean(dim=(1, 2))  # (B,)
        traj_divergences.extend(divergence.cpu().tolist())

        # Outcome sensitivity
        p_fact = torch.sigmoid(out_fact["outcome_logit"])
        p_cf = out_cf["outcome_cf"]
        sensitivity = (p_fact - p_cf).abs()
        outcome_sensitivities.extend(sensitivity.cpu().tolist())

    result = {
        "trajectory_divergence_mean": float(np.mean(traj_divergences)),
        "trajectory_divergence_std": float(np.std(traj_divergences)),
        "outcome_sensitivity_mean": float(np.mean(outcome_sensitivities)),
        "outcome_sensitivity_std": float(np.std(outcome_sensitivities)),
        "cf_strategy": cf_strategy,
    }
    logger.info("Counterfactual metrics: %s", result)
    return result


# ---------------------------------------------------------------------------
# Per-hospital breakdown
# ---------------------------------------------------------------------------

def evaluate_per_hospital(
    probs: np.ndarray,
    labels: np.ndarray,
    hospital_ids: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """Compute metrics per hospital."""
    results = {}
    for hid in np.unique(hospital_ids):
        mask = hospital_ids == hid
        if mask.sum() < 5:  # too few samples
            continue
        results[int(hid)] = compute_all_metrics(labels[mask], probs[mask])
    return results


# ---------------------------------------------------------------------------
# Ablation studies
# ---------------------------------------------------------------------------

def run_ablations(
    cfg: Dict,
    train_seqs,
    val_seqs,
    test_seqs,
    device: torch.device,
    base_checkpoint: Optional[str] = None,
    batch_size: int = 64,
    num_epochs: int = 10,
    output_dir: str = "outputs/ablations",
) -> Dict[str, Dict]:
    """
    Systematic ablation of model components.

    Ablations:
        - no_hosp_adv   : remove hospital adversary (lambda_hosp_adv=0)
        - no_trt_adv    : remove treatment adversary (lambda_trt_adv=0)
        - no_env_branch : remove e_t branch (e_dim=0 → constant zero)
        - no_disentangle: collapse s and e into single hidden (standard GRU)
        - no_contrastive: remove contrastive loss (lambda_contrastive=0)
    """
    from data import get_dataloaders
    from train import train_model
    from model import build_model

    results = {}
    train_loader, val_loader, test_loader = get_dataloaders(
        train_seqs, val_seqs, test_seqs, batch_size=batch_size,
        hospital_id_map=cfg.get("hospital_id_map")
    )

    ablation_configs = {
        "no_hosp_adv": {"lambda_hosp_adv": 0.0},
        "no_trt_adv": {"lambda_trt_adv": 0.0},
        "no_env_branch": {"e_dim": 0},
        "no_contrastive": {"lambda_contrastive": 0.0},
        "no_adversaries": {"lambda_hosp_adv": 0.0, "lambda_trt_adv": 0.0},
    }

    for name, overrides in ablation_configs.items():
        logger.info("Running ablation: %s  overrides=%s", name, overrides)
        abl_cfg = {**cfg, **overrides}
        # e_dim=0 means no environment branch — set to 1 (minimal)
        if abl_cfg.get("e_dim", 32) == 0:
            abl_cfg["e_dim"] = 1  # minimal
        model = build_model(abl_cfg, device)
        metrics = train_model(
            model, train_loader, val_loader, abl_cfg, device,
            max_epochs=num_epochs, checkpoint_path=f"{output_dir}/{name}_best.pt"
        )
        test_result = evaluate_in_distribution(model, test_loader, device)
        results[name] = test_result["metrics"]
        logger.info("Ablation %s: %s", name, results[name])

    return results


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

def generate_report(
    main_model: nn.Module,
    baselines: Dict[str, nn.Module],
    test_loader: DataLoader,
    ooh_test_loader: Optional[DataLoader],
    device: torch.device,
    output_path: str = "outputs/results.json",
) -> Dict[str, Any]:
    """
    Compile a complete evaluation report for the main model and all baselines.
    """
    report: Dict[str, Any] = {}

    # Main model — in-distribution
    logger.info("Evaluating main model (in-distribution)…")
    indist = evaluate_in_distribution(main_model, test_loader, device, is_disentangled=True)
    report["main_model_indist"] = indist["metrics"]

    # Main model — out-of-hospital
    if ooh_test_loader is not None:
        logger.info("Evaluating main model (out-of-hospital)…")
        ooh = evaluate_out_of_hospital(
            main_model, test_loader, ooh_test_loader, device, is_disentangled=True
        )
        report["main_model_ooh"] = ooh

    # Counterfactual
    logger.info("Running counterfactual evaluation…")
    cf = evaluate_counterfactual(main_model, test_loader, device)
    report["counterfactual"] = cf

    # Baselines
    report["baselines"] = {}
    for name, bl_model in baselines.items():
        logger.info("Evaluating baseline: %s", name)
        bl_result = evaluate_in_distribution(bl_model, test_loader, device,
                                             is_disentangled=False)
        report["baselines"][name] = bl_result["metrics"]
        if ooh_test_loader is not None:
            ooh_bl = evaluate_out_of_hospital(
                bl_model, test_loader, ooh_test_loader, device, is_disentangled=False
            )
            report["baselines"][f"{name}_ooh"] = ooh_bl

    save_json(report, output_path)
    logger.info("Report saved to %s", output_path)
    return report
