"""
Evaluation utilities for the ICU trajectory model.

Includes:
  - Standard metrics: AUROC, AUPRC, accuracy
  - Calibration: ECE + reliability bin stats
  - Counterfactual PROXY evaluation (clearly labelled as proxy, not ground truth)
      * consistency test: same u_t → same rollout
      * temporal intervention: intervene at t=k vs t=k+Δ, compare divergence
      * monotonicity sanity (skipped with logged note if directionality unknown)
      * stress tests (zero / max / flip) — labelled as stress tests
      * grounded sanity: stratify by treatment intensity, compare predictions to observed
  - Competitiveness flagging: ΔAUROC vs best baseline
  - Disentanglement check: performance when e_t branch removed
  - s_t predictive sufficiency probe: AUROC of linear classifier on s_t alone
  - Domain generalization metrics: generalization gap + relative drop
eval.py — Comprehensive evaluation for ICU trajectory models.

Functions
---------
evaluate_model            — Generic evaluation of any model on a DataLoader
evaluate_in_distribution  — Evaluate on random test split
evaluate_out_of_hospital  — Evaluate on held-out hospital test set
evaluate_counterfactual   — Counterfactual proxy evaluation (consistency + temporal tests)
monitor_disentanglement   — Monitor latent collapse and e_t contribution
evaluate_per_hospital     — Per-hospital metric breakdown
run_ablations             — Systematically ablate model components
generate_report           — Full report: ΔAUROC, calibration, OOH, counterfactual
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from utils import get_logger

logger = get_logger("synthica.eval")

# Named constants
ST_COLLAPSE_THRESHOLD = 0.55   # s_t-only AUROC below this → warns of useless representation
EPSILON_DIV_SAFE = 1e-8        # added to denominators to prevent division by zero


# ---------------------------------------------------------------------------
# Standard metrics
# ---------------------------------------------------------------------------

def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class in y_true — AUROC undefined, returning 0.5")
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class in y_true — AUPRC undefined, returning baseline prevalence")
        return float(y_true.mean())
    return float(average_precision_score(y_true, y_score))


def compute_accuracy(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (y_score >= threshold).astype(int)
    return float(accuracy_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None,
) -> Tuple[float, List[dict]]:
    """
    Compute Expected Calibration Error.

    Args:
        y_true: ground truth binary labels
        y_prob: predicted probabilities
        n_bins: number of calibration bins
        save_path: if set, write bin stats as JSON to this path

    Returns:
        (ece, bin_stats_list)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_stats = []
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        count = int(mask.sum())
        if count == 0:
            bin_stats.append({"bin_lo": lo, "bin_hi": hi, "count": 0, "avg_conf": None, "avg_acc": None})
            continue
        avg_conf = float(y_prob[mask].mean())
        avg_acc = float(y_true[mask].mean())
        bin_stats.append({"bin_lo": float(lo), "bin_hi": float(hi),
                          "count": count, "avg_conf": avg_conf, "avg_acc": avg_acc})
        ece += (count / n) * abs(avg_conf - avg_acc)

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, "w") as fh:
            json.dump({"ece": ece, "bins": bin_stats}, fh, indent=2)
        logger.info("Reliability data saved → %s", save_path)

    return float(ece), bin_stats


# ---------------------------------------------------------------------------
# Full metrics bundle
# ---------------------------------------------------------------------------

def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    tag: str = "eval",
    ece_save_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Run model on loader and return AUROC, AUPRC, accuracy, ECE.
    """
    model.eval()
    all_y_true: List[float] = []
    all_y_score: List[float] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            u = batch["u"].to(device)
            y_true = batch["outcome"].cpu().numpy()
            env_ids = batch["env_id"].to(device)

            out = model(x, u, env_ids=env_ids)
            y_logit = out["y_pred"].squeeze(-1).cpu().numpy()
            y_prob = 1.0 / (1.0 + np.exp(-y_logit))  # sigmoid

            all_y_true.extend(y_true.tolist())
            all_y_score.extend(y_prob.tolist())

    y_true_arr = np.array(all_y_true)
    y_score_arr = np.array(all_y_score)

    auroc = compute_auroc(y_true_arr, y_score_arr)
    auprc = compute_auprc(y_true_arr, y_score_arr)
    acc = compute_accuracy(y_true_arr, y_score_arr)
    ece, _ = compute_ece(y_true_arr, y_score_arr, save_path=ece_save_path)

    metrics = {"auroc": auroc, "auprc": auprc, "accuracy": acc, "ece": ece}
    logger.info("[%s] AUROC=%.4f  AUPRC=%.4f  Acc=%.4f  ECE=%.4f", tag, auroc, auprc, acc, ece)
    return metrics


# ---------------------------------------------------------------------------
# Competitiveness flagging
# ---------------------------------------------------------------------------

def flag_competitiveness(model_auroc: float, baseline_results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Compare main model AUROC against best baseline AUROC.
    Logs ΔAUROC and emits prominent warning if negative.
    """
    if not baseline_results:
        logger.warning("No baseline results available for competitiveness check.")
        return {"delta_auroc": None, "best_baseline": None, "best_baseline_auroc": None}

    best_baseline = max(baseline_results.keys(), key=lambda k: baseline_results[k].get("auroc", 0.0))
    best_baseline_auroc = baseline_results[best_baseline].get("auroc", 0.0)
    delta = model_auroc - best_baseline_auroc

    info = {
        "delta_auroc": round(delta, 4),
        "best_baseline": best_baseline,
        "best_baseline_auroc": round(best_baseline_auroc, 4),
        "model_auroc": round(model_auroc, 4),
    }

    if delta < 0:
        logger.warning(
            "!!! COMPETITIVENESS WARNING !!!\n"
            "  Main model AUROC (%.4f) < Best baseline '%s' AUROC (%.4f)\n"
            "  ΔAUROC = %.4f  — model is UNDERPERFORMING best baseline.",
            model_auroc, best_baseline, best_baseline_auroc, delta,
        )
    else:
        logger.info(
            "Competitiveness check PASSED: ΔAUROC = +%.4f over baseline '%s'",
            delta, best_baseline,
        )

    return info


# ---------------------------------------------------------------------------
# Counterfactual PROXY Evaluation
# ---------------------------------------------------------------------------
# NOTE: This is a PROXY evaluation, not a causal ground truth.
# It measures model self-consistency and sensitivity to interventions.
# It does NOT establish causal validity.

def _rollout_batch(model, x0: torch.Tensor, u: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Utility: encode initial frame and roll out trajectory."""
    model.eval()
    with torch.no_grad():
        if hasattr(model, "rollout_counterfactual"):
            return model.rollout_counterfactual(x0, u, n_steps=u.shape[1]).cpu()
        # Fallback for baselines without explicit rollout
        out = model(x0.expand(-1, u.shape[1], -1), u)
        return out["x_pred"].cpu()


def test_consistency(
    model: torch.nn.Module,
    sequences: List[dict],
    device: torch.device,
    n_samples: int = 8,
    atol: float = 1e-4,
) -> Dict[str, Any]:
    """
    CONSISTENCY TEST (proxy): identical u_t → identical rollout (within numerical tolerance).
    """
    logger.info("[Counterfactual Proxy] Running consistency test …")
    model.eval()
    passed = 0
    total = 0
    max_diff = 0.0

    indices = np.random.choice(len(sequences), size=min(n_samples, len(sequences)), replace=False)
    for idx in indices:
        s = sequences[idx]
        x = s["x"].unsqueeze(0).to(device)  # [1, T, F]
        u = s["u"].unsqueeze(0).to(device)  # [1, T, U]

        traj_a = _rollout_batch(model, x[:, :1, :], u, device)
        traj_b = _rollout_batch(model, x[:, :1, :], u, device)

        diff = (traj_a - traj_b).abs().max().item()
        max_diff = max(max_diff, diff)
        if diff <= atol:
            passed += 1
        total += 1

    result = {"test": "consistency", "passed": passed, "total": total, "max_diff": max_diff,
               "pass_rate": passed / total if total > 0 else 0.0}
    logger.info("[Counterfactual Proxy] Consistency: %d/%d passed (max_diff=%.2e)", passed, total, max_diff)
    return result


def test_temporal_intervention(
    model: torch.nn.Module,
    sequences: List[dict],
    device: torch.device,
    k: int = 4,
    delta: int = 4,
    n_samples: int = 8,
) -> Dict[str, Any]:
    """
    TEMPORAL INTERVENTION TEST (proxy):
    Intervene at t=k vs t=k+delta and measure divergence between resulting trajectories.
    Earlier intervention should lead to larger cumulative divergence.
    """
    logger.info("[Counterfactual Proxy] Running temporal intervention test (k=%d, delta=%d) …", k, delta)
    model.eval()
    divergences_k: List[float] = []
    divergences_k_delta: List[float] = []

    indices = np.random.choice(len(sequences), size=min(n_samples, len(sequences)), replace=False)
    for idx in indices:
        s = sequences[idx]
        T = s["x"].shape[0]
        if T < k + delta + 2:
            continue
        x = s["x"].unsqueeze(0).to(device)
        u = s["u"].unsqueeze(0).to(device)

        # Factual rollout
        traj_factual = _rollout_batch(model, x[:, :1, :], u, device)  # [1, T-1, F]

        # Counterfactual: zero out u from t=k onward
        u_cf_k = u.clone()
        u_cf_k[:, k:, :] = 0.0
        traj_cf_k = _rollout_batch(model, x[:, :1, :], u_cf_k, device)

        # Counterfactual: zero out u from t=k+delta onward
        u_cf_kd = u.clone()
        u_cf_kd[:, k + delta:, :] = 0.0
        traj_cf_kd = _rollout_batch(model, x[:, :1, :], u_cf_kd, device)

        n_steps = min(traj_factual.shape[1], traj_cf_k.shape[1], traj_cf_kd.shape[1])
        div_k = (traj_factual[:, :n_steps] - traj_cf_k[:, :n_steps]).norm(dim=-1).mean().item()
        div_kd = (traj_factual[:, :n_steps] - traj_cf_kd[:, :n_steps]).norm(dim=-1).mean().item()
        divergences_k.append(div_k)
        divergences_k_delta.append(div_kd)

    if not divergences_k:
        logger.warning("[Counterfactual Proxy] Temporal intervention: no valid samples.")
        return {"test": "temporal_intervention", "error": "no valid samples"}

    mean_div_k = float(np.mean(divergences_k))
    mean_div_kd = float(np.mean(divergences_k_delta))
    # Earlier intervention (k) should cause larger or equal divergence
    ordering_correct = mean_div_k >= mean_div_kd

    result = {
        "test": "temporal_intervention",
        "k": k,
        "k_plus_delta": k + delta,
        "mean_div_at_k": round(mean_div_k, 6),
        "mean_div_at_k_delta": round(mean_div_kd, 6),
        "earlier_intervention_larger_divergence": ordering_correct,
    }
    logger.info(
        "[Counterfactual Proxy] Temporal intervention: div@k=%.4f div@k+Δ=%.4f — ordering_correct=%s",
        mean_div_k, mean_div_kd, ordering_correct,
    )
    return result


def test_monotonicity(
    model: torch.nn.Module,
    sequences: List[dict],
    device: torch.device,
    n_samples: int = 8,
) -> Dict[str, Any]:
    """
    MONOTONICITY SANITY TEST (proxy):
    For treatments with known directionality (e.g. vasopressors → MAP increase),
    verify model predicts change in expected direction.

    Since directionality is not known a priori for this dataset, this test
    is SKIPPED with a logged note. Replace with domain-specific logic if known.
    """
    logger.info(
        "[Counterfactual Proxy] Monotonicity sanity test SKIPPED — "
        "treatment directionality unknown for this dataset. "
        "Add domain-specific logic to enable this test."
    )
    return {
        "test": "monotonicity",
        "status": "skipped",
        "reason": "treatment directionality unknown for this dataset",
    }


def run_stress_tests(
    model: torch.nn.Module,
    sequences: List[dict],
    device: torch.device,
    n_samples: int = 8,
) -> Dict[str, Any]:
    """
    STRESS TESTS (not causal ground truth):
    - zero-treatment: all u_t = 0
    - max-treatment: all u_t = 1
    - flip-treatment: u_t = 1 - u_t
    Measures sensitivity of trajectory predictions.
    """
    logger.info("[Stress Tests] Running stress tests (zero/max/flip) …")
    model.eval()
    results: Dict[str, List[float]] = {"zero": [], "max": [], "flip": []}

    indices = np.random.choice(len(sequences), size=min(n_samples, len(sequences)), replace=False)
    for idx in indices:
        s = sequences[idx]
        x = s["x"].unsqueeze(0).to(device)
        u = s["u"].unsqueeze(0).to(device)

        traj_orig = _rollout_batch(model, x[:, :1, :], u, device)

        for key, u_mod in [
            ("zero", torch.zeros_like(u)),
            ("max", torch.ones_like(u)),
            ("flip", 1.0 - u.clamp(0, 1)),
        ]:
            traj_mod = _rollout_batch(model, x[:, :1, :], u_mod, device)
            n = min(traj_orig.shape[1], traj_mod.shape[1])
            div = (traj_orig[:, :n] - traj_mod[:, :n]).norm(dim=-1).mean().item()
            results[key].append(div)

    summary = {k: round(float(np.mean(v)), 6) if v else None for k, v in results.items()}
    logger.info("[Stress Tests] zero=%.4f  max=%.4f  flip=%.4f",
                summary["zero"] or 0, summary["max"] or 0, summary["flip"] or 0)
    return {"test": "stress_tests", "mean_divergence": summary}


def run_counterfactual_proxy_evaluation(
    model: torch.nn.Module,
    sequences: List[dict],
    device: torch.device,
    n_samples: int = 16,
) -> Dict[str, Any]:
    """
    Run all counterfactual PROXY evaluations.
    NOTE: These are proxy tests measuring model consistency and sensitivity.
          They are NOT causal ground truth evaluations.
    """
    logger.info("=== COUNTERFACTUAL PROXY EVALUATION (not causal ground truth) ===")
    results = {
        "disclaimer": "proxy evaluation — not causal ground truth",
        "consistency": test_consistency(model, sequences, device, n_samples=n_samples),
        "temporal_intervention": test_temporal_intervention(model, sequences, device, n_samples=n_samples),
        "monotonicity": test_monotonicity(model, sequences, device, n_samples=n_samples),
        "stress_tests": run_stress_tests(model, sequences, device, n_samples=n_samples),
    }
    return results


# ---------------------------------------------------------------------------
# Disentanglement check
# ---------------------------------------------------------------------------

def disentanglement_check(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    full_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """
    Check that removing the e_t branch degrades reconstruction/outcome performance.
    Uses a forward pass with e_t zeroed out to simulate branch removal.
    """
    logger.info("[Disentanglement] Running e_t branch removal check …")
    from losses import reconstruction_loss, outcome_loss

    model.eval()
    orig_recon_losses: List[float] = []
    ablated_recon_losses: List[float] = []
    orig_outcome_losses: List[float] = []
    ablated_outcome_losses: List[float] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            u = batch["u"].to(device)
            y_true = batch["outcome"].to(device)
            env_ids = batch["env_id"].to(device)

            # Full forward
            out = model(x, u, env_ids=env_ids)
            orig_recon_losses.append(reconstruction_loss(out["x_pred"], out["x_true"]).item())
            orig_outcome_losses.append(outcome_loss(out["y_pred"], y_true).item())

            # Ablated: zero e_t
            if hasattr(model, "decoder") and hasattr(model, "inv_dynamics"):
                s_seq, e_seq, _ = model.encoder(x, u)
                T = x.shape[1]
                s_next_list, e_next_list = [], []
                for t in range(T - 1):
                    s_next_t = model.inv_dynamics(s_seq[:, t, :])
                    e_next_t = torch.zeros_like(e_seq[:, t, :])  # zero out e_t
                    s_next_list.append(s_next_t)
                    e_next_list.append(e_next_t)
                s_next = torch.stack(s_next_list, dim=1)
                e_next = torch.stack(e_next_list, dim=1)
                x_pred_abl = model.decoder(s_next, e_next)
                y_pred_abl = model.outcome_head(s_seq[:, -1, :])
                ablated_recon_losses.append(reconstruction_loss(x_pred_abl, out["x_true"]).item())
                ablated_outcome_losses.append(outcome_loss(y_pred_abl, y_true).item())
            else:
                # Baseline: e_t branch doesn't exist, skip
                ablated_recon_losses.append(orig_recon_losses[-1])
                ablated_outcome_losses.append(orig_outcome_losses[-1])

    orig_recon = float(np.mean(orig_recon_losses))
    abl_recon = float(np.mean(ablated_recon_losses))
    orig_outcome = float(np.mean(orig_outcome_losses))
    abl_outcome = float(np.mean(ablated_outcome_losses))

    recon_degraded = abl_recon > orig_recon * 1.01  # >1% degradation
    outcome_degraded = abl_outcome > orig_outcome * 1.01

    result = {
        "orig_recon_loss": round(orig_recon, 6),
        "ablated_recon_loss": round(abl_recon, 6),
        "recon_degraded": recon_degraded,
        "orig_outcome_loss": round(orig_outcome, 6),
        "ablated_outcome_loss": round(abl_outcome, 6),
        "outcome_degraded": outcome_degraded,
        "disentanglement_confirmed": recon_degraded or outcome_degraded,
    }

    if not result["disentanglement_confirmed"]:
        logger.warning(
            "[Disentanglement] e_t removal did NOT significantly degrade performance — "
            "disentanglement may not be real."
        )
    else:
        logger.info(
            "[Disentanglement] e_t removal degraded recon_loss by %.4f%% — disentanglement confirmed.",
            100.0 * (abl_recon - orig_recon) / (orig_recon + EPSILON_DIV_SAFE),
        )

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import compute_all_metrics, safe_auroc, calibration_error, save_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named thresholds (avoid magic numbers in monitoring logic)
# ---------------------------------------------------------------------------

# Minimum per-dimension variance of s_t before triggering a collapse warning.
# Below this value, at least one dimension of the invariant latent has effectively
# collapsed to a constant — a sign of over-regularisation or GRL dominance.
_COLLAPSE_VARIANCE_THRESHOLD: float = 1e-4

# Minimum fractional reconstruction degradation required to confirm that e_t
# carries genuine environment signal.  If zeroing e_t raises reconstruction MSE
# by less than 1%, the environment branch is likely not contributing meaningfully.
_MIN_RECON_DEGRADATION: float = 0.01

# Tolerance for temporal monotonicity check: earlier CF interventions should
# produce at least as much divergence as later ones (up to this tolerance).
_TEMPORAL_MONOTONICITY_TOLERANCE: float = 1e-6


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
# Counterfactual evaluation (proxy — not causal ground truth)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_counterfactual(
    model: nn.Module,  # Must be DisentangledICUModel
    loader: DataLoader,
    device: torch.device,
    cf_strategy: str = "zero_treatment",
) -> Dict[str, Any]:
    """
    **Counterfactual proxy evaluation** — NOT causal ground truth.

    Simulates counterfactual trajectories using three complementary checks:
      1. Trajectory divergence  — mean per-step L2 distance factual vs. CF
      2. Outcome sensitivity    — |P(death|factual) − P(death|CF)|
      3. Consistency check      — re-running with identical u_t yields zero divergence
      4. Temporal intervention  — divergence grows when intervention is applied
                                  earlier in the sequence

    All results are labelled as proxy measures.  Causal guarantees require
    access to randomised interventions or an explicit structural causal model.

    Parameters
    ----------
    cf_strategy : How to construct counterfactual treatment:
        "zero_treatment"  — set all treatments to zero
        "max_treatment"   — set all treatments to 1
        "flip_treatment"  — flip binary treatments
    """
    logger.info(
        "=== Counterfactual PROXY evaluation (strategy=%s) — not causal ground truth ===",
        cf_strategy,
    )
    model.eval()

    traj_divergences: List[float] = []
    outcome_sensitivities: List[float] = []
    consistency_errors: List[float] = []        # should be ~0
    temporal_early_div: List[float] = []        # intervene at T//3
    temporal_late_div: List[float] = []         # intervene at 2*T//3

    for batch in loader:
        x = batch["x"].to(device)
        u = batch["u"].to(device)
        mask = batch["mask"].to(device)

        # ------------------------------------------------------------------
        # Build counterfactual treatment
        # ------------------------------------------------------------------
        if cf_strategy == "zero_treatment":
            u_cf = torch.zeros_like(u)
        elif cf_strategy == "max_treatment":
            u_cf = torch.ones_like(u)
        elif cf_strategy == "flip_treatment":
            u_cf = 1.0 - u.clamp(0, 1)
        else:
            raise ValueError(f"Unknown cf_strategy: {cf_strategy}")

        # ------------------------------------------------------------------
        # 1 & 2: Standard trajectory divergence + outcome sensitivity
        # ------------------------------------------------------------------
        out_fact = model(x, u, mask)
        x_fact = out_fact["x_pred"]
        out_cf = model.simulate_counterfactual(x, u, u_cf, mask)
        x_cf = out_cf["x_cf_pred"]

        diff = (x_fact - x_cf) ** 2
        if mask is not None:
            diff = diff * mask.unsqueeze(-1).float()
            seq_counts = mask.float().sum(dim=1).clamp(min=1)
            divergence = diff.sum(dim=(1, 2)) / (seq_counts * x_fact.size(-1) + 1e-8)
        else:
            divergence = diff.mean(dim=(1, 2))
        traj_divergences.extend(divergence.cpu().tolist())

        p_fact = torch.sigmoid(out_fact["outcome_logit"])
        p_cf = out_cf["outcome_cf"]
        sensitivity = (p_fact - p_cf).abs()
        outcome_sensitivities.extend(sensitivity.cpu().tolist())

        # ------------------------------------------------------------------
        # 3: Consistency check — same u → same trajectory (deterministic check)
        #    Under the same treatment, the counterfactual should equal factual.
        # ------------------------------------------------------------------
        out_same = model.simulate_counterfactual(x, u, u, mask)   # u_cf == u
        same_diff = (out_fact["x_pred"] - out_same["x_cf_pred"]).abs()
        if mask is not None:
            same_diff = same_diff * mask.unsqueeze(-1).float()
        consistency_err = same_diff.mean().item()
        consistency_errors.append(consistency_err)

        # ------------------------------------------------------------------
        # 4: Temporal intervention test
        #    Intervene at T//3 (early) vs 2*T//3 (late); earlier intervention
        #    should produce larger cumulative divergence from factual.
        # ------------------------------------------------------------------
        T = x.size(1)
        t_early = max(1, T // 3)
        t_late = max(1, 2 * T // 3)

        # Early intervention: apply CF treatment only from t_early onwards
        u_early = u.clone()
        u_early[:, t_early:, :] = u_cf[:, t_early:, :]
        out_early = model.simulate_counterfactual(x, u, u_early, mask)
        diff_early = (x_fact - out_early["x_cf_pred"]) ** 2
        if mask is not None:
            diff_early = diff_early * mask.unsqueeze(-1).float()
            div_early = diff_early.sum(dim=(1, 2)) / (seq_counts * x_fact.size(-1) + 1e-8)
        else:
            div_early = diff_early.mean(dim=(1, 2))
        temporal_early_div.extend(div_early.cpu().tolist())

        # Late intervention: apply CF treatment only from t_late onwards
        u_late = u.clone()
        u_late[:, t_late:, :] = u_cf[:, t_late:, :]
        out_late = model.simulate_counterfactual(x, u, u_late, mask)
        diff_late = (x_fact - out_late["x_cf_pred"]) ** 2
        if mask is not None:
            diff_late = diff_late * mask.unsqueeze(-1).float()
            div_late = diff_late.sum(dim=(1, 2)) / (seq_counts * x_fact.size(-1) + 1e-8)
        else:
            div_late = diff_late.mean(dim=(1, 2))
        temporal_late_div.extend(div_late.cpu().tolist())

    mean_early = float(np.mean(temporal_early_div))
    mean_late = float(np.mean(temporal_late_div))
    # Sanity: earlier intervention should produce at least as much divergence as later
    temporal_monotone_ok = bool(mean_early >= mean_late - _TEMPORAL_MONOTONICITY_TOLERANCE)

    result = {
        "proxy_evaluation_disclaimer": (
            "These are counterfactual PROXY metrics — not causal ground truth. "
            "They test model responsiveness to treatment changes. "
            "Causal claims require randomised interventions or an explicit SCM."
        ),
        "cf_strategy": cf_strategy,
        # Standard metrics
        "trajectory_divergence_mean": float(np.mean(traj_divergences)),
        "trajectory_divergence_std": float(np.std(traj_divergences)),
        "outcome_sensitivity_mean": float(np.mean(outcome_sensitivities)),
        "outcome_sensitivity_std": float(np.std(outcome_sensitivities)),
        # Consistency check
        "consistency_error_mean": float(np.mean(consistency_errors)),
        "consistency_check_pass": bool(np.mean(consistency_errors) < 1e-4),
        # Temporal intervention test
        "temporal_early_divergence_mean": mean_early,
        "temporal_late_divergence_mean": mean_late,
        "temporal_monotonicity_ok": temporal_monotone_ok,
    }
    if not temporal_monotone_ok:
        logger.warning(
            "Temporal monotonicity FAILED: early_div=%.4f < late_div=%.4f  "
            "(model may not be sensitive to timing of intervention)",
            mean_early, mean_late,
        )
    logger.info("Counterfactual proxy metrics: %s", result)
    return result


# ---------------------------------------------------------------------------
# s_t predictive sufficiency probe
# ---------------------------------------------------------------------------

def evaluate_st_predictive_probe(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    full_model_auroc: float,
) -> Dict[str, Any]:
    """
    Train a linear classifier on s_t alone (no e_t) to predict y.
    Compares AUROC(s_t only) vs AUROC(full model) to verify s_t carries
    outcome signal and is not invariant-but-useless.

    If s_t_only_auroc << full_model_auroc, s_t is predictive but less so —
    expected and acceptable. If s_t_only_auroc ≈ 0.5, s_t has collapsed to
    a useless representation.
    """
    import torch.optim as optim

    logger.info("[s_t Probe] Training linear probe on s_t → outcome …")

    if not hasattr(model, "encoder") or not hasattr(model, "s_dim"):
        logger.warning("[s_t Probe] Model has no encoder/s_dim — skipping.")
        return {"status": "skipped", "reason": "model lacks s_t branch"}

    s_dim = model.s_dim

    # Collect s_T and labels from loader
    model.eval()
    all_s: List[np.ndarray] = []
    all_y: List[float] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            u = batch["u"].to(device)
            y_true = batch["outcome"].cpu().numpy()
            out = model(x, u)
            # Use mean s across time as probe input (same as adversary)
            s_mean = out["s_seq"].mean(dim=1).cpu().numpy()  # [B, s_dim]
            all_s.append(s_mean)
            all_y.extend(y_true.tolist())

    S = np.concatenate(all_s, axis=0)  # [N, s_dim]
    Y = np.array(all_y)                # [N]

    if len(np.unique(Y)) < 2:
        logger.warning("[s_t Probe] Only one class in labels — probe AUROC undefined.")
        return {"status": "skipped", "reason": "single class in labels"}

    # Split for probe training (80/20)
    n = len(S)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    tr_idx, va_idx = idx[:split], idx[split:]

    S_tr = torch.tensor(S[tr_idx], dtype=torch.float32).to(device)
    Y_tr = torch.tensor(Y[tr_idx], dtype=torch.float32).to(device)
    S_va = torch.tensor(S[va_idx], dtype=torch.float32).to(device)
    Y_va = Y[va_idx]

    # Linear probe
    probe = torch.nn.Linear(s_dim, 1).to(device)
    probe_opt = optim.Adam(probe.parameters(), lr=1e-3)

    for _ in range(50):
        probe.train()
        probe_opt.zero_grad()
        logits = probe(S_tr).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, Y_tr)
        loss.backward()
        probe_opt.step()

    probe.eval()
    with torch.no_grad():
        va_logits = probe(S_va).squeeze(-1).cpu().numpy()
    va_probs = 1.0 / (1.0 + np.exp(-va_logits))
    st_auroc = compute_auroc(Y_va, va_probs)

    delta = st_auroc - full_model_auroc
    result = {
        "st_only_auroc": round(st_auroc, 4),
        "full_model_auroc": round(full_model_auroc, 4),
        "delta_vs_full": round(delta, 4),
    }

    if st_auroc < ST_COLLAPSE_THRESHOLD:
        logger.warning(
            "!!! s_t PREDICTIVE SUFFICIENCY WARNING !!!\n"
            "  s_t-only AUROC = %.4f — s_t may have collapsed to invariant but useless representation.\n"
            "  Check collapse monitoring and reduce adversarial weight.",
            st_auroc,
        )
    else:
        logger.info(
            "[s_t Probe] s_t-only AUROC=%.4f  Full model AUROC=%.4f  Δ=%.4f",
            st_auroc, full_model_auroc, delta,
        )

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
# Latent collapse monitoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def monitor_disentanglement(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Monitor the invariant latent s_t for representation collapse.

    Checks
    ------
    1. **s_t variance** — if near zero, s_t has collapsed to a constant.
       Reported as ``s_var_mean`` (average per-dim variance across the batch).

    2. **Reconstruction degradation when e_t is zeroed** — the decoder receives
       (s_t, 0) instead of (s_t, e_t).  A large increase in MSE confirms that
       e_t carries genuine environment signal and the disentanglement is real.

    Returns a dict logged after each training epoch.
    """
    model.eval()

    s_vecs: List[torch.Tensor] = []
    recon_full: List[float] = []
    recon_no_e: List[float] = []

    for batch in loader:
        x = batch["x"].to(device)
        u = batch["u"].to(device)
        mask = batch["mask"].to(device)

        out = model(x, u, mask)
        s = out["s"]          # (B, T, s_dim)
        e = out["e"]          # (B, T, e_dim)
        x_pred = out["x_pred"]

        # Collect s representations for variance computation
        if mask is not None:
            valid = mask.view(-1)
            s_flat = s.view(-1, s.size(-1))[valid]
        else:
            s_flat = s.view(-1, s.size(-1))
        s_vecs.append(s_flat.cpu())

        # Reconstruction with full e_t
        if mask.any():
            recon_full.append(
                torch.nn.functional.mse_loss(x_pred[mask], x[mask]).item()
            )

        # Reconstruction with e_t zeroed — should degrade if e_t matters
        e_zero = torch.zeros_like(e)
        x_no_e = model.decoder(out.get("s_next", s), e_zero)
        if mask.any():
            recon_no_e.append(
                torch.nn.functional.mse_loss(x_no_e[mask], x[mask]).item()
            )

    s_all = torch.cat(s_vecs, dim=0)  # (N, s_dim)
    s_var_per_dim = s_all.var(dim=0)  # (s_dim,)
    s_var_mean = float(s_var_per_dim.mean().item())
    s_var_min = float(s_var_per_dim.min().item())

    mean_recon_full = float(np.mean(recon_full)) if recon_full else float("nan")
    mean_recon_no_e = float(np.mean(recon_no_e)) if recon_no_e else float("nan")
    recon_degradation = (
        (mean_recon_no_e - mean_recon_full) / (mean_recon_full + 1e-8)
        if recon_full else float("nan")
    )

    if s_var_min < _COLLAPSE_VARIANCE_THRESHOLD:
        logger.warning(
            "[DisentanglementMonitor] s_t collapse detected: "
            "min per-dim variance=%.2e  (some dimensions may have collapsed to constant)",
            s_var_min,
        )

    if recon_degradation < _MIN_RECON_DEGRADATION:
        logger.warning(
            "[DisentanglementMonitor] e_t contributes < 1%% to reconstruction "
            "(recon_no_e=%.4f vs recon_full=%.4f).  "
            "e_t may not carry genuine environment signal.",
            mean_recon_no_e, mean_recon_full,
        )

    result = {
        "s_var_mean": s_var_mean,
        "s_var_min": s_var_min,
        "recon_full_mse": mean_recon_full,
        "recon_no_e_mse": mean_recon_no_e,
        "recon_degradation_frac": recon_degradation,
        "collapse_warning": s_var_min < _COLLAPSE_VARIANCE_THRESHOLD,
    }
    logger.info("[DisentanglementMonitor] %s", result)
    return result


# ---------------------------------------------------------------------------
# Grounded counterfactual sanity check
# ---------------------------------------------------------------------------

def grounded_counterfactual_sanity(
    model: torch.nn.Module,
    sequences: List[dict],
    device: torch.device,
    n_samples: int = 64,
) -> Dict[str, Any]:
    """
    SEMI-GROUNDED COUNTERFACTUAL SANITY CHECK (proxy, not causal ground truth).

    Approach:
      1. Stratify patients by observed treatment intensity (mean u_t magnitude).
      2. For low-treatment patients, predict outcome under INCREASED treatment.
      3. Compare mean predicted outcome (high-treatment counterfactual) against
         mean observed outcome in the actual high-treatment group.

    This anchors predictions to real data distributions — if the model is
    totally arbitrary, the counterfactual predictions will diverge from the
    observed high-treatment distribution.

    NOTE: This is NOT a causal test. It is a sanity check on distributional
    plausibility. Confounding and selection bias are not controlled for.
    """
    logger.info(
        "[Grounded CF Sanity] Stratifying by treatment intensity "
        "(proxy sanity check — not causal ground truth) …"
    )

    if not hasattr(model, "rollout_counterfactual"):
        logger.warning("[Grounded CF Sanity] Model lacks rollout_counterfactual — skipping.")
        return {"status": "skipped", "reason": "model lacks rollout_counterfactual"}

    model.eval()

    # Compute treatment intensity per patient (mean absolute u_t)
    intensities = []
    for s in sequences:
        u = s["u"]  # [T, U]
        intensities.append(float(u.abs().mean().item()))

    intensities = np.array(intensities)
    median_intensity = float(np.median(intensities))

    lo_idx = np.where(intensities <= median_intensity)[0]
    hi_idx = np.where(intensities > median_intensity)[0]

    if len(lo_idx) == 0 or len(hi_idx) == 0:
        return {"status": "skipped", "reason": "cannot stratify — all intensities identical"}

    # Sample from each stratum
    n_each = min(n_samples // 2, len(lo_idx), len(hi_idx))
    lo_sample = np.random.choice(lo_idx, size=n_each, replace=False)
    hi_sample = np.random.choice(hi_idx, size=n_each, replace=False)

    # Observed outcomes in high-treatment group
    hi_outcomes = np.array([float(sequences[i]["outcome"].item()) for i in hi_sample])
    mean_hi_observed = float(hi_outcomes.mean())

    # For low-treatment patients: predict outcome under max treatment (counterfactual)
    cf_outcomes: List[float] = []
    with torch.no_grad():
        for idx in lo_sample:
            s = sequences[idx]
            x0 = s["x"][:1, :].unsqueeze(0).to(device)   # [1, 1, F]
            u_orig = s["u"].unsqueeze(0).to(device)       # [1, T, U]
            T = u_orig.shape[1]
            # Max treatment counterfactual
            u_cf = torch.ones_like(u_orig)
            try:
                x_cf = model.rollout_counterfactual(x0, u_cf, n_steps=T)  # [1, T, F]
                # Encode the counterfactual trajectory to predict outcome
                s_seq, _, _ = model.encoder(x_cf, u_cf[:, :x_cf.shape[1], :])
                y_cf_logit = model.outcome_head(s_seq[:, -1, :]).squeeze(-1).cpu().numpy()
                y_cf_prob = float(1.0 / (1.0 + np.exp(-y_cf_logit[0])))
                cf_outcomes.append(y_cf_prob)
            except Exception as ex:
                logger.debug("[Grounded CF Sanity] rollout failed for sample %d: %s", idx, ex)

    if not cf_outcomes:
        return {"status": "skipped", "reason": "all rollouts failed"}

    mean_cf_predicted = float(np.mean(cf_outcomes))
    # Plausibility: both should be above 0.5 * mean_hi_observed (rough anchor)
    plausible = mean_cf_predicted >= 0.5 * mean_hi_observed

    result = {
        "disclaimer": "proxy sanity check — not causal ground truth; confounding not controlled",
        "median_treatment_intensity": round(median_intensity, 6),
        "n_low_treatment": n_each,
        "n_high_treatment": n_each,
        "mean_hi_treatment_observed_outcome": round(mean_hi_observed, 4),
        "mean_cf_predicted_outcome_under_max_treatment": round(mean_cf_predicted, 4),
        "distributional_plausibility": plausible,
    }
    logger.info(
        "[Grounded CF Sanity] Observed hi-treatment outcome=%.4f  CF predicted=%.4f  plausible=%s",
        mean_hi_observed, mean_cf_predicted, plausible,
    )
    return result


# ---------------------------------------------------------------------------
# Domain generalization metrics
# ---------------------------------------------------------------------------

def compute_domain_generalization_metrics(
    model_results: Dict[str, Dict],
) -> Dict[str, Any]:
    """
    Compute domain generalization gap and relative drop for each model.

    Expects each entry in model_results to have:
      - "auroc": in-distribution AUROC
      - "ooh_auroc": out-of-hospital AUROC (optional)

    Returns per-model generalization gap and relative drop, plus ranking.
    """
    gen_metrics: Dict[str, Dict] = {}

    for name, res in model_results.items():
        if isinstance(res, dict) and "auroc" in res and "ooh_auroc" in res:
            in_auroc = float(res["auroc"])
            out_auroc = float(res["ooh_auroc"])
            gap = round(in_auroc - out_auroc, 4)
            rel_drop = round(gap / (in_auroc + EPSILON_DIV_SAFE), 4)
            gen_metrics[name] = {
                "in_dist_auroc": round(in_auroc, 4),
                "ood_auroc": round(out_auroc, 4),
                "generalization_gap": gap,
                "relative_drop": rel_drop,
            }

    if not gen_metrics:
        logger.warning("[Domain Gen] No models with both in-dist and OOD AUROC available.")
        return {"status": "no_ood_data", "models": {}}

    # Rank by generalization gap (lower is better)
    ranked = sorted(gen_metrics.items(), key=lambda x: x[1]["generalization_gap"])
    logger.info("[Domain Gen] Generalization gap ranking (lower = more robust):")
    for i, (name, m) in enumerate(ranked):
        logger.info(
            "  %d. %-25s gap=%.4f  rel_drop=%.4f  in=%.4f  ood=%.4f",
            i + 1, name, m["generalization_gap"], m["relative_drop"],
            m["in_dist_auroc"], m["ood_auroc"],
        )

    return {"models": gen_metrics, "ranked_by_gap": [r[0] for r in ranked]}
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

    Includes
    --------
    - In-distribution metrics (AUROC, AUPRC, Accuracy, ECE) for main model
    - Out-of-hospital metrics and domain shift drop
    - Counterfactual proxy evaluation (all 3 strategies)
    - Disentanglement monitoring (latent collapse + e_t degradation check)
    - All baselines with the same metric set
    - **ΔAUROC** = main model AUROC − best baseline AUROC (flagged if negative)
    - Per-hospital breakdown for main model
    """
    report: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Main model — in-distribution
    # ------------------------------------------------------------------
    logger.info("Evaluating main model (in-distribution)…")
    indist_result = evaluate_in_distribution(main_model, test_loader, device,
                                             is_disentangled=True)
    report["main_model_indist"] = indist_result["metrics"]

    # Per-hospital breakdown
    ph = evaluate_per_hospital(
        indist_result["probs"], indist_result["labels"], indist_result["hospital_ids"]
    )
    report["main_model_per_hospital"] = ph

    # ------------------------------------------------------------------
    # Main model — out-of-hospital
    # ------------------------------------------------------------------
    if ooh_test_loader is not None:
        logger.info("Evaluating main model (out-of-hospital)…")
        ooh = evaluate_out_of_hospital(
            main_model, test_loader, ooh_test_loader, device, is_disentangled=True
        )
        report["main_model_ooh"] = ooh

    # ------------------------------------------------------------------
    # Counterfactual proxy evaluation (all 3 strategies)
    # ------------------------------------------------------------------
    logger.info("Running counterfactual proxy evaluation…")
    cf_results = {}
    for strategy in ("zero_treatment", "max_treatment", "flip_treatment"):
        try:
            cf_results[strategy] = evaluate_counterfactual(
                main_model, test_loader, device, cf_strategy=strategy
            )
        except Exception as exc:
            logger.warning("Counterfactual strategy %s failed: %s", strategy, exc)
            cf_results[strategy] = {"error": str(exc)}
    report["counterfactual"] = cf_results

    # ------------------------------------------------------------------
    # Disentanglement monitoring
    # ------------------------------------------------------------------
    logger.info("Running disentanglement monitor…")
    try:
        disent = monitor_disentanglement(main_model, test_loader, device)
        report["disentanglement_monitor"] = disent
    except Exception as exc:
        logger.warning("Disentanglement monitor failed: %s", exc)

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    report["baselines"] = {}
    for name, bl_model in baselines.items():
        logger.info("Evaluating baseline: %s", name)
        bl_result = evaluate_in_distribution(bl_model, test_loader, device,
                                             is_disentangled=False)
        # ECE is always included in compute_all_metrics
        report["baselines"][name] = bl_result["metrics"]
        if ooh_test_loader is not None:
            try:
                ooh_bl = evaluate_out_of_hospital(
                    bl_model, test_loader, ooh_test_loader, device,
                    is_disentangled=False,
                )
                report["baselines"][f"{name}_ooh"] = ooh_bl
            except Exception as exc:
                logger.warning("OOH evaluation failed for baseline %s: %s", name, exc)

    # ------------------------------------------------------------------
    # ΔAUROC: main model vs best baseline
    # ------------------------------------------------------------------
    main_auroc = report["main_model_indist"].get("auroc", float("nan"))
    baseline_aurocs = {
        n: m.get("auroc", float("nan"))
        for n, m in report["baselines"].items()
        if "_ooh" not in n
    }
    if baseline_aurocs:
        best_bl_name = max(baseline_aurocs, key=lambda k: baseline_aurocs[k])
        best_bl_auroc = baseline_aurocs[best_bl_name]
        delta_auroc = main_auroc - best_bl_auroc
        report["auroc_comparison"] = {
            "main_model_auroc": main_auroc,
            "best_baseline_name": best_bl_name,
            "best_baseline_auroc": best_bl_auroc,
            "delta_auroc": delta_auroc,
            "main_model_competitive": bool(delta_auroc >= 0),
        }
        if delta_auroc < 0:
            logger.warning(
                "[WARN] Main model AUROC (%.4f) is BELOW best baseline %s (%.4f) "
                "by delta_AUROC=%.4f.  Investigate disentanglement quality or "
                "increase training epochs / model capacity.",
                main_auroc, best_bl_name, best_bl_auroc, delta_auroc,
            )
        else:
            logger.info(
                "[PASS] Main model outperforms best baseline %s: "
                "delta_AUROC=+%.4f (%.4f vs %.4f)",
                best_bl_name, delta_auroc, main_auroc, best_bl_auroc,
            )

    # ------------------------------------------------------------------
    # Calibration summary (ECE) — required, not optional
    # ------------------------------------------------------------------
    report["calibration_summary"] = {
        "main_model_ece": report["main_model_indist"].get("ece"),
        "baselines_ece": {
            n: m.get("ece")
            for n, m in report["baselines"].items()
            if "_ooh" not in n
        },
    }

    save_json(report, output_path)
    logger.info("Report saved to %s", output_path)
    return report
