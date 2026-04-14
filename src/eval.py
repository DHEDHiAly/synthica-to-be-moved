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
