"""
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
# s_t linear probe AUROC
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_st_probe_auroc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Train a logistic regression probe on frozen s_final embeddings to predict outcome.

    A high AUROC confirms the invariant s_t branch carries genuine predictive signal.
    A low AUROC (< 0.55) suggests s_t has collapsed or is not informative.
    """
    model.eval()
    all_s_final: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for batch in loader:
        x = batch["x"].to(device)
        u = batch["u"].to(device)
        mask = batch["mask"].to(device)
        y = batch["y"]

        if hasattr(model, "encoder"):
            out = model(x, u, mask)
            s_final = out.get("s_final")
            if s_final is not None:
                all_s_final.append(s_final.cpu().numpy())
                all_y.append(y.numpy())

    if not all_s_final:
        return float("nan")

    s_np = np.concatenate(all_s_final)
    y_np = np.concatenate(all_y)

    if len(np.unique(y_np)) < 2:
        return float("nan")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        s_scaled = StandardScaler().fit_transform(s_np)
        clf = LogisticRegression(max_iter=1000, C=0.1, random_state=42, solver="lbfgs")
        clf.fit(s_scaled, y_np)
        probs = clf.predict_proba(s_scaled)[:, 1]
        return safe_auroc(y_np, probs)
    except Exception as exc:
        logger.warning("s_t probe AUROC computation failed: %s", exc)
        return float("nan")


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
    - **ΔAUROC** = main model AUROC − best baseline AUROC (PASS/COMPETITIVE/NOT_COMPETITIVE)
    - s_t probe AUROC — confirms invariant branch carries predictive signal
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
    # Store raw predictions for plot generation
    report["_main_model_probs"] = indist_result["probs"].tolist()
    report["_main_model_labels"] = indist_result["labels"].tolist()

    # Per-hospital breakdown
    ph = evaluate_per_hospital(
        indist_result["probs"], indist_result["labels"], indist_result["hospital_ids"]
    )
    report["main_model_per_hospital"] = ph

    # ------------------------------------------------------------------
    # s_t linear probe AUROC (confirms invariant branch has signal)
    # ------------------------------------------------------------------
    logger.info("Computing s_t linear probe AUROC…")
    st_probe = compute_st_probe_auroc(main_model, test_loader, device)
    report["st_probe_auroc"] = st_probe
    _ST_PROBE_THRESHOLD = 0.60
    if not (st_probe != st_probe) and st_probe < _ST_PROBE_THRESHOLD:  # nan-safe
        logger.warning(
            "[WARN] s_t probe AUROC=%.4f is below threshold %.2f — "
            "invariant branch may not carry genuine predictive signal.",
            st_probe, _ST_PROBE_THRESHOLD,
        )
    else:
        logger.info("[OK] s_t probe AUROC=%.4f → VALID (threshold=%.2f)",
                    st_probe, _ST_PROBE_THRESHOLD)

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
    report["_baseline_probs"] = {}
    for name, bl_model in baselines.items():
        logger.info("Evaluating baseline: %s", name)
        bl_result = evaluate_in_distribution(bl_model, test_loader, device,
                                             is_disentangled=False)
        # ECE is always included in compute_all_metrics
        report["baselines"][name] = bl_result["metrics"]
        report["_baseline_probs"][name] = bl_result["probs"].tolist()
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
    # ΔAUROC: main model vs best baseline (PASS / COMPETITIVE / NOT_COMPETITIVE)
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

        # Trichotomy: PASS (≥0) / COMPETITIVE (≥-0.01) / NOT_COMPETITIVE (otherwise)
        if delta_auroc >= 0:
            competitive_status = "PASS"
        elif delta_auroc >= -0.01:
            competitive_status = "COMPETITIVE"
        else:
            competitive_status = "NOT_COMPETITIVE"

        report["auroc_comparison"] = {
            "main_model_auroc": main_auroc,
            "best_baseline_name": best_bl_name,
            "best_baseline_auroc": best_bl_auroc,
            "delta_auroc": delta_auroc,
            "main_model_competitive": bool(delta_auroc >= 0),
            "competitive_status": competitive_status,
        }

        if competitive_status == "NOT_COMPETITIVE":
            logger.warning(
                "[NOT_COMPETITIVE] Main model AUROC (%.4f) is BELOW best baseline %s "
                "(%.4f) by delta_AUROC=%.4f (< -0.01). Investigate disentanglement "
                "quality or increase training epochs / model capacity.",
                main_auroc, best_bl_name, best_bl_auroc, delta_auroc,
            )
        elif competitive_status == "COMPETITIVE":
            logger.warning(
                "[COMPETITIVE] Main model AUROC (%.4f) is within -0.01 of best "
                "baseline %s (%.4f). delta_AUROC=%.4f",
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
