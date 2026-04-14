"""
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
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    generate_report,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CFG: Dict[str, Any] = {
    # Data
    "data_path": "data/eicu_final_sequences_for_modeling.csv",
    "max_seq_len": 48,
    "min_seq_len": 3,
    "batch_size": 64,
    "split_mode": "out_of_hospital",  # "random" | "out_of_hospital"
    "seed": 42,
    # Model
    "s_dim": 64,
    "e_dim": 32,
    "hidden_dim": 128,
    "enc_layers": 2,
    "enc_dropout": 0.1,
    "num_trt_classes": 2,
    "proj_dim": 64,
    "grl_alpha": 1.0,
    # Training
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "max_epochs": 100,
    "patience": 15,
    "grad_clip": 1.0,
    # Loss weights
    "lambda_outcome": 1.0,
    "lambda_hosp_adv": 0.1,
    "lambda_trt_adv": 0.1,
    "lambda_contrastive": 0.05,
    "lambda_irm": 0.0,
    # Discriminator
    "disc_lr_multiplier": 2.0,
    # Contrastive augmentation
    "contrastive_noise_scale": 0.05,
    # GRL alpha schedule
    "grl_anneal": True,
    "grl_alpha_max": 1.0,
    # Baselines
    "run_baselines": True,
    "baseline_epochs": 30,
    "baseline_patience": 10,
    "baselines_to_run": [
        "erm", "dann", "domain_confusion", "irm",
        "crn", "gnet", "rmsn", "dcrn",
        "causal_transformer", "g_transformer", "mamba_cdsp",
    ],
    # Ablations
    "run_ablations": True,
    "ablation_epochs": 20,
    # Output
    "output_dir": "outputs",
    "checkpoint_dir": "outputs/checkpoints",
    "log_file": "outputs/train.log",
    "results_path": "outputs/results.json",
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
    n_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        u = batch["u"].to(device)
        mask = batch["mask"].to(device)
        y = batch["y"].to(device)
        hospital_id = batch["hospital_id"].to(device)

        # -------- Discriminator step --------
        disc_stats = update_discriminators(
            model, batch, disc_optim, adv_loss_fn, device,
            cfg.get("num_trt_classes", 2),
        )

        # -------- Encoder + model step --------
        optimizer.zero_grad()
        out = model(x, u, mask)

        # Build adversarial labels for the encoder update
        B, T, _ = x.shape
        if mask is not None:
            valid = mask.view(-1)
            h_flat = hospital_id.unsqueeze(1).expand(B, T).reshape(B * T)[valid]
            u_flat = u.view(B * T, -1)[valid]
        else:
            h_flat = hospital_id.unsqueeze(1).expand(B, T).reshape(B * T)
            u_flat = u.view(B * T, -1)
        trt_label = (u_flat.sum(dim=-1) > 0).long()

        # Contrastive: use two augmented views of s (noise-based augmentation)
        s = out["s"]                           # (B, T, s_dim)
        s_avg = s.mean(dim=1)                  # (B, s_dim) — summary over time
        # Simple augmentation: add Gaussian noise
        noise_scale = cfg.get("contrastive_noise_scale", 0.05)
        z1 = model.contrastive_head(
            s_avg + torch.randn_like(s_avg) * noise_scale
        )
        z2 = model.contrastive_head(
            s_avg + torch.randn_like(s_avg) * noise_scale
        )

        losses = loss_fn(
            x_pred=out["x_pred"],
            x_true=x,
            outcome_logits=out["outcome_logit"],
            outcome_labels=y,
            hosp_logits=out["hosp_logits"],
            hosp_labels=h_flat,
            trt_logits=out["trt_logits"],
            trt_labels=trt_label,
            mask=mask,
            z1=z1,
            z2=z2,
        )

        total_loss = losses["total"]

        # Optional IRM penalty (applied across hospital environments)
        if cfg.get("lambda_irm", 0.0) > 0:
            irm_losses = []
            for hid in hospital_id.unique():
                h_mask = hospital_id == hid
                if h_mask.sum() < 2:
                    continue
                scale = torch.ones(1, device=device, requires_grad=True)
                pen = irm_penalty(
                    out["outcome_logit"][h_mask] * scale,
                    y[h_mask],
                    scale,
                )
                irm_losses.append(pen)
            if irm_losses:
                total_loss = total_loss + cfg["lambda_irm"] * torch.stack(irm_losses).mean()

        total_loss.backward()
        if cfg.get("grad_clip", 0) > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        # Accumulate stats
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                total_stats[k] = total_stats.get(k, 0.0) + v.item()
        for k, v in disc_stats.items():
            total_stats[k] = total_stats.get(k, 0.0) + v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_stats.items()}


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
            logger.info("  ✓ New best  AUROC=%.4f  →  saved to %s", val_auroc, checkpoint_path)

        if early_stopping.should_stop:
            logger.info("Early stopping triggered at epoch %d.", epoch)
            break

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
) -> Dict[str, float]:
    """Generic training loop for baselines (outcome + reconstruction losses)."""
    optimizer = AdamW(model.parameters(), lr=cfg.get("lr", 3e-4),
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
    """Train and evaluate ablated variants of the main model."""
    from data import get_dataloaders
    from eval import evaluate_in_distribution

    results = {}

    ablation_overrides = {
        "no_hosp_adv": {"lambda_hosp_adv": 0.0},
        "no_trt_adv": {"lambda_trt_adv": 0.0},
        "no_contrastive": {"lambda_contrastive": 0.0},
        "no_adversaries": {"lambda_hosp_adv": 0.0, "lambda_trt_adv": 0.0},
        "minimal_e": {"e_dim": 4},
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

    logger.info(
        "Data ready: %d features, %d treatments, %d hospitals",
        cfg["num_features"], cfg["num_treatments"], cfg["num_hospitals"],
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

    # Counterfactual
    cf_result = evaluate_counterfactual(main_model, test_loader, device)
    logger.info("Counterfactual results: %s", cf_result)

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    trained_baselines: Dict[str, nn.Module] = {}

    if cfg.get("run_baselines", True):
        logger.info("Training baselines…")
        for bl_name in cfg.get("baselines_to_run", list(BASELINE_REGISTRY.keys())):
            logger.info("  → %s", bl_name)
            try:
                bl_model = build_baseline(
                    bl_name,
                    x_dim=cfg["num_features"],
                    u_dim=cfg["num_treatments"],
                    num_hospitals=cfg["num_hospitals"],
                    hidden_dim=cfg.get("hidden_dim", 128),
                    device=device,
                )
                ckpt_bl = f"{cfg['checkpoint_dir']}/best_{bl_name}.pt"
                train_baseline(
                    bl_model, train_loader, val_loader, cfg, device,
                    name=bl_name,
                    max_epochs=cfg.get("baseline_epochs", 30),
                    checkpoint_path=ckpt_bl,
                )
                trained_baselines[bl_name] = bl_model
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

    # Augment with OOH training result and ablations
    if ooh_result:
        report["main_model_ooh_dedicated"] = ooh_result
    report["ablations"] = ablation_results

    save_json(report, cfg["results_path"])
    logger.info("=" * 60)
    logger.info("DONE. Results saved to %s", cfg["results_path"])
    logger.info("=" * 60)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Main model (in-distribution): {indist['metrics']}")
    if ooh_result:
        print(f"Main model (out-of-hospital): {ooh_result}")
    print(f"Counterfactual: {cf_result}")
    print(f"\nBaselines:")
    for bname, bm in report.get("baselines", {}).items():
        if not bname.endswith("_ooh"):
            print(f"  {bname:25s}: AUROC={bm.get('auroc', float('nan')):.4f}")
    if ablation_results:
        print(f"\nAblations:")
        for aname, am in ablation_results.items():
            print(f"  {aname:25s}: AUROC={am.get('auroc', float('nan')):.4f}")
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
        "--mode", choices=["main", "baselines", "all"], default="all",
        help="What to train"
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
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
        cfg["output_dir"] = args.output_dir
        cfg["checkpoint_dir"] = f"{args.output_dir}/checkpoints"
        cfg["log_file"] = f"{args.output_dir}/train.log"
        cfg["results_path"] = f"{args.output_dir}/results.json"
    if args.split_mode is not None:
        cfg["split_mode"] = args.split_mode
    if args.no_ablations:
        cfg["run_ablations"] = False
    if args.no_baselines:
        cfg["run_baselines"] = False
    if args.mode == "main":
        cfg["run_baselines"] = False
        cfg["run_ablations"] = False
    elif args.mode == "baselines":
        cfg["run_ablations"] = False

    main(cfg)
