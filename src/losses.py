"""
Loss functions for the disentangled ICU model.

Includes:
  - Reconstruction loss (MSE)
  - Outcome loss (BCE)
  - Hospital adversary loss
  - Treatment adversary loss
  - Contrastive (NT-Xent) loss
  - IRM penalty
losses.py — All loss functions for the ICU trajectory model.

Losses:
  - ReconstructionLoss   (MSE, feature-wise)
  - OutcomeLoss          (Binary cross-entropy with pos-weight)
  - AdversarialLoss      (Cross-entropy for hospital / treatment adversary)
  - ContrastiveLoss      (NT-Xent / InfoNCE for invariant representations)
  - IRMPenalty           (Invariant Risk Minimization v1)
  - TotalLoss            (Weighted combination)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def reconstruction_loss(x_pred: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    """MSE over T-1 steps (predict t+1 from t)."""
    return F.mse_loss(x_pred, x_true)


def outcome_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy for mortality / risk outcome."""
    return F.binary_cross_entropy_with_logits(y_pred.squeeze(-1), y_true)


def adversarial_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss on domain classifier."""
    return F.cross_entropy(logits, labels)


def contrastive_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    NT-Xent contrastive loss between two views of the invariant state.
    z_i, z_j: [B, D] — two augmented views.
    """
    B = z_i.size(0)
    z_i = F.normalize(z_i, dim=-1)
    z_j = F.normalize(z_j, dim=-1)
    z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
    sim = torch.mm(z, z.t()) / temperature  # [2B, 2B]

    # Mask out self-similarity
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, float("-inf"))

    # Positive pairs: (i, B+i) and (B+i, i)
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)], dim=0).to(z.device)
    loss = F.cross_entropy(sim, labels)
    return loss

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reconstruction Loss
# ---------------------------------------------------------------------------

class ReconstructionLoss(nn.Module):
    """MSE between predicted and true next-step observations.

    Supports masked computation (padding positions are ignored).
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        x_pred: torch.Tensor,    # (B, T, D)
        x_true: torch.Tensor,    # (B, T, D)
        mask: Optional[torch.Tensor] = None,  # (B, T) bool, True = valid
    ) -> torch.Tensor:
        loss = F.mse_loss(x_pred, x_true, reduction="none")  # (B, T, D)
        if mask is not None:
            loss = loss * mask.unsqueeze(-1).float()
            return loss.sum() / (mask.float().sum() * x_pred.size(-1) + 1e-8)
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


# ---------------------------------------------------------------------------
# Outcome Loss
# ---------------------------------------------------------------------------

class OutcomeLoss(nn.Module):
    """Binary cross-entropy for mortality / risk prediction."""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,   # (B,) or (B, 1)
        labels: torch.Tensor,   # (B,) float
    ) -> torch.Tensor:
        logits = logits.view(-1)
        labels = labels.view(-1).float()
        pw = self.pos_weight
        if pw is not None:
            pw = pw.to(logits.device)
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)


# ---------------------------------------------------------------------------
# Adversarial Loss
# ---------------------------------------------------------------------------

class AdversarialLoss(nn.Module):
    """Cross-entropy for domain / treatment adversary heads.

    The adversary is trained to classify correctly; the encoder is trained
    (via gradient reversal) to fool the adversary.  This function computes
    the classification loss used for *both* purposes — gradient direction is
    handled by the GradientReversalLayer in the model.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,   # (N, num_classes) or (N,) for binary
        labels: torch.Tensor,   # (N,) long  (or float for binary)
    ) -> torch.Tensor:
        if logits.dim() == 1 or logits.size(-1) == 1:
            return F.binary_cross_entropy_with_logits(
                logits.view(-1), labels.float()
            )
        return F.cross_entropy(logits, labels.long())


# ---------------------------------------------------------------------------
# Contrastive Loss (NT-Xent)
# ---------------------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    """NT-Xent contrastive loss for aligning invariant representations.

    Expects pairs of embeddings from the same patient under different hospital
    contexts — or from augmented views of the same time-series.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z1: torch.Tensor,  # (B, D) first view
        z2: torch.Tensor,  # (B, D) second view
    ) -> torch.Tensor:
        B = z1.size(0)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        # (2B, D)
        z = torch.cat([z1, z2], dim=0)
        # (2B, 2B) cosine similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature
        # Mask out self-similarities
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, float("-inf"))
        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
        loss = F.cross_entropy(sim, labels)
        return loss


# ---------------------------------------------------------------------------
# IRM Penalty
# ---------------------------------------------------------------------------

def irm_penalty(
    logits: torch.Tensor,
    labels: torch.Tensor,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    """
    IRM-style gradient penalty across environments.
    Penalizes the variance of per-environment gradients of a 1x1 linear head.
    """
    unique_envs = env_ids.unique()
    if len(unique_envs) < 2:
        return logits.sum() * 0.0  # zero with gradient flow

    w = torch.ones(1, device=logits.device, requires_grad=True)
    penalties = []
    for env in unique_envs:
        mask = env_ids == env
        if mask.sum() == 0:
            continue
        env_logits = logits[mask] * w
        env_labels = labels[mask]
        env_loss = F.binary_cross_entropy_with_logits(env_logits.squeeze(-1), env_labels)
        grad = torch.autograd.grad(env_loss, w, create_graph=True)[0]
        penalties.append(grad ** 2)

    if not penalties:
        return logits.sum() * 0.0  # zero with gradient flow
    return torch.stack(penalties).mean()


class TotalLoss(nn.Module):
    """Weighted sum of all training losses."""
    scale: torch.Tensor,
) -> torch.Tensor:
    """IRM v1 gradient penalty (Arjovsky et al., 2019).

    Penalises the gradient of the risk w.r.t. a fixed scalar `scale`
    (initialised to 1).  The gradient norm should be small at an invariant
    predictor.
    """
    loss = F.binary_cross_entropy_with_logits(logits * scale, labels.float())
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return (grad ** 2).sum()


# ---------------------------------------------------------------------------
# Total Loss
# ---------------------------------------------------------------------------

class TotalLoss(nn.Module):
    """Weighted sum of all sub-losses.

    L = L_recon
      + λ_outcome  * L_outcome
      + λ_hosp_adv * L_hospital_adversary
      + λ_trt_adv  * L_treatment_adversary
      + λ_contrast * L_contrastive        (optional)
      + λ_irm      * L_irm               (optional)
    """

    def __init__(
        self,
        lambda_outcome: float = 1.0,
        lambda_hospital_adv: float = 0.1,
        lambda_treatment_adv: float = 0.1,
        lambda_contrastive: float = 0.05,
        lambda_irm: float = 0.01,
    ):
        super().__init__()
        self.lambda_outcome = lambda_outcome
        self.lambda_hospital_adv = lambda_hospital_adv
        self.lambda_treatment_adv = lambda_treatment_adv
        self.lambda_contrastive = lambda_contrastive
        self.lambda_irm = lambda_irm

        lambda_hosp_adv: float = 0.1,
        lambda_trt_adv: float = 0.1,
        lambda_contrastive: float = 0.05,
        lambda_irm: float = 0.0,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.lambda_outcome = lambda_outcome
        self.lambda_hosp_adv = lambda_hosp_adv
        self.lambda_trt_adv = lambda_trt_adv
        self.lambda_contrastive = lambda_contrastive
        self.lambda_irm = lambda_irm

        self.recon_loss = ReconstructionLoss()
        self.outcome_loss = OutcomeLoss(pos_weight=pos_weight)
        self.adv_loss = AdversarialLoss()
        self.contrastive_loss = ContrastiveLoss()

    def forward(
        self,
        x_pred: torch.Tensor,
        x_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        hosp_logits: Optional[torch.Tensor] = None,
        hosp_labels: Optional[torch.Tensor] = None,
        treat_logits: Optional[torch.Tensor] = None,
        treat_labels: Optional[torch.Tensor] = None,
        s_i: Optional[torch.Tensor] = None,
        s_j: Optional[torch.Tensor] = None,
        irm_logits: Optional[torch.Tensor] = None,
        irm_labels: Optional[torch.Tensor] = None,
        irm_env_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        losses = {}
        losses["recon"] = reconstruction_loss(x_pred, x_true)
        losses["outcome"] = outcome_loss(y_pred, y_true)

        total = losses["recon"] + self.lambda_outcome * losses["outcome"]

        if hosp_logits is not None and hosp_labels is not None:
            losses["hosp_adv"] = adversarial_loss(hosp_logits, hosp_labels)
            total = total + self.lambda_hospital_adv * losses["hosp_adv"]

        if treat_logits is not None and treat_labels is not None:
            losses["treat_adv"] = adversarial_loss(treat_logits, treat_labels)
            total = total + self.lambda_treatment_adv * losses["treat_adv"]

        if s_i is not None and s_j is not None:
            losses["contrastive"] = contrastive_loss(s_i, s_j)
            total = total + self.lambda_contrastive * losses["contrastive"]

        if irm_logits is not None and irm_labels is not None and irm_env_ids is not None:
            losses["irm"] = irm_penalty(irm_logits, irm_labels, irm_env_ids)
            total = total + self.lambda_irm * losses["irm"]

        losses["total"] = total
        return total, losses
        outcome_logits: torch.Tensor,
        outcome_labels: torch.Tensor,
        hosp_logits: torch.Tensor,
        hosp_labels: torch.Tensor,
        trt_logits: torch.Tensor,
        trt_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        z1: Optional[torch.Tensor] = None,
        z2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        l_recon = self.recon_loss(x_pred, x_true, mask)
        l_outcome = self.outcome_loss(outcome_logits, outcome_labels)
        l_hosp = self.adv_loss(hosp_logits, hosp_labels)
        l_trt = self.adv_loss(trt_logits, trt_labels)

        total = (
            l_recon
            + self.lambda_outcome * l_outcome
            + self.lambda_hosp_adv * l_hosp
            + self.lambda_trt_adv * l_trt
        )

        losses = {
            "total": total,
            "recon": l_recon,
            "outcome": l_outcome,
            "hosp_adv": l_hosp,
            "trt_adv": l_trt,
        }

        if z1 is not None and z2 is not None and self.lambda_contrastive > 0:
            l_contrast = self.contrastive_loss(z1, z2)
            total = total + self.lambda_contrastive * l_contrast
            losses["contrastive"] = l_contrast
            losses["total"] = total

        return losses
