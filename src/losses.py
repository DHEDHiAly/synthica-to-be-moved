"""
Loss functions for the disentangled ICU model.

Includes:
  - Reconstruction loss (MSE)
  - Outcome loss (BCE)
  - Hospital adversary loss
  - Treatment adversary loss
  - Contrastive (NT-Xent) loss
  - IRM penalty
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
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

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
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return torch.stack(penalties).mean()


class TotalLoss(nn.Module):
    """Weighted sum of all training losses."""

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
