"""
Baselines for ICU trajectory modeling.

Implemented baselines:
  Treatment-conditioned forecasting:
    - CRN  (Counterfactual Recurrent Network)
    - RMSN (Recurrent Marginal Structural Network)
    - DCRN (Dual Counterfactual Recurrent Network)
    - GNet (G-Net)
    - CausalTransformer
    - GTransformer (G-Transformer)
    - MambaCDSP (Mamba-CDSP, simplified)

  Hospital invariance:
    - ERM (standard supervised)
    - DANN (Domain Adversarial Neural Network)
    - DomainConfusion
    - IRM (Invariant Risk Minimization)

All baselines share a unified API:
  forward(x, u, env_ids=None) → dict("x_pred", "x_true", "y_pred", ...)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from model import GradientReversal
from utils import get_logger

logger = get_logger("synthica.baselines")


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class _BaseGRU(nn.Module):
    """Shared GRU backbone for recurrent baselines."""

    def __init__(self, input_dim: int, treatment_dim: int, hidden_dim: int, dropout: float, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim + treatment_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        xu = torch.cat([x, u], dim=-1)
        h = self.input_proj(xu)
        h, _ = self.gru(h)
        return self.dropout(h)


# ---------------------------------------------------------------------------
# CRN — Counterfactual Recurrent Network
# ---------------------------------------------------------------------------

class CRN(_BaseGRU):
    """
    Counterfactual Recurrent Network (Bica et al., 2020).
    Faithful reduced version: treatment-conditioned encoder + balanced rep via
    gradient reversal on treatment propensity head.
    """

    def __init__(self, input_dim: int, treatment_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__(input_dim, treatment_dim, hidden_dim, dropout)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.outcome_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        self.propensity_head = nn.Sequential(
            GradientReversal(1.0),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max(2, treatment_dim)),
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor, env_ids=None) -> dict:
        h = self.encode(x, u)
        x_pred = self.decoder(h[:, :-1, :])
        y_pred = self.outcome_head(h[:, -1, :])
        propensity_logits = self.propensity_head(h.mean(dim=1))
        return {"x_pred": x_pred, "x_true": x[:, 1:, :], "y_pred": y_pred,
                "propensity_logits": propensity_logits}


# ---------------------------------------------------------------------------
# RMSN — Recurrent Marginal Structural Network
# ---------------------------------------------------------------------------

class RMSN(_BaseGRU):
    """
    RMSN (Lim et al., 2018).
    Two-network structure: propensity network + outcome network.
    Simplified: single GRU with propensity-weighted outcome head.
    """

    def __init__(self, input_dim: int, treatment_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__(input_dim, treatment_dim, hidden_dim, dropout)
        # Propensity network
        self.propensity_gru = nn.GRU(input_dim, hidden_dim // 2, num_layers=1, batch_first=True)
        self.propensity_head = nn.Linear(hidden_dim // 2, max(2, treatment_dim))
        # Outcome network
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.outcome_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x: torch.Tensor, u: torch.Tensor, env_ids=None) -> dict:
        h = self.encode(x, u)
        h_prop, _ = self.propensity_gru(x)
        prop_logits = self.propensity_head(h_prop.mean(dim=1))
        x_pred = self.decoder(h[:, :-1, :])
        y_pred = self.outcome_head(h[:, -1, :])
        return {"x_pred": x_pred, "x_true": x[:, 1:, :], "y_pred": y_pred,
                "propensity_logits": prop_logits}


# ---------------------------------------------------------------------------
# DCRN — Dual Counterfactual Recurrent Network
# ---------------------------------------------------------------------------

class DCRN(_BaseGRU):
    """
    DCRN: dual-stream encoder for factual + counterfactual trajectories.
    """

    def __init__(self, input_dim: int, treatment_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__(input_dim, treatment_dim, hidden_dim, dropout)
        # Second stream for counterfactual
        self.cf_gru = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=1, batch_first=True)
        self.decoder = nn.Linear(hidden_dim + hidden_dim // 2, input_dim)
        self.outcome_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x: torch.Tensor, u: torch.Tensor, env_ids=None) -> dict:
        h = self.encode(x, u)  # factual
        h_cf, _ = self.cf_gru(h)  # counterfactual stream
        h_cat = torch.cat([h[:, :-1, :], h_cf[:, :-1, :]], dim=-1)
        x_pred = self.decoder(h_cat)
        y_pred = self.outcome_head(h[:, -1, :])
        return {"x_pred": x_pred, "x_true": x[:, 1:, :], "y_pred": y_pred}


# ---------------------------------------------------------------------------
# G-Net
# ---------------------------------------------------------------------------

class GNet(_BaseGRU):
    """
    G-Net (Li et al., 2021).
    Iterative g-computation style: sequential prediction of next state.
    """

    def __init__(self, input_dim: int, treatment_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__(input_dim, treatment_dim, hidden_dim, dropout)
        self.step_head = nn.Linear(hidden_dim + treatment_dim, input_dim)
        self.outcome_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x: torch.Tensor, u: torch.Tensor, env_ids=None) -> dict:
        h = self.encode(x, u)
        # G-Net: concatenate h_t and u_t to predict x_{t+1}
        h_u = torch.cat([h[:, :-1, :], u[:, :-1, :]], dim=-1)
        x_pred = self.step_head(h_u)
        y_pred = self.outcome_head(h[:, -1, :])
        return {"x_pred": x_pred, "x_true": x[:, 1:, :], "y_pred": y_pred}


# ---------------------------------------------------------------------------
# Causal Transformer
# ---------------------------------------------------------------------------

class CausalTransformer(nn.Module):
    """
    Causal Transformer (Melnychuk et al., 2022).
    Treatment-conditioned transformer with causal (autoregressive) attention.
    """

    def __init__(self, input_dim: int, treatment_dim: int, hidden_dim: int = 128, dropout: float = 0.1,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim + treatment_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.outcome_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask

    def forward(self, x: torch.Tensor, u: torch.Tensor, env_ids=None) -> dict:
        B, T, _ = x.shape
        xu = torch.cat([x, u], dim=-1)
        h = self.input_proj(xu)
        mask = self._causal_mask(T, x.device)
        h = self.transformer(h, mask=mask)
        x_pred = self.decoder(h[:, :-1, :])
        y_pred = self.outcome_head(h[:, -1, :])
        return {"x_pred": x_pred, "x_true": x[:, 1:, :], "y_pred": y_pred}


# ---------------------------------------------------------------------------
# G-Transformer
# ---------------------------------------------------------------------------

class GTransformer(nn.Module):
    """
    G-Transformer (Li et al., 2022).
    Combines G-Net iterative computation with transformer sequence modeling.
    """

    def __init__(self, input_dim: int, treatment_dim: int, hidden_dim: int = 128, dropout: float = 0.1,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim + treatment_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.step_head = nn.Linear(hidden_dim + treatment_dim, input_dim)
        self.outcome_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x: torch.Tensor, u: torch.Tensor, env_ids=None) -> dict:
        B, T, _ = x.shape
        xu = torch.cat([x, u], dim=-1)
        h = self.input_proj(xu)
        h = self.transformer(h)
        h_u = torch.cat([h[:, :-1, :], u[:, :-1, :]], dim=-1)
        x_pred = self.step_head(h_u)
        y_pred = self.outcome_head(h[:, -1, :])
        return {"x_pred": x_pred, "x_true": x[:, 1:, :], "y_pred": y_pred}


# ---------------------------------------------------------------------------
# Mamba-CDSP (simplified)
# ---------------------------------------------------------------------------

class MambaCDSP(_BaseGRU):
    """
    Mamba-CDSP (simplified): state-space model approximated with a GRU
    + diagonal structured state transition (mimics selective state-space).
    """

    def __init__(self, input_dim: int, treatment_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__(input_dim, treatment_dim, hidden_dim, dropout)
        # Diagonal SSM-like layer
        self.ssm_diag = nn.Parameter(torch.zeros(hidden_dim))
        self.ssm_input = nn.Linear(hidden_dim, hidden_dim)
        self.ssm_out = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.outcome_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x: torch.Tensor, u: torch.Tensor, env_ids=None) -> dict:
        h = self.encode(x, u)  # [B, T, H]
        # Selective SSM approximation
        A = torch.sigmoid(self.ssm_diag)  # [H] — decay per dimension
        inp = self.ssm_input(h)
        out_list = []
        state = torch.zeros(h.size(0), h.size(-1), device=h.device)
        for t in range(h.size(1)):
            state = A * state + (1 - A) * inp[:, t, :]
            out_list.append(state)
        ssm_out = self.ssm_out(torch.stack(out_list, dim=1))

        x_pred = self.decoder(ssm_out[:, :-1, :])
        y_pred = self.outcome_head(ssm_out[:, -1, :])
        return {"x_pred": x_pred, "x_true": x[:, 1:, :], "y_pred": y_pred}


# ---------------------------------------------------------------------------
# ERM — Empirical Risk Minimization (standard supervised)
# ---------------------------------------------------------------------------

class ERM(_BaseGRU):
    """Standard GRU trained with ERM (no domain adaptation)."""

    def __init__(self, input_dim: int, treatment_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__(input_dim, treatment_dim, hidden_dim, dropout)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.outcome_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x: torch.Tensor, u: torch.Tensor, env_ids=None) -> dict:
        h = self.encode(x, u)
        x_pred = self.decoder(h[:, :-1, :])
        y_pred = self.outcome_head(h[:, -1, :])
        return {"x_pred": x_pred, "x_true": x[:, 1:, :], "y_pred": y_pred}


# ---------------------------------------------------------------------------
# DANN — Domain Adversarial Neural Network
# ---------------------------------------------------------------------------

class DANN(_BaseGRU):
    """DANN: GRL-based domain adversarial baseline."""

    def __init__(self, input_dim: int, treatment_dim: int, n_domains: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__(input_dim, treatment_dim, hidden_dim, dropout)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.outcome_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        self.domain_head = nn.Sequential(
            GradientReversal(1.0),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_domains),
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor, env_ids=None) -> dict:
        h = self.encode(x, u)
        x_pred = self.decoder(h[:, :-1, :])
        y_pred = self.outcome_head(h[:, -1, :])
        domain_logits = self.domain_head(h.mean(dim=1))
        return {"x_pred": x_pred, "x_true": x[:, 1:, :], "y_pred": y_pred,
                "domain_logits": domain_logits}


# ---------------------------------------------------------------------------
# Domain Confusion
# ---------------------------------------------------------------------------

class DomainConfusion(_BaseGRU):
    """
    Domain confusion: train domain classifier to uniform distribution
    (minimise KL from uniform) rather than via gradient reversal.
    """

    def __init__(self, input_dim: int, treatment_dim: int, n_domains: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__(input_dim, treatment_dim, hidden_dim, dropout)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.outcome_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_domains),
        )
        self.n_domains = n_domains

    def forward(self, x: torch.Tensor, u: torch.Tensor, env_ids=None) -> dict:
        h = self.encode(x, u)
        x_pred = self.decoder(h[:, :-1, :])
        y_pred = self.outcome_head(h[:, -1, :])
        domain_logits = self.domain_head(h.mean(dim=1))
        return {"x_pred": x_pred, "x_true": x[:, 1:, :], "y_pred": y_pred,
                "domain_logits": domain_logits, "n_domains": self.n_domains}


# ---------------------------------------------------------------------------
# IRM Baseline
# ---------------------------------------------------------------------------

class IRMBaseline(_BaseGRU):
    """
    IRM baseline: same GRU as ERM but training uses IRM penalty externally.
    forward() returns logits needed to compute IRM penalty in train.py.
    """

    def __init__(self, input_dim: int, treatment_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__(input_dim, treatment_dim, hidden_dim, dropout)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.outcome_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x: torch.Tensor, u: torch.Tensor, env_ids=None) -> dict:
        h = self.encode(x, u)
        x_pred = self.decoder(h[:, :-1, :])
        y_pred = self.outcome_head(h[:, -1, :])
        return {"x_pred": x_pred, "x_true": x[:, 1:, :], "y_pred": y_pred}


# ---------------------------------------------------------------------------
# Baseline factory
# ---------------------------------------------------------------------------

BASELINE_REGISTRY: dict[str, type] = {
    "crn": CRN,
    "rmsn": RMSN,
    "dcrn": DCRN,
    "gnet": GNet,
    "causal_transformer": CausalTransformer,
    "g_transformer": GTransformer,
    "mamba_cdsp": MambaCDSP,
    "erm": ERM,
    "dann": DANN,
    "domain_confusion": DomainConfusion,
    "irm": IRMBaseline,
}


def build_baseline(name: str, input_dim: int, treatment_dim: int, n_domains: int, **kwargs) -> nn.Module:
    name_l = name.lower()
    if name_l not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINE_REGISTRY.keys())}")
    cls = BASELINE_REGISTRY[name_l]
    if name_l in ("dann", "domain_confusion"):
        return cls(input_dim, treatment_dim, n_domains=n_domains, **kwargs)
    return cls(input_dim, treatment_dim, **kwargs)
