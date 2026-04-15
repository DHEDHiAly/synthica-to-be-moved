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
baselines.py — SOTA baseline models for ICU trajectory modeling.

Treatment-conditioned forecasting baselines
-------------------------------------------
1.  CRN           — Counterfactual Recurrent Network (Bica et al., 2020)
2.  GNet          — G-Net (Li et al., 2021)
3.  RMSN          — Recurrent Marginal Structural Network (Lim et al., 2018)
4.  DCRN          — Domain-Conditioned Recurrent Network
5.  CausalTransformer — Causal Transformer (Melnychuk et al., 2022)
6.  GTransformer  — G-Transformer (Li et al., 2021 Transformer variant)
7.  MambaCDSP     — Simplified SSM with treatment conditioning

Hospital invariance baselines
------------------------------
8.  ERMBaseline         — Standard supervised GRU
9.  DANNBaseline        — Domain Adversarial Neural Network (Ganin et al.)
10. DomainConfusion     — Domain confusion via adversary without GRL
11. IRMBaseline         — Invariant Risk Minimization (Arjovsky et al., 2019)

Each model exposes:
    forward(x, u, mask) → dict with at least 'outcome_logit' and 'x_pred'
    get_outcome_logit(x, u, mask) → (B,) for evaluation convenience
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
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import GradientReversalLayer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2,
         dropout: float = 0.1) -> nn.Sequential:
    layers: list = []
    dims = [in_dim] + [hidden_dim] * max(n_layers - 1, 1) + [out_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def _get_final_hidden(h: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Extract hidden state at the last valid timestep per sequence."""
    if mask is None:
        return h[:, -1]
    B = h.size(0)
    seq_lens = mask.long().sum(dim=1).clamp(min=1) - 1  # (B,)
    return h[torch.arange(B, device=h.device), seq_lens]


# ---------------------------------------------------------------------------
# 1. CRN — Counterfactual Recurrent Network
# ---------------------------------------------------------------------------

class CRN(nn.Module):
    """
    Faithful reduced implementation of CRN (Bica et al., NeurIPS 2020).

    Encoder: LSTM over (x_t, u_t).
    Balancing: propensity head predicts u_t from encoded state h_t.
    Decoder: treatment-conditioned MLP predicts x_{t+1}.
    Outcome: MLP on final hidden state.
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(x_dim + u_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        # Propensity head for treatment balancing
        self.propensity_head = _mlp(hidden_dim, hidden_dim // 2, u_dim, n_layers=2)
        # Treatment-conditioned decoder
        self.decoder = _mlp(hidden_dim + u_dim, hidden_dim, x_dim, n_layers=2)
        # Outcome head
        self.outcome_head = _mlp(hidden_dim, hidden_dim // 2, 1, n_layers=2)

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        inp = torch.cat([x, u], dim=-1)
        h, _ = self.lstm(inp)                       # (B, T, H)
        # Decode next-step (use current h conditioned on next treatment u)
        x_pred = self.decoder(torch.cat([h, u], dim=-1))
        # Propensity prediction (for inverse-propensity weighting)
        u_pred = self.propensity_head(h)
        # Outcome
        h_final = _get_final_hidden(h, mask)
        outcome_logit = self.outcome_head(h_final).squeeze(-1)
        return {
            "x_pred": x_pred,
            "outcome_logit": outcome_logit,
            "propensity_pred": u_pred,
            "hidden": h,
        }

    def get_outcome_logit(self, x, u, mask=None):
        return self.forward(x, u, mask)["outcome_logit"]


# ---------------------------------------------------------------------------
# 2. G-Net
# ---------------------------------------------------------------------------

class GNet(nn.Module):
    """
    G-Net (Li et al., 2021) — regime-specific GRU with separate heads per
    treatment regime.  Simplified to two regimes (treated / untreated).
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        hidden_dim: int = 128,
        num_regimes: int = 2,
    ) -> None:
        super().__init__()
        self.num_regimes = num_regimes
        # Shared encoder
        self.encoder = nn.GRU(x_dim, hidden_dim, batch_first=True)
        # Per-regime decoders
        self.regime_decoders = nn.ModuleList([
            _mlp(hidden_dim + u_dim, hidden_dim, x_dim, n_layers=2)
            for _ in range(num_regimes)
        ])
        self.outcome_head = _mlp(hidden_dim, hidden_dim // 2, 1, n_layers=2)
        self.regime_classifier = _mlp(u_dim, hidden_dim // 2, num_regimes, n_layers=1)

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h, _ = self.encoder(x)
        # Determine regime from treatment
        regime_weights = torch.softmax(self.regime_classifier(u), dim=-1)  # (B, T, R)
        # Weighted sum of regime predictions
        x_pred_all = torch.stack([dec(torch.cat([h, u], dim=-1))
                                   for dec in self.regime_decoders], dim=-1)  # (B, T, Dx, R)
        x_pred = (x_pred_all * regime_weights.unsqueeze(-2)).sum(-1)
        h_final = _get_final_hidden(h, mask)
        outcome_logit = self.outcome_head(h_final).squeeze(-1)
        return {"x_pred": x_pred, "outcome_logit": outcome_logit}

    def get_outcome_logit(self, x, u, mask=None):
        return self.forward(x, u, mask)["outcome_logit"]


# ---------------------------------------------------------------------------
# 3. RMSN — Recurrent Marginal Structural Network
# ---------------------------------------------------------------------------

class RMSN(nn.Module):
    """
    Simplified RMSN (Lim et al., 2018).

    Encoder-decoder architecture with separate encoder and decoder RNNs.
    Propensity network estimates treatment probabilities for IPW.
    """

    def __init__(self, x_dim: int, u_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        # Encoder RNN
        self.enc_rnn = nn.GRU(x_dim + u_dim, hidden_dim, batch_first=True)
        # Propensity model
        self.propensity = nn.GRU(x_dim + u_dim, hidden_dim // 2, batch_first=True)
        self.prop_head = _mlp(hidden_dim // 2, hidden_dim // 4, u_dim, n_layers=1)
        # Decoder RNN
        self.dec_rnn = nn.GRU(hidden_dim + u_dim, hidden_dim, batch_first=True)
        self.dec_out = nn.Linear(hidden_dim, x_dim)
        # Outcome
        self.outcome_head = _mlp(hidden_dim, hidden_dim // 2, 1, n_layers=2)

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        enc_inp = torch.cat([x, u], dim=-1)
        h_enc, _ = self.enc_rnn(enc_inp)
        # Propensity
        h_prop, _ = self.propensity(enc_inp)
        u_prop = self.prop_head(h_prop)
        # Decoder
        dec_inp = torch.cat([h_enc, u], dim=-1)
        h_dec, _ = self.dec_rnn(dec_inp)
        x_pred = self.dec_out(h_dec)
        # Outcome
        h_final = _get_final_hidden(h_enc, mask)
        outcome_logit = self.outcome_head(h_final).squeeze(-1)
        return {
            "x_pred": x_pred,
            "outcome_logit": outcome_logit,
            "propensity_pred": u_prop,
        }

    def get_outcome_logit(self, x, u, mask=None):
        return self.forward(x, u, mask)["outcome_logit"]


# ---------------------------------------------------------------------------
# 4. DCRN — Domain-Conditioned Recurrent Network
# ---------------------------------------------------------------------------

class DCRN(nn.Module):
    """
    Domain-Conditioned Recurrent Network.

    Conditions both the encoder and decoder on a domain (hospital) embedding.
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        num_hospitals: int,
        hidden_dim: int = 128,
        domain_emb_dim: int = 16,
    ) -> None:
        super().__init__()
        self.domain_emb = nn.Embedding(num_hospitals, domain_emb_dim)
        self.encoder = nn.GRU(x_dim + u_dim + domain_emb_dim, hidden_dim,
                               num_layers=2, batch_first=True, dropout=0.1)
        self.decoder = _mlp(hidden_dim + domain_emb_dim, hidden_dim, x_dim, n_layers=2)
        self.outcome_head = _mlp(hidden_dim + domain_emb_dim, hidden_dim // 2, 1, n_layers=2)

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        hospital_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape
        if hospital_id is None:
            hospital_id = torch.zeros(B, dtype=torch.long, device=x.device)
        d = self.domain_emb(hospital_id)           # (B, D)
        d_seq = d.unsqueeze(1).expand(B, T, -1)   # (B, T, D)
        enc_inp = torch.cat([x, u, d_seq], dim=-1)
        h, _ = self.encoder(enc_inp)
        x_pred = self.decoder(torch.cat([h, d_seq], dim=-1))
        h_final = _get_final_hidden(h, mask)
        outcome_logit = self.outcome_head(torch.cat([h_final, d], dim=-1)).squeeze(-1)
        return {"x_pred": x_pred, "outcome_logit": outcome_logit}

    def get_outcome_logit(self, x, u, mask=None, hospital_id=None):
        return self.forward(x, u, mask, hospital_id)["outcome_logit"]


# ---------------------------------------------------------------------------
# 5. Causal Transformer
# ---------------------------------------------------------------------------

class _CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        return self.dropout(out)


class CausalTransformer(nn.Module):
    """
    Causal Transformer for treatment-conditioned forecasting
    (Melnychuk et al., NeurIPS 2022 — simplified).

    Combines causal self-attention with treatment cross-attention.
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        hidden_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(x_dim + u_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": _CausalSelfAttention(hidden_dim, nhead, dropout),
                "cross_attn": nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout,
                                                    batch_first=True),
                "ff": nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                ),
                "norm1": nn.LayerNorm(hidden_dim),
                "norm2": nn.LayerNorm(hidden_dim),
            })
            for _ in range(num_layers)
        ])
        self.treatment_proj = nn.Linear(u_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, x_dim)
        self.outcome_head = _mlp(hidden_dim, hidden_dim // 2, 1, n_layers=2)

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.input_proj(torch.cat([x, u], dim=-1))  # (B, T, H)
        t_emb = self.treatment_proj(u)                   # (B, T, H)
        for layer in self.layers:
            # Causal self-attention
            h = layer["norm1"](h + layer["self_attn"](h))
            # Cross-attention to treatment
            ca_out, _ = layer["cross_attn"](h, t_emb, t_emb)
            h = layer["norm2"](h + ca_out)
            h = h + layer["ff"](h)
        x_pred = self.output_proj(h)
        h_final = _get_final_hidden(h, mask)
        outcome_logit = self.outcome_head(h_final).squeeze(-1)
        return {"x_pred": x_pred, "outcome_logit": outcome_logit, "hidden": h}

    def get_outcome_logit(self, x, u, mask=None):
        return self.forward(x, u, mask)["outcome_logit"]


# ---------------------------------------------------------------------------
# 6. G-Transformer
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
    G-Transformer: Transformer variant of G-computation for counterfactual
    inference (Li et al., 2021 — simplified Transformer adaptation).
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        hidden_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feat_proj = nn.Linear(x_dim, hidden_dim)
        self.trt_proj = nn.Linear(u_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_head = nn.Linear(hidden_dim, x_dim)
        self.outcome_head = _mlp(hidden_dim, hidden_dim // 2, 1, n_layers=2)

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape
        h = self.feat_proj(x) + self.trt_proj(u)
        # Build causal mask
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        # key_padding_mask: True = ignore
        kp_mask = ~mask if mask is not None else None
        h = self.transformer(h, mask=causal, src_key_padding_mask=kp_mask)
        x_pred = self.out_head(h)
        h_final = _get_final_hidden(h, mask)
        outcome_logit = self.outcome_head(h_final).squeeze(-1)
        return {"x_pred": x_pred, "outcome_logit": outcome_logit}

    def get_outcome_logit(self, x, u, mask=None):
        return self.forward(x, u, mask)["outcome_logit"]


# ---------------------------------------------------------------------------
# 7. Mamba-CDSP (Simplified SSM with treatment conditioning)
# ---------------------------------------------------------------------------

class _SSMLayer(nn.Module):
    """
    Diagonal linear SSM:  h_t = A * h_{t-1} + B * (x_t ‖ u_t),  y_t = C * h_t
    A is a learnable diagonal matrix parameterised in log-space for stability.
    """

    def __init__(self, in_dim: int, state_dim: int, out_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        # Log-diagonal of A for stable parameterisation
        self.log_A = nn.Parameter(torch.randn(state_dim) * 0.5)
        self.B = nn.Linear(in_dim, state_dim, bias=False)
        self.C = nn.Linear(state_dim, out_dim, bias=False)
        self.D = nn.Linear(in_dim, out_dim, bias=True)  # skip connection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, in_dim) → y: (B, T, out_dim)"""
        B, T, _ = x.shape
        A = torch.exp(self.log_A)                      # (state_dim,)
        h = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            h = A * h + self.B(x[:, t])               # diagonal state update
            outputs.append(self.C(h))
        y = torch.stack(outputs, dim=1)               # (B, T, out_dim)
        return y + self.D(x)


class MambaCDSP(nn.Module):
    """
    Mamba-CDSP: Simplified structured SSM with treatment conditioning.
    Inspired by Mamba (Gu & Dao, 2023) and CDSP (causal disentangled SSP).
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        hidden_dim: int = 128,
        state_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_dim = x_dim + u_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        # Stack of SSM layers with residual connections
        self.ssm_layers = nn.ModuleList([
            _SSMLayer(hidden_dim, state_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, x_dim)
        self.outcome_head = _mlp(hidden_dim, hidden_dim // 2, 1, n_layers=2)

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.input_proj(torch.cat([x, u], dim=-1))
        for ssm, norm in zip(self.ssm_layers, self.norms):
            h = norm(h + self.dropout(ssm(h)))
        x_pred = self.output_proj(h)
        h_final = _get_final_hidden(h, mask)
        outcome_logit = self.outcome_head(h_final).squeeze(-1)
        return {"x_pred": x_pred, "outcome_logit": outcome_logit}

    def get_outcome_logit(self, x, u, mask=None):
        return self.forward(x, u, mask)["outcome_logit"]


# ---------------------------------------------------------------------------
# 8. ERM Baseline
# ---------------------------------------------------------------------------

class ERMBaseline(nn.Module):
    """Standard supervised GRU (Empirical Risk Minimization baseline)."""

    def __init__(self, x_dim: int, u_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.gru = nn.GRU(x_dim + u_dim, hidden_dim, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.decoder = _mlp(hidden_dim, hidden_dim, x_dim, n_layers=2)
        self.outcome_head = _mlp(hidden_dim, hidden_dim // 2, 1, n_layers=2)

    def forward(self, x, u, mask=None):
        h, _ = self.gru(torch.cat([x, u], dim=-1))
        x_pred = self.decoder(h)
        h_final = _get_final_hidden(h, mask)
        outcome_logit = self.outcome_head(h_final).squeeze(-1)
        return {"x_pred": x_pred, "outcome_logit": outcome_logit}

    def get_outcome_logit(self, x, u, mask=None):
        return self.forward(x, u, mask)["outcome_logit"]


# ---------------------------------------------------------------------------
# 9. DANN Baseline
# ---------------------------------------------------------------------------

class DANNBaseline(nn.Module):
    """
    Domain-Adversarial Neural Network (Ganin et al., JMLR 2016).

    Standard GRU encoder + domain adversary via gradient reversal.
    No disentanglement — entire hidden state is reversed.
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        num_hospitals: int,
        hidden_dim: int = 128,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(x_dim + u_dim, hidden_dim, 2, batch_first=True, dropout=0.1)
        self.decoder = _mlp(hidden_dim, hidden_dim, x_dim, n_layers=2)
        self.outcome_head = _mlp(hidden_dim, hidden_dim // 2, 1, n_layers=2)
        self.grl = GradientReversalLayer(alpha)
        self.domain_head = _mlp(hidden_dim, hidden_dim // 2, num_hospitals, n_layers=2)

    def forward(self, x, u, mask=None, hospital_id=None):
        h, _ = self.gru(torch.cat([x, u], dim=-1))
        x_pred = self.decoder(h)
        h_final = _get_final_hidden(h, mask)
        outcome_logit = self.outcome_head(h_final).squeeze(-1)
        domain_logit = self.domain_head(self.grl(h_final))
        return {
            "x_pred": x_pred,
            "outcome_logit": outcome_logit,
            "domain_logit": domain_logit,
        }

    def get_outcome_logit(self, x, u, mask=None, hospital_id=None):
        return self.forward(x, u, mask, hospital_id)["outcome_logit"]


# ---------------------------------------------------------------------------
# 10. Domain Confusion Baseline
# ---------------------------------------------------------------------------

class DomainConfusionBaseline(nn.Module):
    """
    Domain confusion via adversarial domain head *without* gradient reversal.
    The classifier is trained to maximise confusion (uniform domain posterior),
    following Tzeng et al., 2015.
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        num_hospitals: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(x_dim + u_dim, hidden_dim, 2, batch_first=True, dropout=0.1)
        self.decoder = _mlp(hidden_dim, hidden_dim, x_dim, n_layers=2)
        self.outcome_head = _mlp(hidden_dim, hidden_dim // 2, 1, n_layers=2)
        self.domain_head = _mlp(hidden_dim, hidden_dim // 2, num_hospitals, n_layers=2)
        self.num_hospitals = num_hospitals

    def forward(self, x, u, mask=None, hospital_id=None):
        h, _ = self.gru(torch.cat([x, u], dim=-1))
        x_pred = self.decoder(h)
        h_final = _get_final_hidden(h, mask)
        outcome_logit = self.outcome_head(h_final).squeeze(-1)
        domain_logit = self.domain_head(h_final)
        # Confusion loss: KL divergence from uniform distribution
        uniform = torch.full_like(domain_logit, 1.0 / self.num_hospitals)
        confusion_loss = F.kl_div(
            F.log_softmax(domain_logit, dim=-1), uniform, reduction="batchmean"
        )
        return {
            "x_pred": x_pred,
            "outcome_logit": outcome_logit,
            "domain_logit": domain_logit,
            "confusion_loss": confusion_loss,
        }

    def get_outcome_logit(self, x, u, mask=None, hospital_id=None):
        return self.forward(x, u, mask, hospital_id)["outcome_logit"]


# ---------------------------------------------------------------------------
# 11. IRM Baseline
# ---------------------------------------------------------------------------

class IRMBaseline(nn.Module):
    """
    Invariant Risk Minimization (Arjovsky et al., 2019).

    Standard GRU encoder; the IRM penalty is applied externally in the
    training loop (see train.py) using the irm_penalty function from losses.py.
    Each hospital is treated as a separate environment.
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(x_dim + u_dim, hidden_dim, 2, batch_first=True,
                          dropout=dropout)
        self.decoder = _mlp(hidden_dim, hidden_dim, x_dim, n_layers=2)
        self.outcome_head = _mlp(hidden_dim, hidden_dim // 2, 1, n_layers=2)
        # Dummy learnable scalar used in IRM penalty
        self.irm_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, u, mask=None):
        h, _ = self.gru(torch.cat([x, u], dim=-1))
        x_pred = self.decoder(h)
        h_final = _get_final_hidden(h, mask)
        outcome_logit = self.outcome_head(h_final).squeeze(-1)
        # Apply IRM scale for gradient penalty computation
        scaled_logit = outcome_logit * self.irm_scale
        return {
            "x_pred": x_pred,
            "outcome_logit": outcome_logit,
            "scaled_logit": scaled_logit,
        }

    def get_outcome_logit(self, x, u, mask=None):
        return self.forward(x, u, mask)["outcome_logit"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BASELINE_REGISTRY: Dict[str, type] = {
    "crn": CRN,
    "gnet": GNet,
    "rmsn": RMSN,
    "dcrn": DCRN,
    "causal_transformer": CausalTransformer,
    "g_transformer": GTransformer,
    "mamba_cdsp": MambaCDSP,
    "erm": ERMBaseline,
    "dann": DANNBaseline,
    "domain_confusion": DomainConfusionBaseline,
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
def build_baseline(
    name: str,
    x_dim: int,
    u_dim: int,
    num_hospitals: int,
    hidden_dim: int = 128,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Factory function for baselines."""
    if device is None:
        device = torch.device("cpu")
    name = name.lower()
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list(BASELINE_REGISTRY)}")

    cls = BASELINE_REGISTRY[name]
    # Inspect constructor signature to pass only supported kwargs
    import inspect
    sig = inspect.signature(cls.__init__)
    params = sig.parameters

    kwargs: Dict = {"x_dim": x_dim, "u_dim": u_dim, "hidden_dim": hidden_dim}
    if "num_hospitals" in params:
        kwargs["num_hospitals"] = num_hospitals

    model = cls(**kwargs)
    return model.to(device)
