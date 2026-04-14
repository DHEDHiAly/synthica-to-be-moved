"""
model.py — DisentangledICUModel: the core research architecture.

Architecture
------------
Encoder (GRU):
    (x_t ‖ u_t) → h_t  →  linear split  →  s_t (invariant), e_t (env)

Latent Dynamics:
    InvariantDynamics : s_{t+1} = GRU(s_t)          (no treatment input)
    EnvironmentDynamics: e_{t+1} = GRU(e_t ‖ u_t)

Decoder (MLP):
    x̂_{t+1} = Decoder(s_{t+1} ‖ e_{t+1})

Outcome Head (MLP):
    ŷ = OutcomeHead(s_T)

Invariance Mechanisms:
    HospitalAdversary : GRL → MLP → hospital class logits  (on s_t)
    TreatmentAdversary: GRL → MLP → treatment class logits (on s_t)
    ContrastiveHead   : projects s_t to a unit-sphere (for NT-Xent loss)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import GradientReversalLayer


# ---------------------------------------------------------------------------
# Small helper blocks
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2,
         dropout: float = 0.1) -> nn.Sequential:
    """Build a n_layers-deep ReLU MLP with optional dropout."""
    layers: list = []
    dims = [in_dim] + [hidden_dim] * max(n_layers - 1, 1) + [out_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:           # no activation / dropout after last
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class GRUEncoder(nn.Module):
    """GRU encoder mapping (x_t ‖ u_t) → hidden h_t, then split → (s_t, e_t).

    Returns:
        s : (B, T, s_dim)   invariant disease state
        e : (B, T, e_dim)   environment / treatment state
        h : (B, T, hidden)  raw GRU hidden (for debugging)
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        hidden_dim: int,
        s_dim: int,
        e_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=x_dim + u_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj_s = nn.Linear(hidden_dim, s_dim)
        self.proj_e = nn.Linear(hidden_dim, e_dim)

    def forward(
        self, x: torch.Tensor, u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inp = torch.cat([x, u], dim=-1)         # (B, T, x_dim+u_dim)
        h, _ = self.gru(inp)                    # (B, T, hidden)
        s = self.proj_s(h)                      # (B, T, s_dim)
        e = self.proj_e(h)                      # (B, T, e_dim)
        return s, e, h


class InvariantDynamics(nn.Module):
    """Recurrent dynamics over the invariant state s_t only.

    s_{t+1} = f_inv(s_t)  — no treatment influence.
    Implemented as a single-layer GRU operating on s sequences.
    """

    def __init__(self, s_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(s_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, s_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: (B, T, s_dim) → s_next: (B, T, s_dim)"""
        h, _ = self.gru(s)
        return self.proj(h)


class EnvironmentDynamics(nn.Module):
    """Recurrent dynamics over (e_t ‖ u_t).

    e_{t+1} = f_env(e_t, u_t)  — treatment-conditioned.
    """

    def __init__(self, e_dim: int, u_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(e_dim + u_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, e_dim)

    def forward(self, e: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """e: (B, T, e_dim), u: (B, T, u_dim) → e_next: (B, T, e_dim)"""
        inp = torch.cat([e, u], dim=-1)
        h, _ = self.gru(inp)
        return self.proj(h)


class Decoder(nn.Module):
    """Decode (s_{t+1} ‖ e_{t+1}) → x̂_{t+1}."""

    def __init__(self, s_dim: int, e_dim: int, x_dim: int,
                 hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = _mlp(s_dim + e_dim, hidden_dim, x_dim, n_layers=3)

    def forward(self, s: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, e], dim=-1))


class OutcomeHead(nn.Module):
    """Predict binary outcome from the final invariant state s_T."""

    def __init__(self, s_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = _mlp(s_dim, hidden_dim, 1, n_layers=2)

    def forward(self, s_final: torch.Tensor) -> torch.Tensor:
        """s_final: (B, s_dim) → logits: (B,)"""
        return self.net(s_final).squeeze(-1)


class HospitalAdversary(nn.Module):
    """Adversarially classify hospital from s_t (via gradient reversal)."""

    def __init__(
        self, s_dim: int, num_hospitals: int, hidden_dim: int = 64, alpha: float = 1.0
    ) -> None:
        super().__init__()
        self.grl = GradientReversalLayer(alpha)
        self.net = _mlp(s_dim, hidden_dim, num_hospitals, n_layers=2)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """s: (B*T, s_dim) or (B, s_dim) → logits: same shape[:-1] × num_hospitals"""
        return self.net(self.grl(s))

    def set_alpha(self, alpha: float) -> None:
        self.grl.set_alpha(alpha)


class TreatmentAdversary(nn.Module):
    """Adversarially classify treatment from s_t (via gradient reversal).

    For multi-dimensional treatments we predict a scalar treatment index
    (the argmax treatment) or use an aggregate presence indicator.
    """

    def __init__(
        self, s_dim: int, num_classes: int = 2, hidden_dim: int = 64, alpha: float = 1.0
    ) -> None:
        super().__init__()
        self.grl = GradientReversalLayer(alpha)
        self.net = _mlp(s_dim, hidden_dim, num_classes, n_layers=2)
        self.num_classes = num_classes

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(self.grl(s))

    def set_alpha(self, alpha: float) -> None:
        self.grl.set_alpha(alpha)


class ContrastiveHead(nn.Module):
    """Project s_t to a normalised embedding for NT-Xent / InfoNCE."""

    def __init__(self, s_dim: int, proj_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, s_dim),
            nn.ReLU(),
            nn.Linear(s_dim, proj_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(s), dim=-1)


# ---------------------------------------------------------------------------
# Full Disentangled ICU Model
# ---------------------------------------------------------------------------

class DisentangledICUModel(nn.Module):
    """
    Full ICU trajectory model with disentangled invariant / environment dynamics.

    Inputs (per forward pass):
        x  : (B, T, x_dim)   physiological observations
        u  : (B, T, u_dim)   treatments
        mask: (B, T) bool     valid timesteps

    Outputs (dict):
        x_pred        : (B, T, x_dim)   reconstructed next-step features
        outcome_logit : (B,)            mortality logit
        hosp_logits   : (B*T_valid, num_hospitals)
        trt_logits    : (B*T_valid, num_trt_classes)
        s             : (B, T, s_dim)   invariant latent
        e             : (B, T, e_dim)   environment latent
        s_proj        : (B, T, proj_dim) contrastive projection
        s_final       : (B, s_dim)      final timestep invariant state
    """

    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        num_hospitals: int,
        s_dim: int = 64,
        e_dim: int = 32,
        hidden_dim: int = 128,
        enc_layers: int = 2,
        enc_dropout: float = 0.1,
        num_trt_classes: int = 2,
        proj_dim: int = 64,
        grl_alpha: float = 1.0,
        use_e_for_outcome: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        use_e_for_outcome : bool
            When True, the outcome head is applied to e_final instead of s_final.
            This is used for the "no_invariant_s" ablation study, which tests whether
            the invariant s_t branch actually carries predictive signal.  If removing
            s_t causes a large performance drop, it confirms s_t is non-trivial.
        """
        super().__init__()
        self.s_dim = s_dim
        self.e_dim = e_dim
        self.use_e_for_outcome = use_e_for_outcome

        self.encoder = GRUEncoder(x_dim, u_dim, hidden_dim, s_dim, e_dim,
                                  enc_layers, enc_dropout)
        self.inv_dynamics = InvariantDynamics(s_dim, hidden_dim // 2)
        self.env_dynamics = EnvironmentDynamics(e_dim, u_dim, hidden_dim // 2)
        # Note: dynamics modules use hidden_dim // 2 to keep parameter count balanced
        # relative to the larger encoder while avoiding overfitting in the dynamics.
        self.decoder = Decoder(s_dim, e_dim, x_dim, hidden_dim)
        # Outcome head dimension depends on ablation mode.
        outcome_in_dim = e_dim if use_e_for_outcome else s_dim
        self.outcome_head = OutcomeHead(outcome_in_dim, hidden_dim // 2)
        self.hosp_adversary = HospitalAdversary(s_dim, num_hospitals,
                                                hidden_dim // 2, grl_alpha)
        self.trt_adversary = TreatmentAdversary(s_dim, num_trt_classes,
                                                hidden_dim // 2, grl_alpha)
        self.contrastive_head = ContrastiveHead(s_dim, proj_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape

        # 1. Encode
        s, e, _ = self.encoder(x, u)           # (B, T, s_dim/e_dim)

        # 2. Latent dynamics — predict NEXT step
        s_next = self.inv_dynamics(s)           # (B, T, s_dim)
        e_next = self.env_dynamics(e, u)        # (B, T, e_dim)

        # 3. Decode next-step
        x_pred = self.decoder(s_next, e_next)  # (B, T, x_dim)

        # 4. Outcome from final valid state (invariant branch or env branch)
        if mask is not None:
            seq_lens = mask.long().sum(dim=1).clamp(min=1) - 1  # (B,)
            s_final = s[torch.arange(B, device=s.device), seq_lens]
            e_final = e[torch.arange(B, device=e.device), seq_lens]
        else:
            s_final = s[:, -1]                  # (B, s_dim)
            e_final = e[:, -1]                  # (B, e_dim)

        # use_e_for_outcome=True → ablation: outcome predicted from e_t only
        outcome_input = e_final if self.use_e_for_outcome else s_final
        outcome_logit = self.outcome_head(outcome_input)

        # 5. Adversarial heads — flatten time, optionally apply mask
        if mask is not None:
            valid_mask = mask.view(-1)                   # (B*T,)
            s_flat = s.view(B * T, self.s_dim)[valid_mask]
        else:
            s_flat = s.view(B * T, self.s_dim)
        hosp_logits = self.hosp_adversary(s_flat)
        trt_logits = self.trt_adversary(s_flat)

        # 6. Contrastive projection
        s_proj = self.contrastive_head(s)        # (B, T, proj_dim)

        return {
            "x_pred": x_pred,
            "outcome_logit": outcome_logit,
            "hosp_logits": hosp_logits,
            "trt_logits": trt_logits,
            "s": s,
            "e": e,
            "s_next": s_next,
            "e_next": e_next,
            "s_proj": s_proj,
            "s_final": s_final,
            "e_final": e_final,
        }

    # ------------------------------------------------------------------
    def simulate_counterfactual(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        u_cf: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate counterfactual trajectory under alternative treatment u_cf.

        The invariant state s is held fixed (shared between factual and
        counterfactual); only the environment branch changes with u_cf.
        """
        B, T, _ = x.shape
        with torch.no_grad():
            # Factual pass
            s, e_fact, _ = self.encoder(x, u)

            # Counterfactual: re-run env dynamics with u_cf
            e_cf = self.env_dynamics(e_fact, u_cf)
            s_next = self.inv_dynamics(s)

            x_cf_pred = self.decoder(s_next, e_cf)

            # Outcome under counterfactual
            if mask is not None:
                seq_lens = mask.long().sum(dim=1).clamp(min=1) - 1
                s_final = s[torch.arange(B, device=s.device), seq_lens]
            else:
                s_final = s[:, -1]
            outcome_cf = torch.sigmoid(self.outcome_head(s_final))

        return {
            "x_cf_pred": x_cf_pred,
            "outcome_cf": outcome_cf,
            "s": s,
            "e_cf": e_cf,
        }

    # ------------------------------------------------------------------
    def set_grl_alpha(self, alpha: float) -> None:
        """Anneal the gradient reversal strength."""
        self.hosp_adversary.set_alpha(alpha)
        self.trt_adversary.set_alpha(alpha)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: Dict, device: torch.device) -> DisentangledICUModel:
    model = DisentangledICUModel(
        x_dim=cfg["num_features"],
        u_dim=cfg["num_treatments"],
        num_hospitals=cfg["num_hospitals"],
        s_dim=cfg.get("s_dim", 64),
        e_dim=cfg.get("e_dim", 32),
        hidden_dim=cfg.get("hidden_dim", 128),
        enc_layers=cfg.get("enc_layers", 2),
        enc_dropout=cfg.get("enc_dropout", 0.1),
        num_trt_classes=cfg.get("num_trt_classes", 2),
        proj_dim=cfg.get("proj_dim", 64),
        grl_alpha=cfg.get("grl_alpha", 1.0),
        use_e_for_outcome=cfg.get("use_e_for_outcome", False),
    )
    return model.to(device)
