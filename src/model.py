"""
Disentangled ICU Model: s_t (invariant) + e_t (environment/treatment) dynamics.

Key components:
  - GRUEncoder: encodes (x_t, u_t) → hidden → split into s_t, e_t
  - InvariantDynamics: s_{t+1} = f_inv(s_t)
  - EnvironmentDynamics: e_{t+1} = f_env(e_t, u_t)
  - Decoder: x̂_{t+1} = Dec(s_{t+1}, e_{t+1})
  - OutcomeHead: ŷ = Head(s_T)
  - HospitalAdversary: grad-reversed domain classifier on s_t
  - TreatmentAdversary: grad-reversed treatment classifier on s_t
  - ContrastiveHead: optional NT-Xent projection head
  - Collapse monitoring: s_t variance + MI proxy probe
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from utils import get_logger

logger = get_logger("synthica.model")


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.alpha, None


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

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
        input_dim: int,
        treatment_dim: int,
        hidden_dim: int = 128,
        s_dim: int = 64,
        e_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.s_dim = s_dim
        self.e_dim = e_dim
        self.input_proj = nn.Linear(input_dim + treatment_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.to_s = nn.Linear(hidden_dim, s_dim)
        self.to_e = nn.Linear(hidden_dim, e_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, T, input_dim]
        u: [B, T, treatment_dim]
        Returns:
          s_seq: [B, T, s_dim]
          e_seq: [B, T, e_dim]
          h_seq: [B, T, hidden_dim]  (for ablation / probing)
        """
        xu = torch.cat([x, u], dim=-1)
        xu = self.input_proj(xu)
        h_seq, _ = self.gru(xu)  # [B, T, hidden_dim]
        h_seq = self.dropout(h_seq)
        s_seq = self.to_s(h_seq)  # [B, T, s_dim]
        e_seq = self.to_e(h_seq)  # [B, T, e_dim]
        return s_seq, e_seq, h_seq


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

class InvariantDynamics(nn.Module):
    """s_{t+1} = f_inv(s_t)  — hospital-independent."""

    def __init__(self, s_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, s_dim),
        )

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        return self.net(s_t)


class EnvironmentDynamics(nn.Module):
    """e_{t+1} = f_env(e_t, u_t)  — treatment/hospital dependent."""

    def __init__(self, e_dim: int, treatment_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(e_dim + treatment_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, e_dim),
        )

    def forward(self, e_t: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        eu = torch.cat([e_t, u_t], dim=-1)
        return self.net(eu)


# ---------------------------------------------------------------------------
# Decoder & Heads
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """x̂_{t+1} = Dec(s_{t+1}, e_{t+1})."""

    def __init__(self, s_dim: int, e_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + e_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
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
    """ŷ = Head(s_T)  — classification from final invariant state."""

    def __init__(self, s_dim: int, hidden_dim: int = 64, n_classes: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, s_T: torch.Tensor) -> torch.Tensor:
        return self.net(s_T)


class HospitalAdversary(nn.Module):
    """Grad-reversed hospital domain classifier on s_t."""

    def __init__(self, s_dim: int, n_hospitals: int, hidden_dim: int = 64, alpha: float = 1.0):
        super().__init__()
        self.grl = GradientReversal(alpha)
        self.classifier = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_hospitals),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.grl(s))


class TreatmentAdversary(nn.Module):
    """Grad-reversed treatment classifier on s_t."""

    def __init__(self, s_dim: int, n_treatments: int, hidden_dim: int = 64, alpha: float = 1.0):
        super().__init__()
        self.grl = GradientReversal(alpha)
        self.classifier = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_treatments),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.grl(s))


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
        return self.net(s)


class MIProbe(nn.Module):
    """
    Linear probe to predict x_t from s_t only.
    Used as a mutual-information proxy to detect invariant state collapse.
    """

    def __init__(self, s_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(s_dim, output_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(s)


# ---------------------------------------------------------------------------
# Main disentangled model
        return F.normalize(self.net(s), dim=-1)


# ---------------------------------------------------------------------------
# Full Disentangled ICU Model
# ---------------------------------------------------------------------------

class DisentangledICUModel(nn.Module):
    """
    Full disentangled ICU trajectory model.

    Learns:
      s_t: invariant disease state (pushed to be hospital-independent via adversaries)
      e_t: environment + treatment state (captures hospital/treatment variation)

    Enables:
      - cross-hospital generalisation
      - counterfactual trajectory rollout
      - robust clinical outcome prediction
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
        input_dim: int,
        treatment_dim: int,
        n_hospitals: int = 4,
        hidden_dim: int = 128,
        s_dim: int = 64,
        e_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        grl_alpha: float = 1.0,
        use_hospital_adv: bool = True,
        use_treatment_adv: bool = True,
        use_contrastive: bool = True,
        use_irm: bool = True,
    ):
        super().__init__()
        self.s_dim = s_dim
        self.e_dim = e_dim
        self.input_dim = input_dim
        self.treatment_dim = treatment_dim
        self.use_hospital_adv = use_hospital_adv
        self.use_treatment_adv = use_treatment_adv
        self.use_contrastive = use_contrastive
        self.use_irm = use_irm

        self.encoder = GRUEncoder(input_dim, treatment_dim, hidden_dim, s_dim, e_dim, num_layers, dropout)
        self.inv_dynamics = InvariantDynamics(s_dim, hidden_dim // 2)
        self.env_dynamics = EnvironmentDynamics(e_dim, treatment_dim, hidden_dim // 2)
        self.decoder = Decoder(s_dim, e_dim, input_dim, hidden_dim)
        self.outcome_head = OutcomeHead(s_dim, hidden_dim // 2)
        self.mi_probe = MIProbe(s_dim, input_dim)

        if use_hospital_adv:
            self.hospital_adv = HospitalAdversary(s_dim, n_hospitals, hidden_dim // 2, grl_alpha)
        if use_treatment_adv:
            # Binary treatment adversary (ventilation / vasopressor etc.)
            n_treat_classes = max(2, treatment_dim)
            self.treatment_adv = TreatmentAdversary(s_dim, n_treat_classes, hidden_dim // 2, grl_alpha)
        if use_contrastive:
            self.contrastive_head = ContrastiveHead(s_dim, s_dim // 2)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        env_ids: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        x: [B, T, input_dim]
        u: [B, T, treatment_dim]
        Returns dict of tensors.
        """
        B, T, _ = x.shape

        # Encode
        s_seq, e_seq, h_seq = self.encoder(x, u)  # [B, T, s/e_dim]

        # --- Dynamics unrolling ---
        # Predict next-step using dynamics modules
        s_next_list, e_next_list = [], []
        for t in range(T - 1):
            s_next_t = self.inv_dynamics(s_seq[:, t, :])       # [B, s_dim]
            e_next_t = self.env_dynamics(e_seq[:, t, :], u[:, t, :])  # [B, e_dim]
            s_next_list.append(s_next_t)
            e_next_list.append(e_next_t)
        s_next = torch.stack(s_next_list, dim=1)  # [B, T-1, s_dim]
        e_next = torch.stack(e_next_list, dim=1)  # [B, T-1, e_dim]

        # Decode next-step predictions
        x_pred = self.decoder(s_next, e_next)  # [B, T-1, input_dim]
        x_true = x[:, 1:, :]                   # [B, T-1, input_dim]

        # Outcome from final invariant state
        y_pred = self.outcome_head(s_seq[:, -1, :])  # [B, 1]

        # MI probe: predict x_t from s_t only (used for collapse monitoring)
        mi_pred = self.mi_probe(s_seq[:, :-1, :])  # [B, T-1, input_dim]

        out = {
            "x_pred": x_pred,
            "x_true": x_true,
            "y_pred": y_pred,
            "s_seq": s_seq,
            "e_seq": e_seq,
            "s_next": s_next,
            "e_next": e_next,
            "mi_pred": mi_pred,
        }

        # Adversarial heads (on mean s across time)
        s_mean = s_seq.mean(dim=1)  # [B, s_dim]
        if self.use_hospital_adv and hasattr(self, "hospital_adv"):
            out["hosp_logits"] = self.hospital_adv(s_mean)
        if self.use_treatment_adv and hasattr(self, "treatment_adv"):
            out["treat_logits"] = self.treatment_adv(s_mean)
        if self.use_contrastive and hasattr(self, "contrastive_head"):
            out["s_proj"] = self.contrastive_head(s_mean)

        return out

    @torch.no_grad()
    def rollout_counterfactual(
        self,
        x0: torch.Tensor,
        u_cf: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        """
        Counterfactual rollout: fix s_0 from x0, apply modified u_cf.
        x0: [B, 1, input_dim]
        u_cf: [B, n_steps, treatment_dim]
        Returns x_cf: [B, n_steps, input_dim]
        """
        # Encode initial state
        u0 = u_cf[:, :1, :]
        s_seq, e_seq, _ = self.encoder(x0, u0)
        s_t = s_seq[:, 0, :]
        e_t = e_seq[:, 0, :]

        preds = []
        for t in range(n_steps):
            s_next_t = self.inv_dynamics(s_t)
            e_next_t = self.env_dynamics(e_t, u_cf[:, t, :])
            x_next = self.decoder(s_next_t, e_next_t)
            preds.append(x_next)
            s_t = s_next_t
            e_t = e_next_t

        return torch.stack(preds, dim=1)  # [B, n_steps, input_dim]

    def monitor_collapse(self, s_seq: torch.Tensor) -> dict:
        """
        Compute collapse monitoring statistics for s_t.
        s_seq: [B, T, s_dim]
        Returns dict with variance and std metrics.
        """
        B, T, D = s_seq.shape
        s_flat = s_seq.reshape(-1, D)  # [B*T, D]
        var_per_dim = s_flat.var(dim=0)  # [D]
        mean_var = var_per_dim.mean().item()
        min_var = var_per_dim.min().item()
        return {
            "s_t_mean_var": mean_var,
            "s_t_min_var": min_var,
            "s_t_std_mean": float(s_flat.std(dim=0).mean().item()),
        }

    def set_grl_alpha(self, alpha: float) -> None:
        """Anneal the gradient reversal strength."""
        if hasattr(self, "hospital_adv"):
            self.hospital_adv.set_alpha(alpha)
        if hasattr(self, "treatment_adv"):
            self.treatment_adv.set_alpha(alpha)


# ---------------------------------------------------------------------------
# Ablation variant: no s_t — uses only e_t
# ---------------------------------------------------------------------------

class EtOnlyModel(nn.Module):
    """
    Ablation: remove s_t entirely; predict next-step and outcome from e_t only.
    Tests whether the 'invariant' branch actually carries signal.
    """

    def __init__(
        self,
        input_dim: int,
        treatment_dim: int,
        hidden_dim: int = 128,
        e_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.e_dim = e_dim
        self.input_dim = input_dim

        self.input_proj = nn.Linear(input_dim + treatment_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.to_e = nn.Linear(hidden_dim, e_dim)
        self.env_dynamics = EnvironmentDynamics(e_dim, treatment_dim, hidden_dim // 2)
        self.decoder = nn.Sequential(
            nn.Linear(e_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.outcome_head = nn.Sequential(
            nn.Linear(e_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, u: torch.Tensor, env_ids=None) -> dict:
        B, T, _ = x.shape
        xu = torch.cat([x, u], dim=-1)
        xu = self.input_proj(xu)
        h_seq, _ = self.gru(xu)
        h_seq = self.dropout(h_seq)
        e_seq = self.to_e(h_seq)  # [B, T, e_dim]

        e_next_list = []
        for t in range(T - 1):
            e_next_t = self.env_dynamics(e_seq[:, t, :], u[:, t, :])
            e_next_list.append(e_next_t)
        e_next = torch.stack(e_next_list, dim=1)  # [B, T-1, e_dim]

        x_pred = self.decoder(e_next)        # [B, T-1, input_dim]
        x_true = x[:, 1:, :]
        y_pred = self.outcome_head(e_seq[:, -1, :])  # [B, 1]

        return {
            "x_pred": x_pred,
            "x_true": x_true,
            "y_pred": y_pred,
            "s_seq": torch.zeros(B, T, 1, device=x.device),  # dummy
            "e_seq": e_seq,
        }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(cfg: Dict, device: torch.device) -> DisentangledICUModel:
    model = DisentangledICUModel(
        input_dim=cfg["num_features"],
        treatment_dim=cfg["num_treatments"],
        n_hospitals=cfg["num_hospitals"],
        s_dim=cfg.get("s_dim", 64),
        e_dim=cfg.get("e_dim", 32),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_layers=cfg.get("enc_layers", 2),
        dropout=cfg.get("enc_dropout", 0.1),
        grl_alpha=cfg.get("grl_alpha", 1.0),
    )
    return model.to(device)
