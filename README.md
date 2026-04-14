# Synthica — ICU Trajectory Modeling

Research-grade machine learning system for ICU time-series modeling, targeting ICML-level rigor.

## Objective

- **Treatment-conditioned forecasting** of physiological trajectories
- **Counterfactual trajectory simulation** under modified treatment sequences
- **Hospital-invariant representation learning** via disentangled latent dynamics
- **Cross-domain generalization** across ICUs / hospitals

## Dataset

Download `eicu_final_sequences_for_modeling.csv` and place it at:

```
data/eicu_final_sequences_for_modeling.csv
```

The file name must be exact. All data loading uses:

```python
df = pd.read_csv("data/eicu_final_sequences_for_modeling.csv")
```

## Quick Start

```bash
pip install torch scikit-learn pandas numpy
python src/train.py
```

Results are written to `outputs/results.json`.

## Code Structure

```
src/
  data.py        — Dataset loading, schema inference, environment validation
  model.py       — DisentangledICUModel (s_t / e_t split, GRL adversaries)
  baselines.py   — All baseline models (CRN, RMSN, DCRN, G-Net, etc.)
  losses.py      — Reconstruction, outcome, adversarial, IRM, contrastive losses
  eval.py        — AUROC/AUPRC/ECE, counterfactual proxy eval, competitiveness flag
  train.py       — Main training loop, baseline tuning harness, ablation suite
  utils.py       — Seeding, early stopping, checkpointing
data/
  eicu_final_sequences_for_modeling.csv  (not committed — download separately)
outputs/
  results.json                — Full experiment results
  ece_main_model.json         — Calibration reliability data
  best_baseline_configs.json  — Best hyperparameter configs per baseline
  checkpoints/                — Model checkpoints
```

## Key Features

### 1. Credible Baselines with Tuning
Hyperparameter search over `hidden_dim ∈ {64,128,256}`, `lr ∈ {1e-3,3e-4,1e-4}`, `dropout ∈ {0,0.2,0.5}`.  
Best-of-k-trials config is logged and persisted.

### 2. Environment Validation
Environment/hospital column is inferred programmatically and validated:
- ≥ 3 unique environments required
- Non-trivial patient distribution per environment
- If invalid → hospital invariance mechanisms auto-disabled with prominent warning

### 3. Counterfactual Proxy Evaluation
Clearly labelled as **proxy evaluation** (not causal ground truth):
- Consistency test (same u_t → same rollout)
- Temporal intervention test (intervene at t=k vs t=k+Δ)
- Monotonicity sanity (skipped with logged reason if unknown)
- Stress tests (zero/max/flip treatments)

### 4. Calibration (Mandatory)
Expected Calibration Error (ECE) reported alongside AUROC/AUPRC/accuracy.  
Reliability bin stats saved to disk for plotting.

### 5. Competitiveness Flagging
ΔAUROC = model - best_baseline logged after evaluation.  
Prominent warning emitted if negative.

### 6. Key Ablation: No s_t
`EtOnlyModel` removes s_t entirely — uses only e_t — to validate that the invariant branch carries signal.

### 7. Invariant Latent Collapse Monitoring
s_t variance monitored during training; warning emitted if collapse detected.  
Disentanglement check: performance degradation when e_t branch is zeroed.

## Reproducibility

- Seed: 42 (deterministic splits and training)
- Early stopping on validation AUROC (patience=10)
- Best model checkpointed to `outputs/checkpoints/main_model_best.pt`