# synthica-to-be-moved

# ICML-Level ICU Trajectory Model

A research-grade machine learning system for ICU time-series modelling:
treatment-conditioned forecasting, counterfactual trajectory simulation,
and hospital-invariant representation learning.

---

## Scientific Goal

> *"We learn invariant latent disease dynamics disentangled from hospital and
> treatment effects, enabling robust cross-hospital generalisation and
> counterfactual ICU trajectory simulation."*

---

## Dataset

Download the dataset from Google Drive and place it at:

```
data/eicu_final_sequences_for_modeling.csv
```

Download link: <https://drive.google.com/file/d/1cjf5yjnQXpuMLJSyQlhQkn75g7Pi9GUK/view?usp=sharing>

> **Note**: The file name must remain exactly `eicu_final_sequences_for_modeling.csv`.

---

## Quick Start

```bash
pip install -r requirements.txt
python src/train.py
```

The pipeline will run end-to-end automatically: no manual edits required.

Optional CLI flags:

```bash
python src/train.py --mode all           # train main model + baselines + ablations (default)
python src/train.py --mode main          # main model only (fastest)
python src/train.py --epochs 100         # override epoch count
python src/train.py --split_mode random  # random patient split instead of out-of-hospital
python src/train.py --output_dir runs/   # custom output directory
python src/train.py --no_ablations       # skip ablation studies
python src/train.py --no_baselines       # skip baselines
```

---

## Architecture

```
                ┌────────────────────────────────────────┐
Input           │  GRUEncoder: (x_t ‖ u_t) → h_t        │
(x_t, u_t)  ───▶│      Linear split → s_t , e_t          │
                └────────────┬──────────────┬─────────────┘
                             │              │
                   s_t (invariant)      e_t (environment)
                             │              │
              ┌──────────────▼──┐   ┌───────▼──────────────┐
              │ InvariantDyn.   │   │  EnvironmentDyn.     │
              │ s_{t+1}=GRU(s_t)│   │  e_{t+1}=GRU(e_t,u_t)│
              └──────────────┬──┘   └───────┬──────────────┘
                             │              │
                        ┌────▼──────────────▼────┐
                        │  Decoder: MLP(s‖e)→x̂  │
                        └─────────────────────────┘

Invariance:  HospitalAdversary(GRL → s_t)  → hospital label
             TreatmentAdversary(GRL → s_t) → treatment label
             ContrastiveHead(s_t)           → NT-Xent alignment
Outcome:     OutcomeHead(s_T) → mortality logit
```

---

## Loss Function

```
L = L_reconstruction
  + λ₁ · L_outcome
  + λ₂ · L_hospital_adversary
  + λ₃ · L_treatment_adversary
  + λ₄ · L_contrastive          (optional)
  + λ₅ · L_IRM                  (optional)
```

Discriminators are trained in a **separate** optimiser step (no gradient
reversal needed for their update); the encoder is trained with gradient
reversal to fool them.

---

## Baselines

| Name | Category | Reference |
|------|----------|-----------|
| CRN | Counterfactual | Bica et al., NeurIPS 2020 |
| G-Net | Counterfactual | Li et al., 2021 |
| RMSN | Counterfactual | Lim et al., 2018 |
| DCRN | Counterfactual | — |
| CausalTransformer | Counterfactual | Melnychuk et al., NeurIPS 2022 |
| GTransformer | Counterfactual | Li et al., 2021 |
| MambaCDSP | Counterfactual | Simplified SSM |
| ERM | Hospital invariance | Standard supervised GRU |
| DANN | Hospital invariance | Ganin et al., JMLR 2016 |
| DomainConfusion | Hospital invariance | Tzeng et al., 2015 |
| IRM | Hospital invariance | Arjovsky et al., 2019 |

---

## Experiments

1. **In-distribution** — random 70/10/20 patient split (seed=42)
2. **Out-of-hospital** — train on N-k hospitals, test on held-out k hospitals
3. **Counterfactual** — intervene on `u_t`, measure trajectory divergence &
   outcome sensitivity
4. **Ablation** — remove adversarial heads, e_t branch, contrastive loss

---

## Metrics

- AUROC (primary)
- AUPRC
- Accuracy
- Expected Calibration Error (ECE)
- Domain shift degradation (in-dist vs. OOH AUROC drop)
- Counterfactual sensitivity score

---

## Repository Structure

```
src/
  data.py        — Data loading, schema discovery, sequence construction, splits
  model.py       — DisentangledICUModel (main architecture)
  baselines.py   — 11 SOTA baseline implementations
  train.py       — Training loop, early stopping, checkpointing, ablations
  eval.py        — All evaluation metrics and report generation
  losses.py      — Reconstruction, outcome, adversarial, contrastive, IRM losses
  utils.py       — GradientReversalLayer, logging, seeding, metric helpers
data/
  README.md      — Dataset download instructions
requirements.txt
```

---

## Outputs

After training, the following outputs are written to `outputs/` (default):

```
outputs/
  config.json           — Resolved configuration
  results.json          — Full evaluation report (all models, all experiments)
  train.log             — Training log
  checkpoints/
    best_main.pt        — Best main model checkpoint
    best_<baseline>.pt  — Per-baseline checkpoints
    ablation_*.pt       — Ablation checkpoints
```
