"""
Data loading and preprocessing for ICU trajectory modeling.

Hard rule: dataset must be at  data/eicu_final_sequences_for_modeling.csv
Environment / hospital column is inferred programmatically and validated before use.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from utils import get_logger, set_seed, SEED

logger = get_logger("synthica.data")

DATA_PATH = "data/eicu_final_sequences_for_modeling.csv"

# --- Column-name heuristics ------------------------------------------------

_ENV_KEYWORDS = ["hospital", "hospitalid", "hospital_id", "site", "icu", "unit", "ward"]
_OUTCOME_KEYWORDS = ["mortality", "death", "deceased", "expire", "outcome", "died"]
_TREATMENT_KEYWORDS = ["vent", "vasopressor", "insulin", "treatment", "drug", "medication", "therapy", "intervention"]
_TIME_KEYWORDS = ["time", "hour", "day", "offset", "step", "t_"]
_PATIENT_KEYWORDS = ["patient", "patientid", "patient_id", "subject", "stay", "encounter", "id"]

MIN_ENV_UNIQUE = 3
MIN_PATIENTS_PER_ENV = 10


def _match(col: str, keywords: List[str]) -> bool:
    col_l = col.lower()
    return any(k in col_l for k in keywords)


def _infer_column(cols: List[str], keywords: List[str]) -> Optional[str]:
    for col in cols:
        if _match(col, keywords):
            return col
    return None


def load_raw() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    logger.info("Loading dataset from %s …", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    logger.info("Raw shape: %s", df.shape)
    return df


def _validate_env_column(df: pd.DataFrame, env_col: str) -> Tuple[bool, str]:
    """
    Validate that env_col is usable for out-of-hospital evaluation.
    Returns (is_valid, reason_if_invalid).
    """
    unique_envs = df[env_col].dropna().unique()
    n_envs = len(unique_envs)
    if n_envs < MIN_ENV_UNIQUE:
        return False, f"only {n_envs} unique envs (need ≥ {MIN_ENV_UNIQUE})"

    counts = df[env_col].value_counts()
    min_count = int(counts.min())
    if min_count < MIN_PATIENTS_PER_ENV:
        return False, f"smallest env has {min_count} rows (need ≥ {MIN_PATIENTS_PER_ENV})"

    logger.info("Environment column '%s': %d unique envs", env_col, n_envs)
    for env_val, cnt in counts.items():
        logger.info("  env=%s  rows=%d", env_val, cnt)

    return True, ""


class ICUDatasetConfig:
    """Resolved schema: all inferred column names and flags."""

    def __init__(
        self,
        patient_col: Optional[str],
        time_col: Optional[str],
        outcome_col: Optional[str],
        treatment_cols: List[str],
        feature_cols: List[str],
        env_col: Optional[str],
        env_valid: bool,
        env_invalid_reason: str = "",
    ):
        self.patient_col = patient_col
        self.time_col = time_col
        self.outcome_col = outcome_col
        self.treatment_cols = treatment_cols
        self.feature_cols = feature_cols
        self.env_col = env_col
        self.env_valid = env_valid
        self.env_invalid_reason = env_invalid_reason

    @property
    def hospital_invariance_enabled(self) -> bool:
        return self.env_valid

    def log_summary(self) -> None:
        logger.info("=== Schema Summary ===")
        logger.info("  patient_col  : %s", self.patient_col)
        logger.info("  time_col     : %s", self.time_col)
        logger.info("  outcome_col  : %s", self.outcome_col)
        logger.info("  treatment_cols (%d): %s", len(self.treatment_cols), self.treatment_cols)
        logger.info("  feature_cols (%d): %s", len(self.feature_cols), self.feature_cols[:10])
        logger.info("  env_col      : %s", self.env_col)
        logger.info("  env_valid    : %s", self.env_valid)
        if not self.env_valid:
            logger.warning(
                "!!! HOSPITAL INVARIANCE CLAIM DISABLED — invalid env column: %s !!!",
                self.env_invalid_reason,
            )


def infer_schema(df: pd.DataFrame) -> ICUDatasetConfig:
    cols = list(df.columns)

    # Infer env_col first (higher specificity for hospital/icu columns)
    env_col = _infer_column(cols, _ENV_KEYWORDS)

    # Infer patient_col, excluding env_col to prevent collision
    patient_col = _infer_column(
        [c for c in cols if c != env_col], _PATIENT_KEYWORDS
    )

    time_col = _infer_column(
        [c for c in cols if c not in {env_col, patient_col}], _TIME_KEYWORDS
    )
    outcome_col = _infer_column(cols, _OUTCOME_KEYWORDS)

    # Treatment cols: binary/low-cardinality columns matching keywords
    treatment_cols = [
        c for c in cols
        if _match(c, _TREATMENT_KEYWORDS)
        and c not in {patient_col, time_col, outcome_col, env_col}
        and df[c].nunique() <= 20
    ]

    # Feature cols: everything numeric that is not patient/time/outcome/env/treatment
    exclude = set(filter(None, [patient_col, time_col, outcome_col, env_col] + treatment_cols))
    feature_cols = [
        c for c in cols
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    # Fallback: if no feature cols, use all numeric except special cols
    if not feature_cols:
        feature_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]

    # Validate env column
    env_valid = False
    env_reason = "no env column found"
    if env_col is not None:
        env_valid, env_reason = _validate_env_column(df, env_col)
    else:
        logger.warning("No environment/hospital column detected in schema.")

    cfg = ICUDatasetConfig(
        patient_col=patient_col,
        time_col=time_col,
        outcome_col=outcome_col,
        treatment_cols=treatment_cols,
        feature_cols=feature_cols,
        env_col=env_col,
        env_valid=env_valid,
        env_invalid_reason=env_reason,
    )
    cfg.log_summary()
    return cfg


class ICUSequenceDataset(Dataset):
    """
    Each sample is a fixed-length window: (x_seq, u_seq, env_id, outcome, patient_id).
    x_seq: [T, n_features]
    u_seq: [T, n_treatments]
    env_id: int scalar
    outcome: float scalar
    patient_id: int scalar
    """

    def __init__(
        self,
        sequences: List[dict],
    ):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        return self.sequences[idx]


def _pad_or_truncate(arr: np.ndarray, seq_len: int) -> np.ndarray:
    T = arr.shape[0]
    if T >= seq_len:
        return arr[:seq_len]
    pad = np.zeros((seq_len - T, arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def build_sequences(
    df: pd.DataFrame,
    cfg: ICUDatasetConfig,
    seq_len: int = 24,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False,
) -> Tuple[List[dict], StandardScaler]:
    """Convert flat dataframe → list of sequence dicts."""
    feat_cols = cfg.feature_cols
    treat_cols = cfg.treatment_cols

    if not feat_cols:
        raise ValueError("No feature columns found — check schema inference.")

    # Fill NaN
    df = df.copy()
    df[feat_cols] = df[feat_cols].fillna(0.0)
    if treat_cols:
        df[treat_cols] = df[treat_cols].fillna(0.0)

    # Fit/apply scaler on feature columns
    if scaler is None:
        scaler = StandardScaler()
    # Always fit if not yet fitted (i.e. first call), regardless of fit_scaler flag
    from sklearn.exceptions import NotFittedError
    try:
        scaler.transform(df[feat_cols].values[:1])
    except NotFittedError:
        fit_scaler = True
    if fit_scaler:
        scaler.fit(df[feat_cols].values)
    df[feat_cols] = scaler.transform(df[feat_cols].values)

    # Encode env column
    env_map: dict = {}
    if cfg.env_col and cfg.env_valid:
        env_vals = df[cfg.env_col].fillna("unknown")
        unique_envs = sorted(env_vals.unique())
        env_map = {v: i for i, v in enumerate(unique_envs)}
        df["_env_id"] = env_vals.map(env_map).fillna(0).astype(int)
    else:
        df["_env_id"] = 0

    # Encode outcome
    if cfg.outcome_col:
        outcome_vals = df[cfg.outcome_col].fillna(0.0)
        # If string-like, encode
        if outcome_vals.dtype == object:
            outcome_vals = (outcome_vals.astype(str).str.lower().isin(["1", "true", "yes", "died", "death", "1.0"])).astype(float)
        df["_outcome"] = outcome_vals.values
    else:
        df["_outcome"] = 0.0

    # Group by patient
    if cfg.patient_col and cfg.patient_col in df.columns:
        groups = df.groupby(cfg.patient_col)
    else:
        # Treat each row as its own patient (degenerate)
        df["_fake_patient"] = np.arange(len(df))
        groups = df.groupby("_fake_patient")

    sequences = []
    for pid, group in groups:
        if cfg.time_col and cfg.time_col in group.columns:
            group = group.sort_values(cfg.time_col)

        x = group[feat_cols].values.astype(np.float32)
        u = group[treat_cols].values.astype(np.float32) if treat_cols else np.zeros((len(group), 1), dtype=np.float32)
        env_id = int(group["_env_id"].iloc[0])
        outcome = float(group["_outcome"].iloc[-1])  # last row outcome

        x = _pad_or_truncate(x, seq_len)
        u = _pad_or_truncate(u, seq_len)

        sequences.append({
            "x": torch.tensor(x, dtype=torch.float32),
            "u": torch.tensor(u, dtype=torch.float32),
            "env_id": torch.tensor(env_id, dtype=torch.long),
            "outcome": torch.tensor(outcome, dtype=torch.float32),
            "patient_id": torch.tensor(hash(pid) % (2**31), dtype=torch.long),
        })

    logger.info("Built %d sequences (seq_len=%d, n_feat=%d, n_treat=%d)",
                len(sequences), seq_len, len(feat_cols), len(treat_cols) if treat_cols else 1)
    return sequences, scaler


def split_random(sequences: List[dict], val_frac: float = 0.15, test_frac: float = 0.15) -> Tuple[List, List, List]:
    set_seed(SEED)
    n = len(sequences)
    idx = np.random.permutation(n)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return (
        [sequences[i] for i in train_idx],
        [sequences[i] for i in val_idx],
        [sequences[i] for i in test_idx],
    )


def split_out_of_hospital(
    sequences: List[dict], df: pd.DataFrame, cfg: ICUDatasetConfig
) -> Tuple[List, List, List, List]:
    """
    Hold out one hospital for OOH test; remaining splits randomly.
    Returns train, val, iid_test, ooh_test.
    """
    if not cfg.env_valid or cfg.env_col is None:
        raise ValueError("env column not valid — cannot do OOH split")

    env_ids = [s["env_id"].item() for s in sequences]
    unique_envs = sorted(set(env_ids))
    set_seed(SEED)
    # hold out last env (deterministic by sort)
    held_env = unique_envs[-1]
    logger.info("Out-of-hospital split: holding out env_id=%d", held_env)

    in_dist = [s for s in sequences if s["env_id"].item() != held_env]
    ooh_test = [s for s in sequences if s["env_id"].item() == held_env]

    train, val, iid_test = split_random(in_dist)
    return train, val, iid_test, ooh_test


def make_loaders(
    train_seqs: List[dict],
    val_seqs: List[dict],
    test_seqs: List[dict],
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    def _loader(seqs, shuffle):
        return DataLoader(
            ICUSequenceDataset(seqs),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )

    return _loader(train_seqs, True), _loader(val_seqs, False), _loader(test_seqs, False)
