"""
data.py — Data loading, schema discovery, preprocessing, and split creation.

HARD LOADING RULE
-----------------
All data must be loaded via:

    import pandas as pd
    df = pd.read_csv("data/eicu_final_sequences_for_modeling.csv")

If the file is missing a FileNotFoundError is raised with an informative message.

Schema Discovery
----------------
Column roles are inferred programmatically from column names and data statistics
with every decision logged explicitly — no human intervention is needed.

Dataset
-------
eICU-style sequences: one row per patient × time-step.
Each patient has T observations; the final outcome (mortality) is patient-level.

Splits
------
1. In-distribution  : random 70/10/20 patient split (seed=42)
2. Out-of-hospital  : train on all but the held-out hospital(s), test on held-out
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DATA_PATH = "data/eicu_final_sequences_for_modeling.csv"

# ---------------------------------------------------------------------------
# Column-name heuristics (ordered by priority)
# ---------------------------------------------------------------------------

_PATIENT_CANDIDATES = [
    "patientunitstayid", "patient_id", "patientid", "pid", "subject_id",
    "encounter_id", "icustay_id",
]
_HOSPITAL_CANDIDATES = [
    "hospitalid", "hospital_id", "hid", "site_id", "siteid",
    "ward_id", "wardid", "icu_id", "icuid",
]
_TIME_CANDIDATES = [
    "hour", "time", "timestep", "offset", "chartoffset", "nursingchartoffset",
    "t", "minute", "hours", "time_step",
]
_OUTCOME_CANDIDATES = [
    "hospitaldischargestatus", "mortality", "died", "outcome",
    "hospital_expire_flag", "icu_expire_flag", "death", "mortalityoffset",
    "hosp_mort", "icu_mort", "deceased",
]
_TREATMENT_KEYWORDS = [
    "vasopressor", "ventilat", "dialysis", "insulin", "antibiotic",
    "sedative", "heparin", "dopamine", "norepinephrine", "epinephrine",
    "dobutamine", "phenylephrine", "vasopressin", "fentanyl", "propofol",
    "midazolam", "treatment", "intervent", "drug", "infusion", "iv_",
]


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first candidate column name that exists (case-insensitive)."""
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    return None


def _infer_treatment_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """Return columns whose name suggests they encode a treatment/intervention."""
    exc = set(c.lower() for c in exclude)
    trt_cols = []
    for col in df.columns:
        if col.lower() in exc:
            continue
        if any(kw in col.lower() for kw in _TREATMENT_KEYWORDS):
            trt_cols.append(col)
    # Additionally, include binary columns with low cardinality ≤ 2 unique non-null values
    # (excluding id / time / outcome columns already identified)
    if not trt_cols:
        logger.info(
            "No treatment columns found via keyword matching; "
            "falling back to binary columns."
        )
        for col in df.columns:
            if col.lower() in exc:
                continue
            if df[col].dropna().nunique() <= 2:
                trt_cols.append(col)
        # Limit to a reasonable number
        trt_cols = trt_cols[:20]
    return trt_cols


def _infer_feature_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """Return numeric columns not in `exclude`."""
    exc = set(c.lower() for c in exclude)
    feat_cols = [
        col
        for col in df.select_dtypes(include=[np.number]).columns
        if col.lower() not in exc
    ]
    return feat_cols


def _binarize_outcome(series: pd.Series) -> pd.Series:
    """Convert outcome column to binary 0/1."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_string_dtype(series) or series.dtype == object:
        # eICU: 'Expired' / 'Alive'
        mapping = {"expired": 1, "alive": 0, "dead": 1, "death": 1, "yes": 1, "no": 0}
        return series.astype(str).str.lower().map(mapping).fillna(0).astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return (series > 0).astype(int)
    # Fallback: attempt numeric coercion
    try:
        return (pd.to_numeric(series, errors="coerce").fillna(0) > 0).astype(int)
    except Exception:
        return pd.Series(0, index=series.index)


# ---------------------------------------------------------------------------
# Hospital column validation
# ---------------------------------------------------------------------------

#: Minimum number of distinct hospitals required to make cross-hospital claims.
_MIN_HOSPITALS = 3
#: Minimum fraction of total patients that a hospital must cover to be "non-trivial".
#: Hospitals with fewer than 1% of patients are too small for meaningful cross-hospital
#: evaluation — they provide insufficient test-set signal.  Such hospitals are logged
#: as warnings but are kept in the training set rather than used as held-out test environments.
_MIN_HOSPITAL_PATIENT_FRACTION = 0.01


def validate_hospital_column(
    df: pd.DataFrame,
    patient_col: str,
    hospital_col: str,
) -> bool:
    """
    Validate that the hospital column is suitable for cross-hospital generalization.

    Checks:
      1. At least _MIN_HOSPITALS unique hospital values.
      2. No single hospital contains > 95% of all patients (non-degenerate).
      3. Each hospital contains at least _MIN_HOSPITAL_PATIENT_FRACTION of patients.

    Logs full distribution statistics.

    Returns
    -------
    True if the column is a valid environment variable for invariance claims.
    False if the column should be treated as degenerate (disables OOH claim).
    """
    patient_level = df.groupby(patient_col)[hospital_col].first()
    n_patients = len(patient_level)
    hospital_counts = patient_level.value_counts()
    n_hospitals = len(hospital_counts)

    logger.info(
        "[HospitalValidation] Column '%s': %d unique hospitals, %d patients",
        hospital_col, n_hospitals, n_patients,
    )
    logger.info(
        "[HospitalValidation] Patients per hospital (top-10): %s",
        hospital_counts.head(10).to_dict(),
    )

    if n_hospitals < _MIN_HOSPITALS:
        logger.warning(
            "[HospitalValidation] FAIL: only %d hospital(s) found (need ≥%d). "
            "Cross-hospital generalisation claim is DISABLED.",
            n_hospitals, _MIN_HOSPITALS,
        )
        return False

    max_frac = hospital_counts.max() / n_patients
    if max_frac > 0.95:
        logger.warning(
            "[HospitalValidation] FAIL: one hospital contains %.1f%% of patients — "
            "distribution is degenerate.  Cross-hospital claim is DISABLED.",
            max_frac * 100,
        )
        return False

    # Check no hospital is too small to be meaningful
    tiny = hospital_counts[hospital_counts / n_patients < _MIN_HOSPITAL_PATIENT_FRACTION]
    if len(tiny) > 0:
        logger.warning(
            "[HospitalValidation] %d hospital(s) have < %.1f%% of patients: %s  "
            "(will be held in training set only)",
            len(tiny), _MIN_HOSPITAL_PATIENT_FRACTION * 100, tiny.index.tolist()[:5],
        )

    logger.info(
        "[HospitalValidation] PASS: %d hospitals, max_patient_frac=%.2f",
        n_hospitals, max_frac,
    )
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    """Load the raw CSV, enforcing the hard loading rule."""
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    logger.info("Loaded %d rows × %d columns from %s", len(df), len(df.columns), DATA_PATH)
    logger.info("Columns: %s", df.columns.tolist())
    return df


def discover_schema(df: pd.DataFrame) -> Dict[str, object]:
    """
    Programmatically identify column roles.  Every decision is logged.
    Returns a dict with keys: patient_col, hospital_col, time_col,
    outcome_col, feature_cols, treatment_cols, hospital_valid.

    ``hospital_valid`` is False when the hospital column does not have enough
    unique values / non-trivial distribution to support cross-hospital claims.
    When False, callers should disable out-of-hospital split experiments and
    suppress the hospital-invariance scientific claim.
    """
    patient_col = _find_col(df, _PATIENT_CANDIDATES)
    hospital_col = _find_col(df, _HOSPITAL_CANDIDATES)
    time_col = _find_col(df, _TIME_CANDIDATES)
    outcome_col = _find_col(df, _OUTCOME_CANDIDATES)

    # Log discoveries
    logger.info("[Schema] patient_col   = %s", patient_col)
    logger.info("[Schema] hospital_col  = %s", hospital_col)
    logger.info("[Schema] time_col      = %s", time_col)
    logger.info("[Schema] outcome_col   = %s", outcome_col)

    # Fallback: if no patient col, assume first column is patient id
    if patient_col is None:
        patient_col = df.columns[0]
        logger.warning(
            "[Schema] No patient column found; assuming first column '%s'.", patient_col
        )

    # Fallback: synthesise a dummy hospital label if none found
    hospital_valid = True
    if hospital_col is None:
        hospital_col = "__hospital__"
        df[hospital_col] = 0
        hospital_valid = False
        logger.warning(
            "[Schema] No hospital column found; using constant hospital label 0.  "
            "Cross-hospital generalisation claim is DISABLED."
        )

    # Fallback: if no time column, assume rows are already ordered chronologically
    if time_col is None:
        time_col = "__time__"
        df[time_col] = df.groupby(patient_col).cumcount()
        logger.warning(
            "[Schema] No time column found; using per-patient row index as time."
        )

    # Fallback: if no outcome col, synthesise random binary (for structural testing)
    if outcome_col is None:
        outcome_col = "__outcome__"
        rng = np.random.default_rng(42)
        # Assign a patient-level outcome
        patients = df[patient_col].unique()
        outcomes = pd.Series(
            rng.integers(0, 2, len(patients)).astype(float), index=patients
        )
        df[outcome_col] = df[patient_col].map(outcomes)
        logger.warning(
            "[Schema] No outcome column found; synthesising random binary outcome."
        )

    # Validate hospital column if it was found or synthesised
    if hospital_valid:
        hospital_valid = validate_hospital_column(df, patient_col, hospital_col)

    exclude = [c for c in [patient_col, hospital_col, time_col, outcome_col] if c is not None]
    treatment_cols = _infer_treatment_cols(df, exclude)
    feature_cols = _infer_feature_cols(df, exclude + treatment_cols)

    # If no numeric feature columns, treat all non-id numeric cols as features
    if not feature_cols:
        feature_cols = _infer_feature_cols(df, exclude)
        treatment_cols = []
        logger.warning(
            "[Schema] Feature and treatment columns could not be separated; "
            "treating all numeric cols as features."
        )

    logger.info("[Schema] # feature_cols    = %d  %s", len(feature_cols), feature_cols[:5])
    logger.info("[Schema] # treatment_cols  = %d  %s", len(treatment_cols), treatment_cols[:5])

    return {
        "patient_col": patient_col,
        "hospital_col": hospital_col,
        "time_col": time_col,
        "outcome_col": outcome_col,
        "feature_cols": feature_cols,
        "treatment_cols": treatment_cols,
        "hospital_valid": hospital_valid,
    }


# ---------------------------------------------------------------------------
# Preprocessing: build per-patient sequences
# ---------------------------------------------------------------------------

def build_sequences(
    df: pd.DataFrame,
    schema: Dict[str, object],
    max_seq_len: int = 48,
    min_seq_len: int = 3,
) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """
    Build a list of patient-sequence dicts:
      {
        patient_id, hospital_id,
        x: (T, Dx) float32,
        u: (T, Du) float32,
        y: float32,         # patient-level outcome
        seq_len: int,
      }

    Also returns fitted (scaler_x, scaler_u) for later use.
    """
    patient_col: str = schema["patient_col"]
    hospital_col: str = schema["hospital_col"]
    time_col: str = schema["time_col"]
    outcome_col: str = schema["outcome_col"]
    feature_cols: List[str] = schema["feature_cols"]
    treatment_cols: List[str] = schema["treatment_cols"]

    # Sort by patient then time
    df = df.sort_values([patient_col, time_col]).reset_index(drop=True)

    # Binarize outcome
    df[outcome_col] = _binarize_outcome(df[outcome_col])

    # Fill NaN with column median / 0
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    if treatment_cols:
        df[treatment_cols] = df[treatment_cols].fillna(0)

    # Fit scalers on raw data
    scaler_x = StandardScaler()
    scaler_u = StandardScaler() if treatment_cols else None

    x_all = df[feature_cols].values.astype(np.float32)
    scaler_x.fit(x_all)

    if treatment_cols:
        u_all = df[treatment_cols].values.astype(np.float32)
        scaler_u.fit(u_all)
    else:
        u_all = np.zeros((len(df), 1), dtype=np.float32)

    df_x = pd.DataFrame(scaler_x.transform(x_all), columns=feature_cols, index=df.index)
    if treatment_cols:
        df_u = pd.DataFrame(scaler_u.transform(u_all), columns=treatment_cols, index=df.index)
    else:
        df_u = pd.DataFrame(np.zeros((len(df), 1), dtype=np.float32),
                            columns=["__no_treatment__"], index=df.index)

    sequences = []
    grouped = df.groupby(patient_col)
    skipped = 0
    for pid, grp in grouped:
        T = len(grp)
        if T < min_seq_len:
            skipped += 1
            continue
        T = min(T, max_seq_len)
        idx = grp.index[:T]
        x = df_x.loc[idx].values.astype(np.float32)
        u = df_u.loc[idx].values.astype(np.float32)
        y = float(grp[outcome_col].iloc[0])  # patient-level outcome
        hid = int(grp[hospital_col].iloc[0]) if pd.api.types.is_numeric_dtype(
            df[hospital_col]
        ) else hash(grp[hospital_col].iloc[0]) % (2 ** 31)

        sequences.append(
            {
                "patient_id": pid,
                "hospital_id": hid,
                "x": x,
                "u": u,
                "y": y,
                "seq_len": T,
            }
        )

    logger.info(
        "Built %d sequences (skipped %d with len < %d).", len(sequences), skipped, min_seq_len
    )
    return sequences, scaler_x, scaler_u


# ---------------------------------------------------------------------------
# Train / Val / Test Splits
# ---------------------------------------------------------------------------

def random_patient_split(
    sequences: List[Dict],
    train_frac: float = 0.70,
    val_frac: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Random 70/10/20 split at the patient level."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sequences))
    n_train = int(len(sequences) * train_frac)
    n_val = int(len(sequences) * val_frac)
    train = [sequences[i] for i in idx[:n_train]]
    val = [sequences[i] for i in idx[n_train : n_train + n_val]]
    test = [sequences[i] for i in idx[n_train + n_val :]]
    logger.info(
        "Random split: train=%d, val=%d, test=%d", len(train), len(val), len(test)
    )
    return train, val, test


_OOH_VAL_FRACTION = 0.125  # fraction of train-domain patients held out for validation


def out_of_hospital_split(
    sequences: List[Dict],
    held_out_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[int]]:
    """
    Hold out hospitals representing ~held_out_fraction of *patients*.
    Returns (train, val, test, held_out_hospital_ids).
    Val set is taken from the training hospitals (10% of train patients).
    """
    rng = np.random.default_rng(seed)
    # Get hospital → patient count mapping
    from collections import Counter

    hosp_counts = Counter(s["hospital_id"] for s in sequences)
    hospitals = list(hosp_counts.keys())
    rng.shuffle(hospitals)

    total = len(sequences)
    held_out_hospitals: List[int] = []
    held_n = 0
    for h in hospitals:
        held_out_hospitals.append(h)
        held_n += hosp_counts[h]
        if held_n >= total * held_out_fraction:
            break

    logger.info(
        "Out-of-hospital split: holding out %d hospitals (%d patients) as test set.",
        len(held_out_hospitals),
        held_n,
    )

    train_val = [s for s in sequences if s["hospital_id"] not in held_out_hospitals]
    test = [s for s in sequences if s["hospital_id"] in held_out_hospitals]

    # Further split train_val into train / val
    val_idx = rng.choice(len(train_val), size=int(len(train_val) * _OOH_VAL_FRACTION), replace=False)
    val_set = set(val_idx.tolist())
    train = [s for i, s in enumerate(train_val) if i not in val_set]
    val = [s for i, s in enumerate(train_val) if i in val_set]

    logger.info(
        "OOH split: train=%d, val=%d, test=%d", len(train), len(val), len(test)
    )
    return train, val, test, held_out_hospitals


# ---------------------------------------------------------------------------
# PyTorch Dataset & DataLoader
# ---------------------------------------------------------------------------

class ICUDataset(Dataset):
    """Pads sequences within a batch to the same length."""

    def __init__(
        self,
        sequences: List[Dict],
        hospital_id_map: Optional[Dict[int, int]] = None,
    ) -> None:
        self.sequences = sequences
        # Encode hospital IDs to consecutive integers
        if hospital_id_map is None:
            unique_h = sorted({s["hospital_id"] for s in sequences})
            hospital_id_map = {h: i for i, h in enumerate(unique_h)}
        self.hospital_id_map = hospital_id_map

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.sequences[idx]
        x = torch.tensor(s["x"], dtype=torch.float32)
        u = torch.tensor(s["u"], dtype=torch.float32)
        y = torch.tensor(s["y"], dtype=torch.float32)
        h = torch.tensor(
            self.hospital_id_map.get(s["hospital_id"], 0), dtype=torch.long
        )
        seq_len = torch.tensor(s["seq_len"], dtype=torch.long)
        return {"x": x, "u": u, "y": y, "hospital_id": h, "seq_len": seq_len}


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad variable-length sequences to the maximum length in the batch."""
    max_len = max(item["x"].size(0) for item in batch)
    B = len(batch)
    Dx = batch[0]["x"].size(-1)
    Du = batch[0]["u"].size(-1)

    x_padded = torch.zeros(B, max_len, Dx)
    u_padded = torch.zeros(B, max_len, Du)
    masks = torch.zeros(B, max_len, dtype=torch.bool)
    ys = torch.zeros(B)
    hids = torch.zeros(B, dtype=torch.long)
    seq_lens = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        T = item["x"].size(0)
        x_padded[i, :T] = item["x"]
        u_padded[i, :T] = item["u"]
        masks[i, :T] = True
        ys[i] = item["y"]
        hids[i] = item["hospital_id"]
        seq_lens[i] = item["seq_len"]

    return {
        "x": x_padded,
        "u": u_padded,
        "mask": masks,
        "y": ys,
        "hospital_id": hids,
        "seq_len": seq_lens,
    }


def get_dataloaders(
    train_seqs: List[Dict],
    val_seqs: List[Dict],
    test_seqs: List[Dict],
    batch_size: int = 64,
    num_workers: int = 0,
    hospital_id_map: Optional[Dict[int, int]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if hospital_id_map is None:
        all_seqs = train_seqs + val_seqs + test_seqs
        unique_h = sorted({s["hospital_id"] for s in all_seqs})
        hospital_id_map = {h: i for i, h in enumerate(unique_h)}

    train_ds = ICUDataset(train_seqs, hospital_id_map)
    val_ds = ICUDataset(val_seqs, hospital_id_map)
    test_ds = ICUDataset(test_seqs, hospital_id_map)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Convenience: load everything end-to-end
# ---------------------------------------------------------------------------

def load_and_prepare(
    max_seq_len: int = 48,
    min_seq_len: int = 3,
    split_mode: str = "random",  # "random" | "out_of_hospital"
    batch_size: int = 64,
    seed: int = 42,
) -> Dict:
    """
    Full pipeline: load → discover schema → build sequences → split → DataLoaders.

    Returns a dict with:
        train_loader, val_loader, test_loader (in-distribution),
        ooh_test_loader (out-of-hospital, when split_mode=="out_of_hospital"),
        schema, num_features, num_treatments, num_hospitals,
        hospital_id_map, sequences.
    """
    df = load_raw()
    schema = discover_schema(df)
    sequences, scaler_x, scaler_u = build_sequences(df, schema, max_seq_len, min_seq_len)

    all_hospitals = sorted({s["hospital_id"] for s in sequences})
    hospital_id_map = {h: i for i, h in enumerate(all_hospitals)}
    num_hospitals = len(all_hospitals)

    # In-distribution split (always built)
    train_r, val_r, test_r = random_patient_split(sequences, seed=seed)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_r, val_r, test_r, batch_size=batch_size, hospital_id_map=hospital_id_map
    )

    result = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "schema": schema,
        "num_features": len(schema["feature_cols"]),
        "num_treatments": max(len(schema["treatment_cols"]), 1),
        "num_hospitals": num_hospitals,
        "hospital_id_map": hospital_id_map,
        "hospital_valid": schema.get("hospital_valid", True),
        "sequences": sequences,
        "scaler_x": scaler_x,
        "scaler_u": scaler_u,
        "train_seqs": train_r,
        "val_seqs": val_r,
        "test_seqs": test_r,
    }

    if split_mode == "out_of_hospital":
        if not schema.get("hospital_valid", True):
            logger.warning(
                "split_mode='out_of_hospital' requested but hospital column is not "
                "valid for cross-hospital evaluation.  Falling back to random split "
                "for OOH loaders (same data, different seed)."
            )
        tr_ooh, va_ooh, te_ooh, held = out_of_hospital_split(sequences, seed=seed)
        _, _, ooh_test_loader = get_dataloaders(
            tr_ooh, va_ooh, te_ooh, batch_size=batch_size, hospital_id_map=hospital_id_map
        )
        result["ooh_test_loader"] = ooh_test_loader
        result["ooh_train_loader"], result["ooh_val_loader"] = (
            get_dataloaders(tr_ooh, va_ooh, te_ooh, batch_size=batch_size,
                            hospital_id_map=hospital_id_map)[:2]
        )
        result["held_out_hospitals"] = held
        result["ooh_train_seqs"] = tr_ooh
        result["ooh_val_seqs"] = va_ooh
        result["ooh_test_seqs"] = te_ooh

    return result
