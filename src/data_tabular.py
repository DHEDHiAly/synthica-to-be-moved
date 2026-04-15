import numpy as np
import pandas as pd

DATA_PATH = "data/eicu_final_sequences_for_modeling.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Loaded:", df.shape)
    return df


def define_schema(df):
    patient_col = df.columns[0]

    outcome_col = [c for c in df.columns if "mort" in c.lower()][0]

    time_col = [
        c for c in df.columns
        if "time" in c.lower() or "hour" in c.lower()
    ][0]

    feature_cols = [
        c for c in df.columns
        if c not in {patient_col, outcome_col, time_col}
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    return patient_col, outcome_col, time_col, feature_cols


def build_tabular(df, patient_col, outcome_col, time_col, feature_cols):

    df = df.sort_values([patient_col, time_col])

    X, y, groups = [], [], []

    for pid, g in df.groupby(patient_col):

        if len(g) < 6:
            continue

        vals = g[feature_cols].values
        vals = np.nan_to_num(vals)

        mean = vals.mean(axis=0)
        max_ = vals.max(axis=0)
        min_ = vals.min(axis=0)
        last = vals[-1]
        std = vals.std(axis=0)

        # more stable trend
        trend = vals.mean(axis=0) - vals[0]

        last6 = vals[-6:].mean(axis=0)

        feat = np.concatenate([
            mean, max_, min_, last, std, trend, last6
        ])

        X.append(feat)

        # SAFE: patient-level label consistency assumption
        y.append(float(g[outcome_col].iloc[0]))

        groups.append(pid)

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    print("Tabular shape:", X.shape)

    return X, y, groups
