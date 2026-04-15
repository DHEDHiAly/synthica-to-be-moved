import numpy as np
import pandas as pd


def risk_stratification_table(y_true, y_pred, n_bins=5):

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })

    df = df.sort_values("y_pred")

    # quantile-based bins
    df["risk_bin"] = pd.qcut(df["y_pred"], q=n_bins, labels=False)

    table = df.groupby("risk_bin").agg(
        n=("y_true", "count"),
        mortality_rate=("y_true", "mean"),
        avg_risk=("y_pred", "mean")
    ).reset_index()

    return table


def top_k_analysis(y_true, y_pred, k=0.1):
    """
    Clinical-style: top risk decile performance
    """

    threshold = np.quantile(y_pred, 1 - k)
    preds = (y_pred >= threshold).astype(int)

    tp = ((preds == 1) & (y_true == 1)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    tn = ((preds == 0) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)

    return {
        "top_k": k,
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }
