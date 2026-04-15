import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve


def main():

    with open("outputs/oof_predictions.json", "r") as f:
        preds = json.load(f)

    with open("outputs/y_true.json", "r") as f:
        y_true = np.array(json.load(f))

    os.makedirs("outputs/figures", exist_ok=True)

    best_model = "lgbm"
    y_pred = np.array(preds[best_model])

    # -------------------------
    # 1. Risk stratification
    # -------------------------
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })

    df["risk_bin"] = pd.qcut(df["y_pred"], q=10, labels=False)

    risk_table = df.groupby("risk_bin").agg(
        n=("y_true", "count"),
        mortality_rate=("y_true", "mean"),
        avg_risk=("y_pred", "mean")
    ).reset_index()

    risk_table.to_csv("outputs/risk_stratification.csv", index=False)

    plt.figure()
    plt.bar(risk_table["risk_bin"], risk_table["mortality_rate"])
    plt.xlabel("Risk Decile (0=lowest risk, 9=highest risk)")
    plt.ylabel("Observed Mortality Rate")
    plt.title("Risk Stratification by Model Score")
    plt.savefig("outputs/figures/risk_stratification.png", dpi=300)
    plt.close()

    # -------------------------
    # 2. Calibration curve
    # -------------------------
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_pred,
        n_bins=10,
        strategy="quantile"
    )

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="LGBM")
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Probability")
    plt.title("Calibration Curve")
    plt.legend()
    plt.savefig("outputs/figures/calibration_curve.png", dpi=300)
    plt.close()

    print("Saved clinical figures + risk table")


if __name__ == "__main__":
    main()
