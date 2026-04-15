import os
import json

from data_tabular import load_data, define_schema, build_tabular
from model_tabular import run_cv
from clinical_risk import risk_stratification_table, top_k_analysis


def main():

    df = load_data()

    patient_col, outcome_col, time_col, features = define_schema(df)

    X, y, groups = build_tabular(
        df,
        patient_col,
        outcome_col,
        time_col,
        features
    )

    results, oof = run_cv(X, y, groups)

    os.makedirs("outputs", exist_ok=True)

    # save core results
    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # choose best model (lightgbm)
    best_preds = oof["lgbm"]

    # 1. risk stratification table
    risk_table = risk_stratification_table(y, best_preds, n_bins=5)
    risk_table.to_csv("outputs/risk_stratification.csv", index=False)

    # 2. clinical operating point analysis
    top_decile = top_k_analysis(y, best_preds, k=0.1)

    with open("outputs/clinical_operating_point.json", "w") as f:
        json.dump(top_decile, f, indent=2)

import json

with open("outputs/y_true.json", "w") as f:
    json.dump(y.tolist(), f)

    print("\nSaved:")
    print("- outputs/results.json")
    print("- outputs/risk_stratification.csv")
    print("- outputs/clinical_operating_point.json")


if __name__ == "__main__":
    main()
