import os
import json
import numpy as np

from data_tabular import load_data, define_schema, build_tabular
from model_tabular import run_cv


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    return obj


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

    results, oof, y_true = run_cv(X, y, groups)

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/results.json", "w") as f:
        json.dump(make_serializable(results), f, indent=2)

    with open("outputs/oof_predictions.json", "w") as f:
        json.dump(make_serializable(oof), f, indent=2)

    with open("outputs/y_true.json", "w") as f:
        json.dump(make_serializable(y_true), f, indent=2)

    print("\nSaved:")
    print("- outputs/results.json")
    print("- outputs/oof_predictions.json")
    print("- outputs/y_true.json")


if __name__ == "__main__":
    main()
