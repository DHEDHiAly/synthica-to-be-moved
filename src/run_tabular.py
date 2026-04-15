import os
import json

from data_tabular import load_data, define_schema, build_tabular
from model_tabular import run_cv


def main():

    df = load_data()

    patient_col, outcome_col, time_col, feature_cols = define_schema(df)

    X, y, groups = build_tabular(
        df,
        patient_col,
        outcome_col,
        time_col,
        feature_cols
    )

    results = run_cv(X, y, groups)

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved: outputs/experiment_results.json")


if __name__ == "__main__":
    main()
