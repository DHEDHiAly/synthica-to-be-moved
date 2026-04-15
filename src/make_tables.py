import json
import pandas as pd


def main():

    with open("outputs/results.json", "r") as f:
        results = json.load(f)

    rows = []

    for model, vals in results.items():
        rows.append({
            "Model": model,
            "AUROC": vals["auroc"],
            "AUROC_Low": vals["auroc_ci"][0],
            "AUROC_High": vals["auroc_ci"][1],
            "AUPRC": vals["auprc"],
            "AUPRC_Low": vals["auprc_ci"][0],
            "AUPRC_High": vals["auprc_ci"][1],
            "Brier": vals["brier"]
        })

    df = pd.DataFrame(rows)

    df.to_csv("outputs/results_table.csv", index=False)

    print(df)


if __name__ == "__main__":
    main()
