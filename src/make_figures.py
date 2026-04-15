import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, precision_recall_curve


def main():

    with open("outputs/oof_predictions.json", "r") as f:
        preds = json.load(f)

    with open("outputs/y_true.json", "r") as f:
        y_true = np.array(json.load(f))

    preds_dict = {k: np.array(v) for k, v in preds.items()}

    os.makedirs("outputs/figures", exist_ok=True)

    # ROC curve
    plt.figure()
    for name, p in preds_dict.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        plt.plot(fpr, tpr, label=name)

    plt.plot([0, 1], [0, 1], "--")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("outputs/figures/roc_curve.png", dpi=300)
    plt.close()

    # PR curve
    plt.figure()
    for name, p in preds_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, p)
        plt.plot(recall, precision, label=name)

    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig("outputs/figures/pr_curve.png", dpi=300)
    plt.close()

    print("Saved figures to outputs/figures/")


if __name__ == "__main__":
    main()
