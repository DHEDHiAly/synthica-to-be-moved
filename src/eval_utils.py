import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)

        if len(np.unique(y_true[idx])) < 2:
            continue

        scores.append(metric_fn(y_true[idx], y_pred[idx]))

    return {
        "mean": float(np.mean(scores)),
        "ci_low": float(np.percentile(scores, 2.5)),
        "ci_high": float(np.percentile(scores, 97.5))
    }


def compute_metrics(y_true, y_pred):
    return {
        "auroc": float(roc_auc_score(y_true, y_pred)),
        "auprc": float(average_precision_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_pred))
    }


def summarize(y_true, y_pred):
    base = compute_metrics(y_true, y_pred)

    auroc_ci = bootstrap_ci(y_true, y_pred, roc_auc_score)
    auprc_ci = bootstrap_ci(y_true, y_pred, average_precision_score)

    return {
        "auroc": base["auroc"],
        "auroc_ci": [auroc_ci["ci_low"], auroc_ci["ci_high"]],
        "auprc": base["auprc"],
        "auprc_ci": [auprc_ci["ci_low"], auprc_ci["ci_high"]],
        "brier": base["brier"]
    }
