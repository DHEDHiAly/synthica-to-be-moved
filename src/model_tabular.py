import numpy as np
import lightgbm as lgb

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from eval_utils import summarize


def get_models():
    return {
        "logreg": LogisticRegression(max_iter=3000),
        "rf": RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            class_weight="balanced"
        ),
        "lgbm": lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42
        )
    }


def remove_bad_features(X):
    var = X.var(axis=0)
    return X[:, var > 1e-6]


def run_cv(X, y, groups, n_splits=5):

    X = remove_bad_features(X)

    gkf = GroupKFold(n_splits=n_splits)
    models = get_models()

    oof = {name: np.zeros(len(y)) for name in models}

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X[tr], X[te]
        y_tr = y[tr]

        for name, model in models.items():
            model.fit(X_tr, y_tr)
            preds = model.predict_proba(X_te)[:, 1]
            oof[name][te] = preds

        print(f"Fold {fold+1} done")

    results = {}

    print("\n===== FINAL RESULTS =====")

    for name, preds in oof.items():
        res = summarize(y, preds)
        results[name] = res
        print(name)
        print(res)

    y_true_full = y.copy()

    return results, oof, y_true_full
