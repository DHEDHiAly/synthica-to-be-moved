import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def features_from_sequences(sequences: list, mode="mean") -> tuple:
    """Sequence → static features."""
    X, y = [], []
    for s in sequences:
        x = s["x"].numpy()
        if mode == "mean":
            feat = np.nanmean(x, axis=0)
        else:
            feat = x[-1]
        X.append(feat)
        y.append(int(s["y"]))
    return np.array(X), np.array(y)

def xgb_benchmark(train: list, test: list, mode="mean") -> dict:
    """Production XGBoost with imbalance handling."""
    X_tr, y_tr = features_from_sequences(train, mode)
    X_te, y_te = features_from_sequences(test, mode)
    
    # Imbalance fix
    scale_pos = (len(y_tr) - sum(y_tr)) / (sum(y_tr) + 1e-8)
    
    model = XGBClassifier(
        n_estimators=2000,
        max_depth=10,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.01,
        scale_pos_weight=scale_pos,  # CRITICAL FIX
        random_state=42,
        verbosity=0
    )
    model.fit(X_tr, y_tr)
    
    probs = model.predict_proba(X_te)[:, 1]
    return {
        "auc": float(roc_auc_score(y_te, probs)),
        "ap": float(average_precision_score(y_te, probs))
    }

def rf_benchmark(train: list, test: list) -> dict:
    X_tr, y_tr = features_from_sequences(train)
    X_te, y_te = features_from_sequences(test)
    
    model = RandomForestClassifier(
        n_estimators=1000, max_depth=12,
        class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    
    probs = model.predict_proba(X_te)[:, 1]
    return {
        "auc": float(roc_auc_score(y_te, probs)),
        "ap": float(average_precision_score(y_te, probs))
    }
