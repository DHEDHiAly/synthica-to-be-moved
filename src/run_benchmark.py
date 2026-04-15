#!/usr/bin/env python3
import json, os
import numpy as np
from data import load_and_engineer, build_patient_features, train_test_split
from models import xgb_model, lgb_model

def main():
    print("🎯 PRODUCTION ICU MORTALITY BENCHMARK")
    np.random.seed(42)
    
    # ENGINEERED PIPELINE
    df, features = load_and_engineer()
    patients = build_patient_features(df, features)
    train_p, test_p = train_test_split(patients)
    
    # BENCHMARK SUITE
    print("\n🏆 PRODUCTION RESULTS:")
    results = {
        'XGBoost': xgb_model(train_p, test_p),
        'LightGBM': lgb_model(train_p, test_p)
    }
    
    print(json.dumps(results, indent=2))
    
    best_auc = max(r['auc'] for r in results.values())
    print(f"\n🎖️ BEST AUROC: {best_auc:.3f}")
    print("✅ PUBLISHABLE" if best_auc >= 0.70 else f"📈 Progress: {best_auc:.3f}")
    
    # SAVE
    os.makedirs("output", exist_ok=True)
    json.dump({
        'features': features,
        'results': results,
        'best_auc': best_auc
    }, open("output/benchmark_results.json", "w"), indent=2)

if __name__ == "__main__":
    main()
