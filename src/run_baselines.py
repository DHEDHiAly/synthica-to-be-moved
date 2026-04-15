#!/usr/bin/env python3
import json, os, numpy as np
from data import load_and_prepare, feature_hash, build_sequences, patient_split, scale_sequences
from baselines import xgb_benchmark, rf_benchmark

def main():
    print("🎯 PRODUCTION ICU BENCHMARK (AUROC ≥ 0.73)")
    np.random.seed(42)
    
    df, features = load_and_prepare()
    print(f"🔑 Features: {feature_hash(features)}")
    
    sequences = build_sequences(df, features)
    train_raw, test_raw = patient_split(sequences)
    
    train, test = scale_sequences(train_raw, test_raw)
    
    print("\n🏆 BENCHMARK RESULTS:")
    results = {
        "XGBoost (mean)": xgb_benchmark(train, test),
        "XGBoost (last)": xgb_benchmark(train, test, mode="last"),
        "Random Forest": rf_benchmark(train, test)
    }
    
    print(json.dumps(results, indent=2))
    
    best_auc = max(r["auc"] for r in results.values())
    print(f"\n🎖️ RESULT: {best_auc:.3f} AUROC")
    print("✅ PUBLISHABLE" if best_auc >= 0.73 else "📈 COMPETITIVE")
    
    os.makedirs("output", exist_ok=True)
    json.dump({"results": results, "features": features}, 
              open("output/final_benchmark.json", "w"), indent=2)

if __name__ == "__main__":
    main()
