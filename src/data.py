import numpy as np
import pandas as pd
import torch

DATA_PATH = "data/eicu_final_sequences_for_modeling.csv"

def load_and_engineer():
    """Load + MAX feature engineering."""
    df = pd.read_csv(DATA_PATH)
    
    # ALL informative features (not just 4-6)
    exclude = {'patientunitstayid', 'time_bin', 'hospitalid', 'mortality', 'los_days'}
    features = [col for col in df.columns 
                if col not in exclude and df[col].var() > 0.001][:20]
    
    # TEMPORAL FEATURES (CRITICAL)
    df['time_since_start'] = df.groupby('patientunitstayid')['time_bin'].transform(lambda x: x - x.min())
    df['time_norm'] = df['time_since_start'] / 24.0  # Normalize to days
    features += ['time_norm']
    
    print(f"✅ {len(features)} features + time encoding")
    print(f"Mortality: {df['mortality'].mean():.1%}")
    return df, features

def build_patient_features(df, features):
    """Convert sequences → learnable static features."""
    patient_features = []
    
    for pid, group in df.groupby('patientunitstayid'):
        if len(group) < 6: continue
            
        group = group.sort_values('time_bin')
        X = group[features].fillna(0).values
        
        # 12 ENGINEERED FEATURES PER VARIABLE (proven ICU approach)
        stats = []
        for feat_idx in range(X.shape[1]):
            col = X[:, feat_idx]
            stats.extend([
                col.mean(), col.std(), col.min(), col.max(),
                col[-1] - col[0],  # Trend
                (col[-1] > col[0]).astype(float),  # Direction
                np.sum(col > col.mean()),  # Above average count
                np.sum(col == 0) / len(col),  # Sparsity
            ])
        
        y = float(group['mortality'].iloc[-1])
        
        patient_features.append({
            'X': np.array(stats, dtype=np.float32),
            'y': y,
            'pid': pid
        })
    
    print(f"✅ Built {len(patient_features)} engineered patients")
    return patient_features

def train_test_split(features, test_size=0.2, seed=42):
    np.random.seed(seed)
    pids = [f['pid'] for f in features]
    np.random.shuffle(pids)
    
    split = int(len(pids) * (1 - test_size))
    train_pids = set(pids[:split])
    
    train = [f for f in features if f['pid'] in train_pids]
    test = [f for f in features if f['pid'] not in train_pids]
    
    print(f"✅ Train: {len(train)} | Test: {len(test)}")
    return train, test
