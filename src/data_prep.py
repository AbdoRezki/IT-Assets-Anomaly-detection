import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURES = ['cpu_mean', 'ram_mean', 'disk_usage', 'logins', 
            'failed_logins', 'tickets_opened', 'network_mb']

def get_project_root():
    """Find project root from anywhere (notebooks/ or root)"""
    cwd = os.getcwd()
    root = cwd if os.path.exists(os.path.join(cwd, 'app.py')) else os.path.dirname(cwd)
    return root

def ensure_dirs(root=None):
    root = root or get_project_root()
    os.makedirs(os.path.join(root, 'models'), exist_ok=True)
    os.makedirs(os.path.join(root, 'data'), exist_ok=True)
    print(f"✅ Dirs ready in: {root}")

def load_and_preprocess(file_path):
    root = get_project_root()
    ensure_dirs(root)
    
    df = pd.read_csv(file_path)
    print(f"Loaded {df.shape} rows")

    df['date'] = pd.to_datetime(df['date'])
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])

    # ABSOLUTE PATH save
    scaler_path = os.path.join(root, 'models', 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved: {scaler_path}")

    return X, scaler, df

def preprocess_single(features_dict):
    root = get_project_root()
    scaler_path = os.path.join(root, 'models', 'scaler.joblib')
    scaler = joblib.load(scaler_path)
    
    feature_order = FEATURES
    X = np.array([features_dict[f] for f in feature_order])
    X_scaled = scaler.transform(X.reshape(1, -1))
    return X_scaled
