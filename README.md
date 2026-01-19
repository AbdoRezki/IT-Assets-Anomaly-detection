# IT Asset Anomaly Monitor

This project is an end‑to‑end **machine learning system** that detects anomalous behavior of IT assets (laptops, servers, VMs, etc.) based on operational metrics. It learns what “normal” usage looks like across multiple features and flags unusual patterns that may indicate incidents, misconfigurations, or security issues.

---

## Project purpose

- Model normal behavior of IT assets using historical metrics such as CPU, RAM, disk usage, login activity, ticket volume, and network traffic.  
- Compute an anomaly score for each asset‑day and classify it as normal or anomalous using a trained model.  
- Expose the anomaly detection logic through a REST API so external tools (e.g., Power Automate, SharePoint, Teams) can automatically create alerts and incidents.

---

## Project structure

```text
.
├── app.py                      # Flask application entrypoint
├── data/
│   └── asset_metrics.csv       # Sample/synthetic IT asset metrics dataset
├── models/
│   ├── anomaly_model.pt        # Trained PyTorch autoencoder weights
│   └── scaler.joblib           # Fitted feature scaler (StandardScaler)
├── notebooks/
│   └── 01_eda_and_baseline.ipynb  # Data generation, EDA, and training pipeline
├── src/
│   ├── __init__.py
│   ├── data_prep.py            # Data loading, feature selection, scaling helpers
│   ├── model.py                # Autoencoder architecture and training / scoring logic
│   └── api/
│       ├── __init__.py
│       └── routes.py           # Flask blueprint: /api/health and /api/score endpoints
├── tests/                      # (Optional) tests for API and utilities
├── venv/                       # Local virtual environment (not committed)
├── eda_normal_vs_anomalous.png # EDA plot comparing normal vs incident distributions
└── requirements.txt            # Python dependencies
