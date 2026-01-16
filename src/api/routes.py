from flask import Blueprint, request, jsonify
from src.data_prep import preprocess_single
from src.model import Autoencoder, predict_anomaly
import torch

api = Blueprint('api', __name__)

model = Autoencoder(input_dim=7)  # Update with len(FEATURES)
model.load_state_dict(torch.load('models/anomaly_model.pt'))
model.eval()

@api.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@api.route('/score', methods=['POST'])
def score():
    data = request.json
    asset_id = data.get('asset_id')
    features = {k: data[k] for k in ['cpu_mean', 'ram_mean', 'disk_usage', 
                                     'logins', 'failed_logins', 'tickets_opened', 'network_mb']}
    
    X_scaled = preprocess_single(features)
    score, is_anomaly = predict_anomaly(model, X_scaled)
    
    return jsonify({
        'asset_id': asset_id,
        'anomaly_score': float(score),
        'is_anomaly': is_anomaly
    })
