# app.py - Project root
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from flask import Flask
from flask_cors import CORS
from src.api.routes import api  # Now works

app = Flask(__name__)
CORS(app)
app.register_blueprint(api, url_prefix='/api')

@app.route('/')
def home():
    return "IT Asset Anomaly Monitor API - /api/score for predictions"

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
