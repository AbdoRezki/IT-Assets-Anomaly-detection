import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_model(X_train, epochs=100, lr=0.001):
    
    
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X_tensor = torch.FloatTensor(X_train)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, X_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), 'models/anomaly_model.pt')
    print("✅ Model saved to models/anomaly_model.pt")
    return model

def predict_anomaly(model_path, X_scaled, threshold=0.1):
    model = Autoencoder(input_dim=7)
    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        raise FileNotFoundError("Run training first: models/anomaly_model.pt missing")
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        reconstructed = model(X_tensor)
        mse = torch.mean((X_tensor - reconstructed)**2, dim=1)
        scores = mse.numpy()
        predictions = (scores > threshold).astype(int)
    return scores[0], bool(predictions[0])
