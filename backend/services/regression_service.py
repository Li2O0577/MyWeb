"""PyTorch MLP regression training & prediction."""
import os
import copy
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "reg_best_model.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "reg_scaler.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "reg_config.json")


class RegressionNet(nn.Module):
    def __init__(self, input_dim, h1, h2, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(h2, 1)
        )

    def forward(self, x):
        return self.net(x)


def train(df, target_col, feature_cols, hidden1, hidden2, dropout_rate,
          learning_rate, epochs, batch_size, device_str):
    """Train regression MLP. Returns {r2, mae, rmse, train_losses, val_losses}."""
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")

    cols = feature_cols + [target_col]
    df = df[cols].dropna()
    if len(df) < 10:
        return {"error": f"Insufficient clean data: {len(df)} rows after dropping NaN"}
    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)

    actual_batch = min(batch_size, len(X_train))
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=actual_batch, shuffle=True)

    model = RegressionNet(len(feature_cols), hidden1, hidden2, dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_loss = float('inf')
    early_stop_count = 0
    patience = 10
    best_state = copy.deepcopy(model.state_dict())
    train_losses, val_losses = [], []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        model.train()

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_count = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        y_pred = model(X_test_t).cpu().numpy()

    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # Persist model
    torch.save(model.state_dict(), MODEL_PATH)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            "features": feature_cols,
            "target": target_col,
            "hidden1": hidden1,
            "hidden2": hidden2,
            "dropout_rate": dropout_rate,
            "r2": r2, "mae": mae, "rmse": rmse
        }, f, ensure_ascii=False)

    return {"r2": r2, "mae": mae, "rmse": rmse,
            "train_losses": train_losses, "val_losses": val_losses}


def predict_one(feature_values, device_str="cpu"):
    """Single prediction. Returns predicted value."""
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    if not os.path.exists(MODEL_PATH):
        return None, "No saved model found."

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    n_features = len(config["features"])
    h1 = config.get("hidden1", max(8, n_features))
    h2 = config.get("hidden2", max(4, n_features // 2))
    dr = config.get("dropout_rate", 0.2)
    model = RegressionNet(n_features, h1, h2, dr).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    input_arr = np.array([feature_values])
    input_scaled = scaler.transform(input_arr)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(input_tensor).item()
    return {"result": float(pred)}, None


def predict_batch(rows, device_str="cpu"):
    """Batch prediction. Returns list of predictions."""
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    if not os.path.exists(MODEL_PATH):
        return None, "No saved model found."

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    n_features = len(config["features"])
    h1 = config.get("hidden1", max(8, n_features))
    h2 = config.get("hidden2", max(4, n_features // 2))
    dr = config.get("dropout_rate", 0.2)
    model = RegressionNet(n_features, h1, h2, dr).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    batch_arr = np.array(rows)
    batch_scaled = scaler.transform(batch_arr)
    batch_tensor = torch.tensor(batch_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(batch_tensor).cpu().numpy().flatten().tolist()
    return {"predictions": preds}, None
