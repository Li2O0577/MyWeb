"""PyTorch MLP classification training & prediction."""
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
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "cls_best_model.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "cls_scaler.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "cls_config.json")


class ClassificationNet(nn.Module):
    def __init__(self, input_dim, h1, h2, dropout_rate, num_classes):
        super().__init__()
        output_dim = 1 if num_classes == 2 else num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(h2, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def train(df, target_col, feature_cols, hidden1, hidden2, dropout_rate,
          learning_rate, epochs, batch_size, device_str):
    """Train classification MLP. Returns metrics + losses + cm."""
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")

    y_raw = df[target_col].values
    unique_labels = np.unique(y_raw)
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    reverse_label_map = {i: lbl for lbl, i in label_map.items()}
    n_classes = len(unique_labels)
    y = np.array([label_map[lbl] for lbl in y_raw])

    X = df[feature_cols].values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)

    actual_batch = min(batch_size, len(X_train))
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=actual_batch, shuffle=True)

    model = ClassificationNet(len(feature_cols), hidden1, hidden2, dropout_rate, n_classes).to(device)
    criterion = nn.CrossEntropyLoss() if n_classes > 2 else nn.BCEWithLogitsLoss()
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
            pred = model(batch_x)
            if n_classes == 2:
                pred = pred.squeeze(1)
                batch_y = batch_y.float()
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            if n_classes == 2:
                val_pred = val_pred.squeeze(1)
                val_loss = criterion(val_pred, y_val_t.float()).item()
            else:
                val_loss = criterion(val_pred, y_val_t).item()
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
        y_pred_t = model(X_test_t)
        if n_classes == 2:
            y_pred = (torch.sigmoid(y_pred_t).view(-1) > 0.5).cpu().numpy()
        else:
            y_pred = torch.argmax(y_pred_t, dim=1).cpu().numpy()

    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()
    unique_test_labels = sorted(set(int(y) for y in y_test) | set(int(y) for y in y_pred))
    label_names = [str(reverse_label_map.get(l, l)) for l in unique_test_labels]

    # Persist
    torch.save(model.state_dict(), MODEL_PATH)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            "features": [str(c) for c in feature_cols],
            "target": str(target_col),
            "n_classes": n_classes,
            "hidden1": hidden1,
            "hidden2": hidden2,
            "dropout_rate": dropout_rate,
            "label_map": {str(k): v for k, v in label_map.items()},
            "reverse_label_map": {str(k): str(v) for k, v in reverse_label_map.items()}
        }, f, ensure_ascii=False)

    return {
        "acc": acc, "cm": cm, "label_names": label_names,
        "n_classes": n_classes,
        "reverse_label_map": {str(k): str(v) for k, v in reverse_label_map.items()},
        "train_losses": train_losses, "val_losses": val_losses
    }


def predict_one(feature_values, device_str="cpu"):
    """Single prediction. Returns {pred_class, prob, pred_idx}."""
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    if not os.path.exists(MODEL_PATH):
        return None, "No saved model found."

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    n_classes = config["n_classes"]
    n_features = len(config["features"])
    h1 = config.get("hidden1", max(8, n_features))
    h2 = config.get("hidden2", max(4, n_features // 2))
    dr = config.get("dropout_rate", 0.2)
    model = ClassificationNet(n_features, h1, h2, dr, n_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    input_arr = np.array([feature_values])
    input_scaled = scaler.transform(input_arr)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        if n_classes == 2:
            prob = float(torch.sigmoid(output).squeeze().item())
            pred_idx = 1 if prob > 0.5 else 0
        else:
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            prob = float(probs.max())
            pred_idx = int(np.argmax(probs))

    return {"pred_idx": pred_idx, "prob": prob}, None


def predict_batch(rows, device_str="cpu"):
    """Batch prediction. Returns {pred_indices, confidences}."""
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    if not os.path.exists(MODEL_PATH):
        return None, "No saved model found."

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    n_classes = config["n_classes"]
    n_features = len(config["features"])
    h1 = config.get("hidden1", max(8, n_features))
    h2 = config.get("hidden2", max(4, n_features // 2))
    dr = config.get("dropout_rate", 0.2)
    model = ClassificationNet(n_features, h1, h2, dr, n_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    batch_arr = np.array(rows)
    batch_scaled = scaler.transform(batch_arr)
    batch_tensor = torch.tensor(batch_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(batch_tensor)
        if n_classes == 2:
            probs = torch.sigmoid(output).cpu().numpy().flatten()
            pred_indices = (probs > 0.5).astype(int).tolist()
            confidences = np.where(np.array(pred_indices) == 1, probs, 1 - probs).tolist()
        else:
            probs_all = torch.softmax(output, dim=1).cpu().numpy()
            pred_indices = np.argmax(probs_all, axis=1).tolist()
            confidences = probs_all[np.arange(len(pred_indices)), pred_indices].tolist()

    return {"pred_indices": pred_indices, "confidences": confidences}, None
