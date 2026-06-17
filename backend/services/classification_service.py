"""PyTorch MLP classification training & prediction."""
import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader

from models.registry import (
    get_model_paths, create_version_dir, register_version, generate_version_id,
)
from services._safe_serialize import save_scaler, load_scaler


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
          learning_rate, epochs, batch_size, device_str,
          dataset_name="", session_id="", user_id=None):
    """Train classification MLP. Returns metrics + losses + cm + version_id."""
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")

    cols = feature_cols + [target_col]
    df = df[cols].dropna()
    if len(df) < 10:
        return None, f"Insufficient clean data: {len(df)} rows after dropping NaN"
    y_raw = df[target_col].values
    unique_labels = np.unique(y_raw)
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    reverse_label_map = {i: lbl for lbl, i in label_map.items()}
    n_classes = len(unique_labels)
    y = np.array([label_map[lbl] for lbl in y_raw])

    X = df[feature_cols].values

    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
        )
    except ValueError as e:
        if "stratify" in str(e).lower() or "class" in str(e).lower():
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.2, random_state=42
            )
        else:
            raise

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
    report = classification_report(
        y_test,
        y_pred,
        labels=unique_test_labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    # Persist as version
    version_id = generate_version_id()
    vdir = create_version_dir("classification", version_id)

    model_path = os.path.join(vdir, "model.pth")
    scaler_path = os.path.join(vdir, "scaler.npz")
    config_path = os.path.join(vdir, "config.json")

    torch.save(model.state_dict(), model_path)
    save_scaler(scaler, scaler_path)

    config_dict = {
        "features": [str(c) for c in feature_cols],
        "target": str(target_col),
        "n_classes": n_classes,
        "hidden1": hidden1,
        "hidden2": hidden2,
        "dropout_rate": dropout_rate,
        "label_map": {str(k): v for k, v in label_map.items()},
        "reverse_label_map": {str(k): str(v) for k, v in reverse_label_map.items()}
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False)

    register_version("classification", version_id, {
        "dataset_name": dataset_name,
        "session_id": session_id,
        "user_id": user_id,
        "features": [str(c) for c in feature_cols],
        "target": str(target_col),
        "metrics": {"acc": acc},
        "params": {"hidden1": hidden1, "hidden2": hidden2, "dropout_rate": dropout_rate,
                   "learning_rate": learning_rate, "epochs": epochs, "batch_size": batch_size,
                   "n_classes": n_classes},
        "cm": cm,
        "label_names": label_names,
        "n_classes": n_classes,
        "reverse_label_map": {str(k): str(v) for k, v in reverse_label_map.items()},
        "classification_report": report,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, {"model": "model.pth", "scaler": "scaler.npz", "config": "config.json"})

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "acc": acc, "cm": cm, "label_names": label_names,
        "n_classes": n_classes,
        "reverse_label_map": {str(k): str(v) for k, v in reverse_label_map.items()},
        "classification_report": report,
        "train_losses": train_losses, "val_losses": val_losses,
        "version_id": version_id
    }, None


def _load_model(device, version_id=None, user_id=None):
    """Load model, scaler, config for the active (or specified) version."""
    paths, meta = get_model_paths("classification", version_id, user_id=user_id)
    if not paths:
        return None, None, None, "没有找到已保存的分类模型，请先训练模型或切换到有效版本。"

    with open(paths["config"], 'r') as f:
        config = json.load(f)
    scaler = load_scaler(paths["scaler"])

    n_classes = config["n_classes"]
    n_features = len(config["features"])
    h1 = config.get("hidden1", max(8, n_features))
    h2 = config.get("hidden2", max(4, n_features // 2))
    dr = config.get("dropout_rate", 0.2)
    model = ClassificationNet(n_features, h1, h2, dr, n_classes).to(device)
    model.load_state_dict(torch.load(paths["model"], map_location=device, weights_only=True))
    model.eval()

    return model, scaler, config, None


def predict_one(feature_values, device_str="cpu", version_id=None, user_id=None):
    """Single prediction. Returns {pred_class, prob, pred_idx}."""
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    model, scaler, config, err = _load_model(device, version_id, user_id=user_id)
    if err:
        return None, err

    n_classes = config["n_classes"]
    input_arr = np.array([feature_values])
    input_scaled = scaler.transform(input_arr)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        if n_classes == 2:
            positive_prob = float(torch.sigmoid(output).squeeze().item())
            pred_idx = 1 if positive_prob > 0.5 else 0
            prob = positive_prob if pred_idx == 1 else 1 - positive_prob
            all_probs = [1 - positive_prob, positive_prob]
        else:
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            prob = float(probs.max())
            pred_idx = int(np.argmax(probs))
            all_probs = probs.tolist()

    reverse_label_map = config.get("reverse_label_map", {})
    pred_class = reverse_label_map.get(str(pred_idx), pred_idx)
    label_names = [str(reverse_label_map.get(str(i), i)) for i in range(n_classes)]
    return {
        "pred_idx": pred_idx,
        "pred_class": str(pred_class),
        "prob": prob,
        "all_probs": all_probs,
        "label_names": label_names,
    }, None


MAX_BATCH_SIZE = 10000


def predict_batch(rows, device_str="cpu", version_id=None, user_id=None):
    """Batch prediction. Returns {pred_indices, confidences}."""
    if len(rows) > MAX_BATCH_SIZE:
        return None, f"单次预测最多支持 {MAX_BATCH_SIZE} 行，当前请求 {len(rows)} 行。请分批预测。"
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    model, scaler, config, err = _load_model(device, version_id, user_id=user_id)
    if err:
        return None, err

    n_classes = config["n_classes"]
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
