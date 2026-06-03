"""Custom MLP training & prediction (regression + classification)."""
import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from torch.utils.data import TensorDataset, DataLoader

from models.registry import (
    get_model_paths, create_version_dir, register_version, generate_version_id,
)
from services._safe_serialize import save_scaler, load_scaler

ACT_FNS = {
    "ReLU": nn.ReLU, "LeakyReLU": lambda: nn.LeakyReLU(0.1), "GELU": nn.GELU,
    "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid, "ELU": nn.ELU, "SELU": nn.SELU,
    "无激活": nn.Identity
}


class DynamicMLP(nn.Module):
    def __init__(self, input_dim, layers_config, output_dim):
        super().__init__()
        seq = []
        in_dim = input_dim
        for cfg in layers_config:
            seq.append(nn.Linear(in_dim, cfg["neurons"]))
            if cfg.get("bn", False):
                seq.append(nn.BatchNorm1d(cfg["neurons"]))
            act_name = cfg.get("activation", "ReLU")
            seq.append(ACT_FNS.get(act_name, nn.ReLU)())
            if cfg.get("dropout", 0) > 0:
                seq.append(nn.Dropout(cfg["dropout"]))
            in_dim = cfg["neurons"]
        seq.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


def train(df, target_col, feature_cols, layers_config,
          task_type, n_classes, learning_rate, optimizer_name,
          epochs, batch_size, val_split, patience, device_str,
          dataset_name="", session_id=""):
    """Train custom MLP. Returns task-specific metrics + losses + version_id."""
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    is_cls = (task_type == "classification")

    cols = feature_cols + [target_col]
    df = df[cols].dropna()
    if len(df) < 10:
        return None, f"Insufficient clean data: {len(df)} rows after dropping NaN"
    X = df[feature_cols].values
    if is_cls:
        y_raw = df[target_col].values
        unique_labels = np.unique(y_raw)
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        reverse_label_map = {i: lbl for lbl, i in label_map.items()}
        y = np.array([label_map[lbl] for lbl in y_raw])
        stratify_arg = y
    else:
        y = df[target_col].values.reshape(-1, 1).astype(np.float32)
        stratify_arg = None
        label_map = None
        reverse_label_map = None

    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_arg
        )
        stratify_temp = y_temp if is_cls else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_split, random_state=42, stratify=stratify_temp
        )
    except ValueError as e:
        if stratify_arg is not None and ("stratify" in str(e).lower() or "class" in str(e).lower()):
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_split, random_state=42
            )
        else:
            raise

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)

    if is_cls:
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
    else:
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    actual_batch = min(batch_size, len(X_train))
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=actual_batch, shuffle=True)

    output_dim = 1 if not is_cls or n_classes == 2 else n_classes
    model = DynamicMLP(len(feature_cols), layers_config, output_dim).to(device)

    if not is_cls:
        criterion = nn.MSELoss()
    elif n_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    opt_map = {
        "Adam": lambda p: optim.Adam(p, lr=learning_rate, weight_decay=1e-4),
        "AdamW": lambda p: optim.AdamW(p, lr=learning_rate, weight_decay=1e-4),
        "SGD": lambda p: optim.SGD(p, lr=learning_rate, momentum=0.9, weight_decay=1e-4),
        "RMSprop": lambda p: optim.RMSprop(p, lr=learning_rate, weight_decay=1e-4),
    }
    optimizer = opt_map.get(optimizer_name, opt_map["Adam"])(model.parameters())

    best_loss = float('inf')
    early_stop_count = 0
    best_state = copy.deepcopy(model.state_dict())
    train_losses, val_losses = [], []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            if is_cls and n_classes == 2:
                pred = pred.squeeze(1)
                batch_y_f = batch_y.float()
                loss = criterion(pred, batch_y_f)
            else:
                loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            if is_cls and n_classes == 2:
                val_pred = val_pred.squeeze(1)
                val_loss = criterion(val_pred, y_val_t.float()).item()
            else:
                val_loss = criterion(val_pred, y_val_t).item()
        model.train()

        train_losses.append(epoch_loss / len(train_loader))
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

    result = {"train_losses": train_losses, "val_losses": val_losses}

    with torch.no_grad():
        y_pred_t = model(X_test_t)
        if not is_cls:
            y_pred_np = y_pred_t.cpu().numpy()
            result["r2"] = float(r2_score(y_test, y_pred_np))
            result["mae"] = float(mean_absolute_error(y_test, y_pred_np))
            result["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred_np)))
        else:
            if n_classes == 2:
                y_pred_np = (torch.sigmoid(y_pred_t).view(-1).cpu().numpy() > 0.5).astype(int)
            else:
                y_pred_np = torch.argmax(y_pred_t, dim=1).cpu().numpy()
            result["acc"] = float(accuracy_score(y_test, y_pred_np))

    # Persist as version
    version_id = generate_version_id()
    vdir = create_version_dir("diy_mlp", version_id)

    model_path = os.path.join(vdir, "model.pth")
    scaler_path = os.path.join(vdir, "scaler.npz")
    config_path = os.path.join(vdir, "config.json")

    config_dict = {
        "features": [str(c) for c in feature_cols],
        "target": str(target_col),
        "task": task_type,
        "layers": layers_config,
    }
    if is_cls:
        config_dict["n_classes"] = n_classes
        config_dict["label_map"] = {str(k): v for k, v in label_map.items()}
        config_dict["reverse_label_map"] = {str(k): str(v) for k, v in reverse_label_map.items()}

    torch.save(model.state_dict(), model_path)
    save_scaler(scaler, scaler_path)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False)

    metrics = {}
    params = {"task": task_type, "layers": layers_config, "learning_rate": learning_rate,
              "optimizer": optimizer_name, "epochs": epochs, "batch_size": batch_size}
    if is_cls:
        metrics["acc"] = result.get("acc")
        params["n_classes"] = n_classes
    else:
        metrics["r2"] = result.get("r2")
        metrics["mae"] = result.get("mae")
        metrics["rmse"] = result.get("rmse")

    register_version("diy_mlp", version_id, {
        "dataset_name": dataset_name,
        "session_id": session_id,
        "features": [str(c) for c in feature_cols],
        "target": str(target_col),
        "metrics": metrics,
        "params": params,
    }, {"model": "model.pth", "scaler": "scaler.npz", "config": "config.json"})

    if is_cls:
        result["n_classes"] = n_classes
        result["reverse_label_map"] = {str(k): str(v) for k, v in reverse_label_map.items()}

    if device.type == "cuda":
        torch.cuda.empty_cache()

    result["version_id"] = version_id
    return result, None


def _load_model(device, version_id=None):
    """Load model, scaler, config for the active (or specified) version."""
    paths, meta = get_model_paths("diy_mlp", version_id)
    if not paths:
        return None, None, None, "没有找到已保存的 DIY MLP 模型，请先训练模型或切换到有效版本。"

    with open(paths["config"], 'r') as f:
        config = json.load(f)
    scaler = load_scaler(paths["scaler"])

    output_dim = 1 if config["task"] == "regression" or config.get("n_classes") == 2 else config.get("n_classes", 1)
    model = DynamicMLP(len(config["features"]), config["layers"], output_dim).to(device)
    model.load_state_dict(torch.load(paths["model"], map_location=device, weights_only=True))
    model.eval()

    return model, scaler, config, None


def predict_one(feature_values, device_str="cpu", version_id=None):
    """Single prediction. Returns task-specific result."""
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    model, scaler, config, err = _load_model(device, version_id)
    if err:
        return None, err

    input_arr = np.array([feature_values])
    input_scaled = scaler.transform(input_arr)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        if config["task"] == "regression":
            return {"result": float(output.item())}, None
        else:
            n_cls = config.get("n_classes", 2)
            reverse_label_map = config.get("reverse_label_map", {})
            if n_cls == 2:
                positive_prob = float(torch.sigmoid(output).squeeze().item())
                pred_idx = 1 if positive_prob > 0.5 else 0
                prob = positive_prob if pred_idx == 1 else 1 - positive_prob
                pred_class = reverse_label_map.get(str(pred_idx), pred_idx)
                label_names = [str(reverse_label_map.get(str(i), i)) for i in range(n_cls)]
                return {"pred_idx": pred_idx, "pred_class": str(pred_class), "prob": prob,
                        "all_probs": [1 - positive_prob, positive_prob],
                        "label_names": label_names}, None
            else:
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                pred_class = reverse_label_map.get(str(pred_idx), pred_idx)
                label_names = [str(reverse_label_map.get(str(i), i)) for i in range(n_cls)]
                return {"pred_idx": pred_idx, "pred_class": str(pred_class), "prob": float(probs.max()),
                        "all_probs": probs.tolist(), "label_names": label_names}, None


MAX_BATCH_SIZE = 10000


def predict_batch(rows, device_str="cpu", version_id=None):
    """Batch prediction."""
    if len(rows) > MAX_BATCH_SIZE:
        return None, f"单次预测最多支持 {MAX_BATCH_SIZE} 行，当前请求 {len(rows)} 行。请分批预测。"
    device = torch.device("cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    model, scaler, config, err = _load_model(device, version_id)
    if err:
        return None, err

    batch_arr = np.array(rows)
    batch_scaled = scaler.transform(batch_arr)
    batch_tensor = torch.tensor(batch_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(batch_tensor)
        if config["task"] == "regression":
            return {"predictions": output.cpu().numpy().flatten().tolist()}, None
        else:
            n_cls = config.get("n_classes", 2)
            if n_cls == 2:
                probs = torch.sigmoid(output).cpu().numpy().flatten()
                pred_indices = (probs > 0.5).astype(int).tolist()
                confidences = np.where(np.array(pred_indices) == 1, probs, 1 - probs).tolist()
            else:
                probs_all = torch.softmax(output, dim=1).cpu().numpy()
                pred_indices = np.argmax(probs_all, axis=1).tolist()
                confidences = probs_all[np.arange(len(pred_indices)), pred_indices].tolist()
            return {"pred_indices": pred_indices, "confidences": confidences}, None
