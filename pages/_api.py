"""Centralized API client for Flask backend."""
import requests
import streamlit as st

API_BASE = "http://localhost:5000/api"


def _post(path, json_data=None, files=None, timeout=300):
    try:
        if files:
            resp = requests.post(f"{API_BASE}{path}", files=files, timeout=timeout)
        else:
            resp = requests.post(f"{API_BASE}{path}", json=json_data, timeout=timeout)
        if resp.status_code != 200:
            err = resp.json().get("error", f"HTTP {resp.status_code}")
            st.error(f"API 错误: {err}")
            return None
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("无法连接到 Flask 后端 (http://localhost:5000)。请确保后端已启动。")
        return None
    except requests.exceptions.Timeout:
        st.error("请求超时，请重试。")
        return None


# ── Helpers ──

def _get(path, timeout=30):
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


# ── Data ──

def upload_data(file_bytes, filename):
    return _post("/data/upload", files={"file": (filename, file_bytes)})


def get_outliers(session_id):
    return _get(f"/data/{session_id}/outliers")


def process_data(session_id, operations):
    return _post(f"/data/{session_id}/process", json_data={"operations": operations})


def get_summary(session_id):
    return _get(f"/data/{session_id}/summary")


# ── Regression ──

def train_regression(session_id, target_col, feature_cols, lr, epochs, batch_size, device="cpu"):
    return _post("/regression/train", json_data={
        "session_id": session_id,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device
    })


def predict_regression(features, device="cpu"):
    return _post("/regression/predict", json_data={"features": features, "device": device})


def batch_predict_regression(rows, device="cpu"):
    return _post("/regression/batch_predict", json_data={"rows": rows, "device": device})


def clear_regression():
    return _post("/regression/clear")


# ── Classification ──

def train_classification(session_id, target_col, feature_cols, lr, epochs, batch_size, device="cpu"):
    return _post("/classification/train", json_data={
        "session_id": session_id,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device
    })


def predict_classification(features, device="cpu"):
    return _post("/classification/predict", json_data={"features": features, "device": device})


def batch_predict_classification(rows, device="cpu"):
    return _post("/classification/batch_predict", json_data={"rows": rows, "device": device})


def clear_classification():
    return _post("/classification/clear")


# ── DIY MLP ──

def train_diy_mlp(session_id, target_col, feature_cols, layers, task_type, n_classes,
                   lr, optimizer, epochs, batch_size, val_split, patience, device="cpu"):
    return _post("/diy_mlp/train", json_data={
        "session_id": session_id,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "layers": layers,
        "task_type": task_type,
        "n_classes": n_classes,
        "learning_rate": lr,
        "optimizer": optimizer,
        "epochs": epochs,
        "batch_size": batch_size,
        "val_split": val_split,
        "patience": patience,
        "device": device
    })


def predict_diy_mlp(features, device="cpu"):
    return _post("/diy_mlp/predict", json_data={"features": features, "device": device})


def batch_predict_diy_mlp(rows, device="cpu"):
    return _post("/diy_mlp/batch_predict", json_data={"rows": rows, "device": device})


def clear_diy_mlp():
    return _post("/diy_mlp/clear")


# ── Decision Tree ──

def train_decision_tree(session_id, target_col, feature_cols, task_type, criterion, max_depth):
    return _post("/decision_tree/train", json_data={
        "session_id": session_id,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "task_type": task_type,
        "criterion": criterion,
        "max_depth": max_depth
    })


def predict_decision_tree(input_dict, task_type):
    return _post("/decision_tree/predict", json_data={"input_dict": input_dict, "task_type": task_type})


def clear_decision_tree():
    return _post("/decision_tree/clear")


# ── Clustering ──

def train_clustering(session_id, feature_cols, algorithm, params):
    return _post("/clustering/train", json_data={
        "session_id": session_id,
        "feature_cols": feature_cols,
        "algorithm": algorithm,
        "params": params
    })


def elbow_clustering(session_id, feature_cols, max_k):
    return _post("/clustering/elbow", json_data={
        "session_id": session_id,
        "feature_cols": feature_cols,
        "max_k": max_k
    })


def predict_clustering(features):
    return _post("/clustering/predict", json_data={"features": features})


def clear_clustering():
    return _post("/clustering/clear")


# ── LLM ──

def chat_llm(api_base, api_key, model, messages):
    """Returns requests.Response for SSE streaming."""
    return requests.post(
        f"{API_BASE}/llm/chat",
        json={
            "api_base": api_base,
            "api_key": api_key,
            "model": model,
            "messages": messages
        },
        timeout=180,
        stream=True
    )
