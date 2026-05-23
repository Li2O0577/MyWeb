"""Centralized API client for Flask backend.

All network calls use a short connect timeout (2s) to avoid hanging
when the Flask backend is not running. Read timeout varies by endpoint.
"""
import requests
import streamlit as st
import os

API_BASE = os.environ.get("INDETERMINATE_API_BASE", "http://127.0.0.1:5001/api").rstrip("/")
CONNECT_TIMEOUT = 2  # fail fast if backend is down


def _format_error_payload(payload, fallback):
    """Return a readable error string from new or legacy backend payloads."""
    if not isinstance(payload, dict):
        return fallback

    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code") or "ERROR"
        message = error.get("message") or fallback
        detail = error.get("detail") or ""
        if detail and detail != message:
            return f"{code}: {message} ({detail})"
        return f"{code}: {message}"
    if isinstance(error, str):
        return error
    return fallback


def _backend_ok():
    """Quick check: is Flask reachable? Returns (bool, health_data). Max ~2s."""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=(CONNECT_TIMEOUT, 2))
        if resp.status_code == 200:
            return True, resp.json()
    except Exception:
        pass
    return False, None


def is_backend_connected():
    """Return True when Flask health check passes."""
    ok, _ = _backend_ok()
    return ok


def backend_status_badge():
    """Render a prominent status badge showing Flask backend connectivity.
    Call this at the top of every ML page."""
    ok, info = _backend_ok()
    if ok:
        sessions = info.get("active_sessions", 0)
        recent = info.get("recent_sessions", [])
        models = info.get("saved_models", {})
        model_list = [k for k, v in models.items() if v]
        model_str = ", ".join(model_list) if model_list else "无"
        recent_text = ""
        if recent:
            first = recent[0]
            recent_text = f" | 最近数据: {first.get('source_name', 'unknown')} ({first.get('rows', 0)} 行)"
        st.success(f"✅ Flask 后端已连接 | 活跃会话: {sessions}{recent_text} | 已保存模型: {model_str}")
        return True
    else:
        st.error("❌ Flask 后端未运行！请在新终端中运行 `cd backend && python app.py` 启动后端，否则 ML 训练/预测功能不可用。")
        st.caption("数据加载和可视化功能不受影响。")
        return False


def session_is_valid():
    """Check whether the current Streamlit session_id still exists in Flask."""
    sid = st.session_state.get("session_id")
    if not sid:
        return False
    resp = _get(f"/data/{sid}/summary")
    if resp and "summary" in resp:
        if resp.get("session_meta"):
            st.session_state.session_meta = resp["session_meta"]
        return True
    st.session_state.pop("session_id", None)
    return False


def render_backend_sync_panel(df, compact=False):
    """Show backend data sync status and a manual resync button.

    Use after local data is loaded. This keeps ML pages recoverable when Flask
    restarts or the previous session expires.
    """
    if df is None or len(df) == 0:
        return False

    ok, info = _backend_ok()
    if not ok:
        st.warning("⚠️ 数据已在前端加载，但 Flask 后端未连接。启动后端后点击同步即可训练。")
        return False

    sid = st.session_state.get("session_id")
    if sid and session_is_valid():
        meta = st.session_state.get("session_meta", {})
        source = meta.get("source_name", "当前数据")
        rows = meta.get("rows", len(df))
        if compact:
            st.caption(f"后端数据已同步 · {source} · {rows} 行 · session `{sid}`")
        else:
            sessions = info.get("active_sessions", 0) if info else 0
            st.success(f"✅ 后端数据已同步 | {source} · {rows} 行 | session: `{sid}` | 活跃会话: {sessions}")
        return True

    st.warning("⚠️ 后端没有当前数据 session，可能是后端重启或 session 已过期。")
    if st.button("重新同步当前数据到后端", use_container_width=True, type="primary"):
        with st.spinner("正在同步当前数据到 Flask 后端..."):
            new_sid = ensure_session(df)
        if new_sid:
            st.success(f"同步完成，session: `{new_sid}`")
            st.rerun()
        else:
            st.error("同步失败。请确认 Flask 后端正在运行。")
    return False


def _post(path, json_data=None, files=None, timeout=300):
    """POST with fast-connect timeout. Returns parsed JSON or None on failure."""
    try:
        if files:
            resp = requests.post(
                f"{API_BASE}{path}", files=files,
                timeout=(CONNECT_TIMEOUT, timeout)
            )
        else:
            resp = requests.post(
                f"{API_BASE}{path}", json=json_data,
                timeout=(CONNECT_TIMEOUT, timeout)
            )
        if resp.status_code != 200:
            try:
                err = _format_error_payload(resp.json(), f"HTTP {resp.status_code}")
            except Exception:
                err = f"HTTP {resp.status_code} (非 JSON 响应)"
            st.error(f"API 错误 [{path}]: {err}")
            return None
        try:
            return resp.json()
        except Exception:
            st.error(f"后端返回了无效的 JSON 响应 (HTTP {resp.status_code})")
            return None
    except requests.exceptions.ConnectionError:
        ok, info = _backend_ok()
        if ok:
            st.error(f"Flask 后端连接异常（健康检查通过但 {path} 被拒绝）。请重启后端。")
        else:
            st.error("无法连接到 Flask 后端。请在新终端中运行 `cd backend && python app.py` 启动后端。")
        return None
    except requests.exceptions.Timeout:
        st.error(f"请求超时 ({path})。请检查后端是否正常运行。")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"网络错误 ({path}): {e}")
        return None


def _get(path, timeout=30):
    """GET with fast-connect timeout. Returns parsed JSON or None on failure."""
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=(CONNECT_TIMEOUT, timeout))
        if resp.status_code != 200:
            return None
        try:
            return resp.json()
        except Exception:
            return None
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
            requests.exceptions.RequestException):
        return None


def _delete(path, timeout=30):
    """DELETE with fast-connect timeout. Returns parsed JSON or None on failure."""
    try:
        resp = requests.delete(f"{API_BASE}{path}", timeout=(CONNECT_TIMEOUT, timeout))
        if resp.status_code != 200:
            try:
                err = _format_error_payload(resp.json(), f"HTTP {resp.status_code}")
            except Exception:
                err = f"HTTP {resp.status_code} (非 JSON 响应)"
            st.error(f"API 错误 [{path}]: {err}")
            return None
        try:
            return resp.json()
        except Exception:
            st.error(f"后端返回了无效的 JSON 响应 (HTTP {resp.status_code})")
            return None
    except requests.exceptions.ConnectionError:
        st.error("无法连接到 Flask 后端。请在新终端中运行 `cd backend && python app.py` 启动后端。")
        return None
    except requests.exceptions.Timeout:
        st.error(f"请求超时 ({path})。请检查后端是否正常运行。")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"网络错误 ({path}): {e}")
        return None


# ── Data ──

def ensure_session(df=None):
    """Return a valid session_id, creating one if needed. Returns None only if
    both the stored session is invalid AND a new upload fails."""
    sid = st.session_state.get("session_id")
    if sid and session_is_valid():
        return sid

    if df is None or len(df) == 0:
        return None

    # Prefer re-uploading original file bytes (preserves dtypes)
    raw_bytes = st.session_state.get("_raw_file_bytes")
    raw_name = st.session_state.get("_raw_file_name", "data.csv")
    if raw_bytes:
        result = upload_data(raw_bytes, raw_name)
        if result:
            st.session_state.session_id = result["session_id"]
            if result.get("session_meta"):
                st.session_state.session_meta = result["session_meta"]
            return result["session_id"]

    # Fallback: CSV round-trip (lossy but guarantees data gets through)
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    result = upload_data(csv_bytes, "synced_data.csv")
    if result:
        st.session_state.session_id = result["session_id"]
        if result.get("session_meta"):
            st.session_state.session_meta = result["session_meta"]
        return result["session_id"]
    return None


def sync_session_data(sid, df):
    """Sync a DataFrame to an existing session without creating a new one.
    Updates the Flask backend's session data in-place."""
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    raw_name = st.session_state.get("_raw_file_name", "synced_data.csv")
    result = _post(f"/data/{sid}/sync", files={"file": (raw_name, csv_bytes)}, timeout=120)
    if result:
        if result.get("session_meta"):
            st.session_state.session_meta = result["session_meta"]
        return True
    # If sync fails (session gone), fall back to new upload
    raw_bytes = st.session_state.get("_raw_file_bytes")
    raw_name = st.session_state.get("_raw_file_name", "data.csv")
    if raw_bytes:
        result = upload_data(raw_bytes, raw_name)
    else:
        result = upload_data(csv_bytes, "data.csv")
    if result:
        st.session_state.session_id = result["session_id"]
        if result.get("session_meta"):
            st.session_state.session_meta = result["session_meta"]
        return True
    return False


def upload_data(file_bytes, filename):
    """Upload raw file bytes to Flask. Returns {session_id, ...} or None."""
    return _post("/data/upload", files={"file": (filename, file_bytes)},
                 timeout=120)


def try_upload_backend(file_bytes, filename):
    """Non-critical upload: show a warning on failure but never raise.

    Call this AFTER data is already stored locally, so the UI is not blocked.
    """
    import streamlit as st
    result = upload_data(file_bytes, filename)
    if result:
        st.session_state.session_id = result["session_id"]
        if result.get("session_meta"):
            st.session_state.session_meta = result["session_meta"]
        st.session_state.backend_data = result
        return True
    else:
        st.warning("⚠️ 数据已加载到前端。Flask 后端未连接，启动后端后点击“重新同步当前数据到后端”即可训练。")
        return False


def get_outliers(session_id):
    return _get(f"/data/{session_id}/outliers")


def process_data(session_id, operations):
    return _post(f"/data/{session_id}/process", json_data={"operations": operations})


def get_summary(session_id):
    return _get(f"/data/{session_id}/summary")


# ── Regression ──

def train_regression(session_id, target_col, feature_cols, lr, epochs, batch_size, device="cpu"):
    return _post("/regression/train", json_data={
        "session_id": session_id, "target_col": target_col,
        "feature_cols": feature_cols, "learning_rate": lr,
        "epochs": epochs, "batch_size": batch_size, "device": device
    }, timeout=600)


def predict_regression(features, device="cpu"):
    return _post("/regression/predict", json_data={"features": features, "device": device})


def batch_predict_regression(rows, device="cpu"):
    return _post("/regression/batch_predict", json_data={"rows": rows, "device": device})


def clear_regression():
    return _post("/regression/clear")


def regression_status():
    return _get("/regression/status")


def list_regression_versions():
    return _get("/regression/versions")


def activate_regression_version(version_id):
    return _post("/regression/activate", json_data={"version_id": version_id})


def delete_regression_version(version_id):
    return _delete(f"/regression/version/{version_id}")



# ── Classification ──

def train_classification(session_id, target_col, feature_cols, lr, epochs, batch_size, device="cpu"):
    return _post("/classification/train", json_data={
        "session_id": session_id, "target_col": target_col,
        "feature_cols": feature_cols, "learning_rate": lr,
        "epochs": epochs, "batch_size": batch_size, "device": device
    }, timeout=600)


def predict_classification(features, device="cpu"):
    return _post("/classification/predict", json_data={"features": features, "device": device})


def batch_predict_classification(rows, device="cpu"):
    return _post("/classification/batch_predict", json_data={"rows": rows, "device": device})


def clear_classification():
    return _post("/classification/clear")


def classification_status():
    return _get("/classification/status")


def list_classification_versions():
    return _get("/classification/versions")


def activate_classification_version(version_id):
    return _post("/classification/activate", json_data={"version_id": version_id})


def delete_classification_version(version_id):
    return _delete(f"/classification/version/{version_id}")


# ── DIY MLP ──

def train_diy_mlp(session_id, target_col, feature_cols, layers, task_type, n_classes,
                   lr, optimizer, epochs, batch_size, val_split, patience, device="cpu"):
    return _post("/diy_mlp/train", json_data={
        "session_id": session_id, "target_col": target_col,
        "feature_cols": feature_cols, "layers": layers,
        "task_type": task_type, "n_classes": n_classes,
        "learning_rate": lr, "optimizer": optimizer,
        "epochs": epochs, "batch_size": batch_size,
        "val_split": val_split, "patience": patience, "device": device
    }, timeout=600)


def predict_diy_mlp(features, device="cpu"):
    return _post("/diy_mlp/predict", json_data={"features": features, "device": device})


def batch_predict_diy_mlp(rows, device="cpu"):
    return _post("/diy_mlp/batch_predict", json_data={"rows": rows, "device": device})


def clear_diy_mlp():
    return _post("/diy_mlp/clear")


def diy_mlp_status():
    return _get("/diy_mlp/status")


def list_diy_mlp_versions():
    return _get("/diy_mlp/versions")


def activate_diy_mlp_version(version_id):
    return _post("/diy_mlp/activate", json_data={"version_id": version_id})


def delete_diy_mlp_version(version_id):
    return _delete(f"/diy_mlp/version/{version_id}")


# ── Decision Tree ──

def train_decision_tree(session_id, target_col, feature_cols, task_type, criterion, max_depth):
    return _post("/decision_tree/train", json_data={
        "session_id": session_id, "target_col": target_col,
        "feature_cols": feature_cols, "task_type": task_type,
        "criterion": criterion, "max_depth": max_depth
    })


def predict_decision_tree(input_dict, task_type):
    return _post("/decision_tree/predict", json_data={"input_dict": input_dict, "task_type": task_type})


def clear_decision_tree():
    return _post("/decision_tree/clear")


def decision_tree_status():
    return _get("/decision_tree/status")


def list_decision_tree_versions():
    return _get("/decision_tree/versions")


def activate_decision_tree_version(version_id):
    return _post("/decision_tree/activate", json_data={"version_id": version_id})


def delete_decision_tree_version(version_id):
    return _delete(f"/decision_tree/version/{version_id}")


# ── Clustering ──

def train_clustering(session_id, feature_cols, algorithm, params):
    return _post("/clustering/train", json_data={
        "session_id": session_id, "feature_cols": feature_cols,
        "algorithm": algorithm, "params": params
    })


def elbow_clustering(session_id, feature_cols, max_k):
    return _post("/clustering/elbow", json_data={
        "session_id": session_id, "feature_cols": feature_cols, "max_k": max_k
    })


def predict_clustering(features):
    return _post("/clustering/predict", json_data={"features": features})


def clear_clustering():
    return _post("/clustering/clear")


def clustering_status():
    return _get("/clustering/status")


def list_clustering_versions():
    return _get("/clustering/versions")


def activate_clustering_version(version_id):
    return _post("/clustering/activate", json_data={"version_id": version_id})


def delete_clustering_version(version_id):
    return _delete(f"/clustering/version/{version_id}")


# ── LLM ──

def chat_llm(api_base, api_key, model, messages):
    """Returns requests.Response for SSE streaming."""
    return requests.post(
        f"{API_BASE}/llm/chat",
        json={"api_base": api_base, "api_key": api_key,
              "model": model, "messages": messages},
        timeout=(CONNECT_TIMEOUT, 180),
        stream=True
    )
