"""In-memory session registry with lightweight disk persistence."""
import json
import os
import time
import uuid

import pandas as pd

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "sessions")
SESSION_TTL_SECONDS = 24 * 60 * 60

_sessions = {}


def _ensure_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def _paths(sid):
    return (
        os.path.join(SESSIONS_DIR, f"{sid}.pkl"),
        os.path.join(SESSIONS_DIR, f"{sid}.meta.json"),
    )


def _state_path(sid, state_id):
    return os.path.join(SESSIONS_DIR, f"{sid}.state.{state_id}.pkl")


def _new_state_id():
    return uuid.uuid4().hex[:12]


def _state_entry(state_id, df, label, operations=None, messages=None):
    return {
        "state_id": state_id,
        "label": label,
        "operations": _json_safe(operations or []),
        "messages": _json_safe(messages or []),
        "created_at": time.time(),
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
    }


def _json_safe(value):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return []


def _write_session(sid):
    _ensure_dir()
    data_path, meta_path = _paths(sid)
    record = _sessions[sid]
    record["df"].to_pickle(data_path)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(record["meta"], fh, ensure_ascii=False, indent=2)


def _write_state(sid, state_id, df):
    _ensure_dir()
    df.copy().to_pickle(_state_path(sid, state_id))


def _ensure_history(sid):
    record = _sessions.get(sid)
    if not record:
        return None

    meta = record["meta"]
    if "pipelines" not in meta or not isinstance(meta.get("pipelines"), list):
        meta["pipelines"] = []

    history = meta.get("history")
    if not isinstance(history, list) or not history:
        state_id = _new_state_id()
        _write_state(sid, state_id, record["df"])
        meta["history"] = [_state_entry(state_id, record["df"], "原始数据")]
        meta["history_index"] = 0
        _write_session(sid)
    else:
        index = int(meta.get("history_index", len(history) - 1))
        meta["history_index"] = max(0, min(index, len(history) - 1))
        current = meta["history"][meta["history_index"]]
        if current.get("state_id") and not os.path.exists(_state_path(sid, current["state_id"])):
            _write_state(sid, current["state_id"], record["df"])
            _write_session(sid)
    return meta


def create_session(df, source_name=""):
    sid = uuid.uuid4().hex
    now = time.time()
    state_id = _new_state_id()
    _sessions[sid] = {
        "df": df.copy(),
        "meta": {
            "session_id": sid,
            "source_name": source_name or "",
            "created_at": now,
            "updated_at": now,
            "n_rows": int(len(df)),
            "n_cols": int(len(df.columns)),
            "history": [_state_entry(state_id, df, "原始数据")],
            "history_index": 0,
            "pipelines": [],
        },
    }
    _write_state(sid, state_id, df)
    _write_session(sid)
    return sid


def get_session(sid):
    record = _sessions.get(sid)
    if record:
        return record["df"]

    data_path, meta_path = _paths(sid)
    if not os.path.exists(data_path) or not os.path.exists(meta_path):
        return None

    try:
        df = pd.read_pickle(data_path)
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception:
        return None

    _sessions[sid] = {"df": df, "meta": meta}
    _ensure_history(sid)
    return df


def get_session_meta(sid):
    if get_session(sid) is None:
        return None
    _ensure_history(sid)
    return dict(_sessions[sid]["meta"])


def update_session(sid, df, source_name=None, history_entry=None):
    if get_session(sid) is None:
        return False
    meta = _sessions[sid]["meta"]
    _ensure_history(sid)
    meta = _sessions[sid]["meta"]
    meta["updated_at"] = time.time()
    meta["n_rows"] = int(len(df))
    meta["n_cols"] = int(len(df.columns))
    if source_name is not None:
        meta["source_name"] = source_name

    if history_entry:
        history = list(meta.get("history") or [])
        current_index = int(meta.get("history_index", len(history) - 1))
        history = history[: current_index + 1]
        state_id = _new_state_id()
        label = str(history_entry.get("label") or "数据处理")
        operations = history_entry.get("operations") or []
        messages = history_entry.get("messages") or []
        _write_state(sid, state_id, df)
        history.append(_state_entry(state_id, df, label, operations, messages))
        meta["history"] = history[-50:]
        meta["history_index"] = len(meta["history"]) - 1

    _sessions[sid] = {"df": df.copy(), "meta": meta}
    _write_session(sid)
    return True


def processing_history(sid):
    if get_session(sid) is None:
        return None
    meta = _ensure_history(sid)
    history = meta.get("history") or []
    index = int(meta.get("history_index", len(history) - 1)) if history else -1
    return {
        "history": history,
        "current_index": index,
        "can_undo": index > 0,
        "can_redo": 0 <= index < len(history) - 1,
        "pipelines": meta.get("pipelines") or [],
    }


def _restore_history_state(sid, target_index):
    if get_session(sid) is None:
        return None
    meta = _ensure_history(sid)
    history = meta.get("history") or []
    if target_index < 0 or target_index >= len(history):
        return None
    state_id = history[target_index].get("state_id")
    path = _state_path(sid, state_id)
    if not state_id or not os.path.exists(path):
        return None
    df = pd.read_pickle(path)
    meta["history_index"] = target_index
    meta["updated_at"] = time.time()
    meta["n_rows"] = int(len(df))
    meta["n_cols"] = int(len(df.columns))
    _sessions[sid] = {"df": df.copy(), "meta": meta}
    _write_session(sid)
    return df


def undo_session(sid):
    hist = processing_history(sid)
    if not hist or not hist["can_undo"]:
        return None
    return _restore_history_state(sid, hist["current_index"] - 1)


def redo_session(sid):
    hist = processing_history(sid)
    if not hist or not hist["can_redo"]:
        return None
    return _restore_history_state(sid, hist["current_index"] + 1)


def current_pipeline_operations(sid):
    hist = processing_history(sid)
    if not hist:
        return None
    operations = []
    for entry in hist["history"][1 : hist["current_index"] + 1]:
        operations.extend(entry.get("operations") or [])
    return operations


def save_pipeline(sid, name=None, operations=None):
    if get_session(sid) is None:
        return None
    meta = _ensure_history(sid)
    ops = _json_safe(operations if operations is not None else current_pipeline_operations(sid))
    if not ops:
        return None
    pipeline = {
        "pipeline_id": uuid.uuid4().hex[:12],
        "name": str(name or "").strip() or f"处理流水线 {len(meta.get('pipelines') or []) + 1}",
        "operations": ops,
        "created_at": time.time(),
        "step_count": len(ops),
    }
    pipelines = [pipeline, *(meta.get("pipelines") or [])]
    meta["pipelines"] = pipelines[:20]
    _write_session(sid)
    return pipeline


def get_pipeline(sid, pipeline_id):
    hist = processing_history(sid)
    if not hist:
        return None
    for pipeline in hist.get("pipelines") or []:
        if str(pipeline.get("pipeline_id")) == str(pipeline_id):
            return pipeline
    return None


def restore_sessions():
    _ensure_dir()
    for meta_name in os.listdir(SESSIONS_DIR):
        if not meta_name.endswith(".meta.json"):
            continue
        sid = meta_name[:-10]
        get_session(sid)


def cleanup_expired():
    now = time.time()
    expired = []
    for sid, record in list(_sessions.items()):
        if now - float(record["meta"].get("updated_at", 0)) > SESSION_TTL_SECONDS:
            expired.append(sid)
    for sid in expired:
        _sessions.pop(sid, None)
        paths = list(_paths(sid))
        try:
            paths.extend(os.path.join(SESSIONS_DIR, name) for name in os.listdir(SESSIONS_DIR) if name.startswith(f"{sid}.state."))
        except FileNotFoundError:
            pass
        for path in paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


def active_session_count():
    return len(_sessions)


def recent_sessions(limit=8):
    records = sorted(
        (dict(item["meta"]) for item in _sessions.values()),
        key=lambda item: item.get("updated_at", 0),
        reverse=True,
    )
    return records[:limit]
