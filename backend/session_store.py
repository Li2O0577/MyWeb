"""Session storage shared by Flask routes and the app entrypoint."""
import os
import pickle
import threading
import time
import uuid
from datetime import datetime, timezone

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

_sessions = {}
_sessions_lock = threading.Lock()
SESSION_TTL = 3600  # 1 hour


def _now_iso(ts=None):
    ts = time.time() if ts is None else ts
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _build_meta(df, source_name=None, created_at=None, updated_at=None):
    now = time.time()
    created_at = created_at or now
    updated_at = updated_at or now
    return {
        "created_at": created_at,
        "created_at_iso": _now_iso(created_at),
        "updated_at": updated_at,
        "updated_at_iso": _now_iso(updated_at),
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "n_columns": int(len(df.columns)),
        "source_name": source_name or "unknown",
    }


def _normalize_record(record, sid=None):
    """Upgrade older pickle records that only had df/at."""
    if not record or "df" not in record:
        return None

    df = record["df"]
    at = record.get("at") or record.get("updated_at") or time.time()
    meta = record.get("meta") or _build_meta(df, created_at=at, updated_at=at)
    meta.setdefault("source_name", "unknown")
    meta.setdefault("created_at", at)
    meta.setdefault("created_at_iso", _now_iso(meta["created_at"]))
    meta.setdefault("updated_at", at)
    meta.setdefault("updated_at_iso", _now_iso(meta["updated_at"]))
    meta["last_accessed_at"] = at
    meta["last_accessed_at_iso"] = _now_iso(at)
    meta["rows"] = int(len(df))
    meta["columns"] = [str(c) for c in df.columns]
    meta["n_columns"] = int(len(df.columns))
    if sid:
        meta["session_id"] = sid
    return {"df": df, "at": at, "meta": meta}


def _session_path(sid):
    return os.path.join(SESSIONS_DIR, f"{sid}.pkl")


def _save_to_disk(sid, record):
    try:
        with open(_session_path(sid), "wb") as f:
            pickle.dump(record, f)
    except Exception:
        pass


def _load_from_disk(sid):
    path = _session_path(sid)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return _normalize_record(pickle.load(f), sid)
        except Exception:
            pass
    return None


def _delete_from_disk(sid):
    path = _session_path(sid)
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def create_session(df, source_name=None):
    sid = str(uuid.uuid4())[:8]
    now = time.time()
    record = {"df": df, "at": now, "meta": _build_meta(df, source_name, now, now)}
    record["meta"]["session_id"] = sid
    with _sessions_lock:
        _sessions[sid] = record
    _save_to_disk(sid, record)
    return sid


def get_session(sid):
    with _sessions_lock:
        s = _normalize_record(_sessions.get(sid), sid)
        if s and time.time() - s["at"] < SESSION_TTL:
            s["at"] = time.time()
            s["meta"]["last_accessed_at"] = s["at"]
            s["meta"]["last_accessed_at_iso"] = _now_iso(s["at"])
            _sessions[sid] = s
            _save_to_disk(sid, s)
            return s["df"]
        if s:
            del _sessions[sid]
            _delete_from_disk(sid)
            return None

    disk_data = _load_from_disk(sid)
    if disk_data and time.time() - disk_data["at"] < SESSION_TTL:
        disk_data["at"] = time.time()
        disk_data["meta"]["last_accessed_at"] = disk_data["at"]
        disk_data["meta"]["last_accessed_at_iso"] = _now_iso(disk_data["at"])
        with _sessions_lock:
            _sessions[sid] = disk_data
        _save_to_disk(sid, disk_data)
        return disk_data["df"]
    if disk_data:
        _delete_from_disk(sid)

    return None


def update_session(sid, df, source_name=None):
    """Update the DataFrame stored in a session."""
    now = time.time()
    with _sessions_lock:
        existing = _normalize_record(_sessions.get(sid), sid)
        if existing:
            created_at = existing["meta"].get("created_at", existing.get("at", now))
            source = source_name or existing["meta"].get("source_name")
            record = {"df": df, "at": now, "meta": _build_meta(df, source, created_at, now)}
            record["meta"]["session_id"] = sid
            _sessions[sid] = record
            _save_to_disk(sid, record)
            return True

    disk_data = _load_from_disk(sid)
    if disk_data:
        created_at = disk_data["meta"].get("created_at", disk_data.get("at", now))
        source = source_name or disk_data["meta"].get("source_name")
        record = {"df": df, "at": now, "meta": _build_meta(df, source, created_at, now)}
        record["meta"]["session_id"] = sid
        with _sessions_lock:
            _sessions[sid] = record
        _save_to_disk(sid, record)
        return True
    return False


def active_session_count():
    cleanup_expired()
    with _sessions_lock:
        return len(_sessions)


def get_session_meta(sid):
    with _sessions_lock:
        record = _normalize_record(_sessions.get(sid), sid)
    if not record:
        record = _load_from_disk(sid)
    if not record or time.time() - record["at"] >= SESSION_TTL:
        return None
    meta = dict(record["meta"])
    meta["ttl_seconds_remaining"] = max(0, int(SESSION_TTL - (time.time() - record["at"])))
    return meta


def recent_sessions(limit=5):
    cleanup_expired()
    with _sessions_lock:
        records = [(sid, _normalize_record(record, sid)) for sid, record in _sessions.items()]
    metas = []
    for sid, record in records:
        if not record:
            continue
        meta = dict(record["meta"])
        meta["session_id"] = sid
        meta["ttl_seconds_remaining"] = max(0, int(SESSION_TTL - (time.time() - record["at"])))
        metas.append(meta)
    metas.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
    return metas[:limit]


def restore_sessions():
    now = time.time()
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith(".pkl"):
            continue
        sid = fname[:-4]
        if sid in _sessions:
            continue
        disk_data = _load_from_disk(sid)
        if disk_data and now - disk_data["at"] < SESSION_TTL:
            with _sessions_lock:
                _sessions[sid] = disk_data


def cleanup_expired():
    now = time.time()
    with _sessions_lock:
        expired = [sid for sid, s in _sessions.items() if now - s["at"] >= SESSION_TTL]
    for sid in expired:
        with _sessions_lock:
            if sid in _sessions:
                del _sessions[sid]
        _delete_from_disk(sid)
