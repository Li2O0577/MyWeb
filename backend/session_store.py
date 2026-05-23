"""Session storage — memory cache + disk persistence (Parquet + JSON).

Replaced pickle with Parquet/JSON to eliminate deserialization code-execution risk.
Old .pkl sessions are migrated on first access or discarded if expired.
"""
import os
import json
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone

import pandas as pd

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


# ── Parquet + JSON disk I/O ──

def _parquet_path(sid):
    return os.path.join(SESSIONS_DIR, f"{sid}.parquet")


def _meta_path(sid):
    return os.path.join(SESSIONS_DIR, f"{sid}.meta.json")


def _save_to_disk(sid, record):
    """Persist a session record as Parquet (DataFrame) + JSON (metadata)."""
    df = record.get("df")
    meta = record.get("meta", {})
    try:
        df.to_parquet(_parquet_path(sid), index=False)
    except Exception:
        return
    try:
        with open(_meta_path(sid), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
    except Exception:
        # If meta write fails, clean up the parquet so we don't have orphan data
        try:
            os.remove(_parquet_path(sid))
        except Exception:
            pass


def _load_from_disk(sid):
    """Load a session record from Parquet + JSON. Returns dict or None."""
    pq_path = _parquet_path(sid)
    meta_path = _meta_path(sid)
    if not os.path.exists(pq_path):
        return _migrate_legacy_pickle(sid)
    try:
        df = pd.read_parquet(pq_path)
    except Exception:
        return None
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass
    now = time.time()
    meta.setdefault("created_at", now)
    meta.setdefault("created_at_iso", _now_iso(now))
    meta.setdefault("updated_at", now)
    meta.setdefault("updated_at_iso", _now_iso(now))
    meta["last_accessed_at"] = now
    meta["last_accessed_at_iso"] = _now_iso(now)
    meta["rows"] = int(len(df))
    meta["columns"] = [str(c) for c in df.columns]
    meta["n_columns"] = int(len(df.columns))
    meta["session_id"] = sid
    return {"df": df, "at": now, "meta": meta}


def _delete_from_disk(sid):
    """Remove both Parquet and JSON files for a session."""
    for path_fn in (_parquet_path, _meta_path):
        path = path_fn(sid)
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
    # Also clean up any legacy .pkl file
    legacy = os.path.join(SESSIONS_DIR, f"{sid}.pkl")
    if os.path.exists(legacy):
        try:
            os.remove(legacy)
        except Exception:
            pass


# ── Legacy pickle migration ──

def _migrate_legacy_pickle(sid):
    """Attempt to load an old .pkl session and migrate it to Parquet + JSON.

    Once the pickle has been migrated it is deleted so the unsafe payload
    is gone from disk.  If the pickle is unreadable it is discarded.
    """
    import pickle

    legacy_path = os.path.join(SESSIONS_DIR, f"{sid}.pkl")
    if not os.path.exists(legacy_path):
        return None

    try:
        with open(legacy_path, "rb") as f:
            raw = pickle.load(f)
    except Exception:
        # Corrupt or malicious pickle — discard
        try:
            os.remove(legacy_path)
        except Exception:
            pass
        return None

    if not isinstance(raw, dict) or "df" not in raw:
        try:
            os.remove(legacy_path)
        except Exception:
            pass
        return None

    df = raw["df"]
    if not isinstance(df, pd.DataFrame):
        try:
            os.remove(legacy_path)
        except Exception:
            pass
        return None

    at = raw.get("at") or raw.get("updated_at") or time.time()
    meta = raw.get("meta") or _build_meta(df, created_at=at, updated_at=at)
    record = {"df": df, "at": at, "meta": meta}
    _save_to_disk(sid, record)

    # Remove the legacy pickle after successful migration
    try:
        os.remove(legacy_path)
    except Exception:
        pass

    return record


def _normalize_record(record, sid=None):
    """Ensure a memory record has the expected shape."""
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


# ── Public API ──

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
    """Restore sessions from Parquet files on startup. Migrates legacy .pkl files."""
    now = time.time()
    for fname in os.listdir(SESSIONS_DIR):
        if fname.endswith(".parquet"):
            sid = fname[:-8]  # remove .parquet
            if sid in _sessions:
                continue
            disk_data = _load_from_disk(sid)
            if disk_data and now - disk_data["at"] < SESSION_TTL:
                with _sessions_lock:
                    _sessions[sid] = disk_data
        elif fname.endswith(".pkl"):
            # Legacy pickle — migrate on startup
            sid = fname[:-4]
            if sid in _sessions:
                continue
            disk_data = _load_from_disk(sid)  # triggers _migrate_legacy_pickle
            if disk_data and now - disk_data["at"] < SESSION_TTL:
                with _sessions_lock:
                    _sessions[sid] = disk_data


def cleanup_expired():
    now = time.time()
    with _sessions_lock:
        expired = [(sid, s["at"]) for sid, s in _sessions.items() if now - s["at"] >= SESSION_TTL]
    for sid, cached_at in expired:
        with _sessions_lock:
            s = _sessions.get(sid)
            # Re-verify TTL: the session may have been refreshed between
            # collecting the expired list and acquiring this lock.
            if s is None or now - s["at"] < SESSION_TTL:
                continue
            del _sessions[sid]
        _delete_from_disk(sid)
