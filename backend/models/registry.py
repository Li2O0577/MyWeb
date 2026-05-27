"""Model version registry — manages versioned model save/load/activate/delete."""
import os
import json
import threading
import time
import uuid
from datetime import datetime, timezone

MODELS_DIR = os.path.dirname(__file__)
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")
_lock = threading.Lock()

MODEL_TYPES = ["regression", "classification", "diy_mlp", "decision_tree", "clustering"]


def _now_iso():
    return datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat()


def _read_registry():
    if not os.path.exists(REGISTRY_PATH):
        return {}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_registry(data):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _version_dir(model_type, version_id):
    return os.path.join(MODELS_DIR, model_type, version_id)


def get_registry():
    """Return full registry dict (thread-safe read)."""
    with _lock:
        return _read_registry()


def get_active_version(model_type):
    """Return version_id of the active version, or None."""
    reg = get_registry()
    return reg.get(model_type, {}).get("active")


def list_versions(model_type):
    """Return list of version metadata dicts, newest first."""
    reg = get_registry()
    versions = reg.get(model_type, {}).get("versions", {})
    result = []
    for vid, meta in versions.items():
        result.append({"version_id": vid, **meta})
    result.sort(key=lambda v: v.get("created_at", ""), reverse=True)
    return result


def get_version(model_type, version_id):
    """Return a single version's metadata, or None."""
    reg = get_registry()
    return reg.get(model_type, {}).get("versions", {}).get(version_id)


def register_version(model_type, version_id, metadata, files):
    """Save version metadata + install as active.

    metadata: dict with created_at, dataset_name, session_id, features, target, metrics, params
    files: dict like {"model": "model.pth", "scaler": "scaler.pkl", "config": "config.json"}
    """
    with _lock:
        reg = _read_registry()
        reg.setdefault(model_type, {"versions": {}, "active": None})

        entry = dict(metadata)
        entry.setdefault("created_at", _now_iso())
        entry["files"] = files

        reg[model_type]["versions"][version_id] = entry
        reg[model_type]["active"] = version_id

        _write_registry(reg)

    return version_id


def activate_version(model_type, version_id):
    """Set a version as active. Returns True on success."""
    reg = get_registry()
    if version_id not in reg.get(model_type, {}).get("versions", {}):
        return False

    reg[model_type]["active"] = version_id
    with _lock:
        _write_registry(reg)
    return True


def delete_version(model_type, version_id):
    """Delete a version from registry and disk."""
    import shutil
    with _lock:
        reg = _read_registry()
        versions = reg.get(model_type, {}).get("versions", {})
        if version_id not in versions:
            return False

        vdir = _version_dir(model_type, version_id)
        if os.path.exists(vdir):
            shutil.rmtree(vdir)

        del versions[version_id]

        # If we deleted the active version, pick the newest remaining
        if reg[model_type]["active"] == version_id:
            remaining = list(versions.keys())
            reg[model_type]["active"] = remaining[-1] if remaining else None

        _write_registry(reg)
    return True


def create_version_dir(model_type, version_id):
    """Create the version subdirectory. Returns the path."""
    vdir = _version_dir(model_type, version_id)
    os.makedirs(vdir, exist_ok=True)
    return vdir


def generate_version_id():
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"


def migrate_legacy_files(model_type, legacy_files):
    """Migrate old flat-file models into the versioned registry.

    legacy_files: dict {"model": "reg_best_model.pth", "scaler": "reg_scaler.pkl", "config": "reg_config.json"}
    Returns version_id if migration happened, or None.
    """
    import shutil

    # Check if legacy files exist
    legacy_model = os.path.join(MODELS_DIR, legacy_files.get("model", ""))
    legacy_config = os.path.join(MODELS_DIR, legacy_files.get("config", ""))

    if not os.path.exists(legacy_model) or not os.path.exists(legacy_config):
        return None

    # Already migrated? Check registry
    reg = get_registry()
    if reg.get(model_type, {}).get("versions"):
        return None

    # Read config to extract metadata
    with open(legacy_config, "r", encoding="utf-8") as f:
        config = json.load(f)

    version_id = f"legacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    vdir = create_version_dir(model_type, version_id)

    new_files = {}
    for role, old_name in legacy_files.items():
        old_path = os.path.join(MODELS_DIR, old_name)
        new_name = os.path.basename(old_name)
        new_path = os.path.join(vdir, new_name)
        if os.path.exists(old_path):
            shutil.copy2(old_path, new_path)
            new_files[role] = new_name

    # Build metrics
    metrics = {}
    for k in ("r2", "mae", "rmse", "acc"):
        if k in config:
            metrics[k] = config[k]

    metadata = {
        "created_at": _now_iso(),
        "dataset_name": "migrated",
        "session_id": "",
        "features": config.get("features", []),
        "target": config.get("target", ""),
        "metrics": metrics,
        "params": {k: v for k, v in config.items()
                   if k not in ("features", "target", "label_map", "reverse_label_map",
                                "r2", "mae", "rmse", "acc")},
    }

    register_version(model_type, version_id, metadata, new_files)
    return version_id


# Legacy file name mappings per model type
LEGACY_FILES = {
    "regression": {"model": "reg_best_model.pth", "scaler": "reg_scaler.pkl", "config": "reg_config.json"},
    "classification": {"model": "cls_best_model.pth", "scaler": "cls_scaler.pkl", "config": "cls_config.json"},
    "diy_mlp": {"model": "diy_best_model.pth", "scaler": "diy_scaler.pkl", "config": "diy_config.json"},
    "decision_tree": {"model": "dt_model.pkl", "config": "dt_config.json"},
    "clustering": {"model": "cluster_model.pkl", "scaler": "cluster_scaler.pkl", "config": "cluster_config.json"},
}


def get_model_paths(model_type, version_id=None):
    """Return {model, scaler, config} absolute paths for the given version.

    If version_id is None, uses the active version.
    Returns (paths_dict, metadata) or (None, None) if no model found.
    """
    if version_id is None:
        version_id = get_active_version(model_type)

    if not version_id:
        # Try legacy migration
        legacy = LEGACY_FILES.get(model_type)
        if legacy:
            migrated_vid = migrate_legacy_files(model_type, legacy)
            if migrated_vid:
                version_id = migrated_vid

    if not version_id:
        return None, None

    meta = get_version(model_type, version_id)
    if not meta:
        return None, None

    vdir = _version_dir(model_type, version_id)
    files = meta.get("files", {})
    paths = {}
    for role, fname in files.items():
        p = os.path.join(vdir, fname)
        if os.path.exists(p):
            paths[role] = p

    return paths, meta


def cleanup_orphaned_versions():
    """Remove registry entries whose model file no longer exists on disk.

    Called once at startup so stale entries from manual file deletion are
    purged before any API request sees them.
    """
    reg = _read_registry()
    if not reg:
        return
    changed = False

    for model_type in MODEL_TYPES:
        versions = reg.get(model_type, {}).get("versions", {})
        if not versions:
            continue
        orphaned = []
        for vid, meta in list(versions.items()):
            vdir = _version_dir(model_type, vid)
            files = meta.get("files", {})
            model_file = files.get("model", "")
            model_path = os.path.join(vdir, model_file) if model_file else ""
            if not model_path or not os.path.exists(model_path):
                orphaned.append(vid)
                # Remove leftover directory if present
                if os.path.exists(vdir):
                    import shutil
                    shutil.rmtree(vdir, ignore_errors=True)

        for vid in orphaned:
            del versions[vid]
            changed = True

        # If the active version was orphaned, pick the newest remaining
        if reg[model_type].get("active") in orphaned:
            remaining = list(versions.keys())
            reg[model_type]["active"] = remaining[-1] if remaining else None

    if changed:
        with _lock:
            _write_registry(reg)
