"""Model version registry used by Flask routes and training services."""
import json
import os
import shutil
from datetime import datetime
from uuid import uuid4

MODELS_DIR = os.path.dirname(__file__)
REGISTRY_PATH = os.path.join(MODELS_DIR, "registry.json")
MODEL_TYPES = ("decision_tree", "clustering", "regression", "classification", "diy_mlp")


def _empty_registry():
    return {model_type: {"active": None, "versions": {}} for model_type in MODEL_TYPES}


def _ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for model_type in MODEL_TYPES:
        os.makedirs(os.path.join(MODELS_DIR, model_type), exist_ok=True)


def get_registry():
    _ensure_dirs()
    if not os.path.exists(REGISTRY_PATH):
        return _empty_registry()
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as fh:
            registry = json.load(fh)
    except Exception:
        registry = _empty_registry()
    for model_type in MODEL_TYPES:
        registry.setdefault(model_type, {"active": None, "versions": {}})
        registry[model_type].setdefault("versions", {})
    return registry


def save_registry(registry):
    _ensure_dirs()
    with open(REGISTRY_PATH, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, ensure_ascii=False, indent=2)


def generate_version_id():
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:4]}"


def create_version_dir(model_type, version_id):
    path = os.path.join(MODELS_DIR, model_type, version_id)
    os.makedirs(path, exist_ok=True)
    return path


def register_version(model_type, version_id, metadata, files):
    registry = get_registry()
    registry.setdefault(model_type, {"active": None, "versions": {}})
    entry = {
        **metadata,
        "version_id": version_id,
        "created_at": metadata.get("created_at") or datetime.now().isoformat(timespec="seconds"),
        "files": files,
    }
    registry[model_type]["versions"][version_id] = entry
    registry[model_type]["active"] = version_id
    save_registry(registry)
    return entry


def list_versions(model_type):
    versions = get_registry().get(model_type, {}).get("versions", {})
    return sorted(
        ({"version_id": vid, **meta} for vid, meta in versions.items()),
        key=lambda item: item.get("created_at", ""),
        reverse=True,
    )


def get_version(model_type, version_id):
    if not version_id:
        return None
    return get_registry().get(model_type, {}).get("versions", {}).get(version_id)


def get_active_version(model_type):
    return get_registry().get(model_type, {}).get("active")


def activate_version(model_type, version_id):
    registry = get_registry()
    if version_id not in registry.get(model_type, {}).get("versions", {}):
        return False
    registry[model_type]["active"] = version_id
    save_registry(registry)
    return True


def delete_version(model_type, version_id):
    registry = get_registry()
    versions = registry.get(model_type, {}).get("versions", {})
    if version_id not in versions:
        return False
    versions.pop(version_id, None)
    if registry[model_type].get("active") == version_id:
        registry[model_type]["active"] = next(iter(versions), None)
    shutil.rmtree(os.path.join(MODELS_DIR, model_type, version_id), ignore_errors=True)
    save_registry(registry)
    return True


def get_model_paths(model_type, version_id=None):
    version_id = version_id or get_active_version(model_type)
    meta = get_version(model_type, version_id)
    if not meta:
        return None, None
    base = os.path.join(MODELS_DIR, model_type, version_id)
    paths = {key: os.path.join(base, rel) for key, rel in meta.get("files", {}).items()}
    if not all(os.path.exists(path) for path in paths.values()):
        return None, meta
    return paths, meta


def cleanup_orphaned_versions():
    registry = get_registry()
    changed = False
    for model_type in MODEL_TYPES:
        versions = registry.get(model_type, {}).get("versions", {})
        for vid, meta in list(versions.items()):
            base = os.path.join(MODELS_DIR, model_type, vid)
            files = [os.path.join(base, rel) for rel in meta.get("files", {}).values()]
            if files and not all(os.path.exists(path) for path in files):
                versions.pop(vid, None)
                if registry[model_type].get("active") == vid:
                    registry[model_type]["active"] = None
                changed = True
    if changed:
        save_registry(registry)
