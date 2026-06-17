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
    return {model_type: {"active": None, "active_by_user": {}, "versions": {}} for model_type in MODEL_TYPES}


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
        registry.setdefault(model_type, {"active": None, "active_by_user": {}, "versions": {}})
        registry[model_type].setdefault("active_by_user", {})
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


def _user_versions(versions, user_id=None):
    if user_id is None:
        return versions
    return {
        vid: meta
        for vid, meta in versions.items()
        if str(meta.get("user_id") or "") == str(user_id)
    }


def register_version(model_type, version_id, metadata, files):
    registry = get_registry()
    registry.setdefault(model_type, {"active": None, "active_by_user": {}, "versions": {}})
    registry[model_type].setdefault("active_by_user", {})
    entry = {
        **metadata,
        "version_id": version_id,
        "created_at": metadata.get("created_at") or datetime.now().isoformat(timespec="seconds"),
        "files": files,
    }
    registry[model_type]["versions"][version_id] = entry
    user_id = metadata.get("user_id")
    if user_id:
        registry[model_type]["active_by_user"][str(user_id)] = version_id
    else:
        registry[model_type]["active"] = version_id
    save_registry(registry)
    return entry


def list_versions(model_type, user_id=None):
    versions = get_registry().get(model_type, {}).get("versions", {})
    versions = _user_versions(versions, user_id=user_id)
    return sorted(
        ({"version_id": vid, **meta} for vid, meta in versions.items()),
        key=lambda item: item.get("created_at", ""),
        reverse=True,
    )


def get_version(model_type, version_id, user_id=None):
    if not version_id:
        return None
    meta = get_registry().get(model_type, {}).get("versions", {}).get(version_id)
    if not meta:
        return None
    if user_id is not None and str(meta.get("user_id") or "") != str(user_id):
        return None
    return meta


def get_active_version(model_type, user_id=None):
    model_registry = get_registry().get(model_type, {})
    if user_id is not None:
        active = model_registry.get("active_by_user", {}).get(str(user_id))
        if active and get_version(model_type, active, user_id=user_id):
            return active
        versions = list_versions(model_type, user_id=user_id)
        return versions[0]["version_id"] if versions else None
    return model_registry.get("active")


def activate_version(model_type, version_id, user_id=None):
    registry = get_registry()
    meta = registry.get(model_type, {}).get("versions", {}).get(version_id)
    if not meta:
        return False
    if user_id is not None and str(meta.get("user_id") or "") != str(user_id):
        return False
    registry[model_type].setdefault("active_by_user", {})
    if user_id is not None:
        registry[model_type]["active_by_user"][str(user_id)] = version_id
    else:
        registry[model_type]["active"] = version_id
    save_registry(registry)
    return True


def delete_version(model_type, version_id, user_id=None):
    registry = get_registry()
    versions = registry.get(model_type, {}).get("versions", {})
    if version_id not in versions:
        return False
    if user_id is not None and str(versions[version_id].get("user_id") or "") != str(user_id):
        return False
    versions.pop(version_id, None)
    if user_id is not None:
        active_by_user = registry[model_type].setdefault("active_by_user", {})
        if active_by_user.get(str(user_id)) == version_id:
            remaining = list_versions(model_type, user_id=user_id)
            active_by_user[str(user_id)] = remaining[0]["version_id"] if remaining else None
    elif registry[model_type].get("active") == version_id:
        registry[model_type]["active"] = next(iter(versions), None)
    shutil.rmtree(os.path.join(MODELS_DIR, model_type, version_id), ignore_errors=True)
    save_registry(registry)
    return True


def get_model_paths(model_type, version_id=None, user_id=None):
    version_id = version_id or get_active_version(model_type, user_id=user_id)
    meta = get_version(model_type, version_id, user_id=user_id)
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
