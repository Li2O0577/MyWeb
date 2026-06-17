"""Minimal local account store with cookie sessions.

This is intentionally small and file-backed for a single-machine deployment.
It can later be replaced by a database without changing route-level auth calls.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time

AUTH_DIR = os.path.join(os.path.dirname(__file__), "auth")
USERS_PATH = os.path.join(AUTH_DIR, "users.json")
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
COOKIE_NAME = "myweb1_auth"

_lock = threading.RLock()
_sessions = {}


def _ensure_dir():
    os.makedirs(AUTH_DIR, exist_ok=True)


def _load_users():
    _ensure_dir()
    if not os.path.exists(USERS_PATH):
        return {"users": {}}
    try:
        with open(USERS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {"users": {}}
    data.setdefault("users", {})
    return data


def _save_users(data):
    _ensure_dir()
    with open(USERS_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def _password_hash(password, salt=None):
    salt_bytes = base64.b64decode(salt.encode("ascii")) if salt else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 200_000)
    return {
        "salt": base64.b64encode(salt_bytes).decode("ascii"),
        "hash": base64.b64encode(digest).decode("ascii"),
    }


def _public_user(user):
    return {
        "user_id": user["user_id"],
        "username": user["username"],
        "created_at": user.get("created_at"),
    }


def create_user(username, password):
    username = str(username or "").strip()
    password = str(password or "")
    if len(username) < 3 or len(username) > 40:
        return None, "用户名长度需要在 3 到 40 个字符之间。"
    if len(password) < 8:
        return None, "密码至少需要 8 个字符。"
    with _lock:
        data = _load_users()
        key = username.lower()
        if key in data["users"]:
            return None, "用户名已存在。"
        secret = _password_hash(password)
        user = {
            "user_id": secrets.token_hex(12),
            "username": username,
            "password_salt": secret["salt"],
            "password_hash": secret["hash"],
            "created_at": time.time(),
        }
        data["users"][key] = user
        _save_users(data)
        return _public_user(user), None


def verify_user(username, password):
    username = str(username or "").strip()
    password = str(password or "")
    with _lock:
        user = _load_users()["users"].get(username.lower())
    if not user:
        return None
    expected = user.get("password_hash", "")
    actual = _password_hash(password, user.get("password_salt", ""))["hash"]
    if not hmac.compare_digest(expected, actual):
        return None
    return _public_user(user)


def create_session(user):
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _lock:
        _sessions[token] = {
            "user": user,
            "created_at": now,
            "expires_at": now + SESSION_TTL_SECONDS,
        }
    return token


def get_session_user(token):
    if not token:
        return None
    with _lock:
        session = _sessions.get(token)
        if not session:
            return None
        if session.get("expires_at", 0) < time.time():
            _sessions.pop(token, None)
            return None
        return dict(session["user"])


def delete_session(token):
    if not token:
        return
    with _lock:
        _sessions.pop(token, None)


def clear_auth_state():
    with _lock:
        _sessions.clear()
