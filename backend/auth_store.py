"""Local SQLite account store with persistent cookie sessions."""
import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time

import database
from resource_limits import (
    LOGIN_LOCK_SECONDS,
    LOGIN_MAX_FAILURES,
    LOGIN_WINDOW_SECONDS,
    MAX_AUDIT_LOGS,
    REGISTER_MAX_ATTEMPTS,
    REGISTER_WINDOW_SECONDS,
)

AUTH_DIR = os.path.join(os.path.dirname(__file__), "auth")
USERS_PATH = os.path.join(AUTH_DIR, "users.json")
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
COOKIE_NAME = "myweb1_auth"

_lock = threading.RLock()
_login_failures = {}
_login_blocked_until = {}
_registration_attempts = {}
_migrated_sources = set()


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


def _migrate_legacy_users():
    """Import the old JSON account file without modifying or deleting it."""
    source_key = (os.path.abspath(database.DB_PATH), os.path.abspath(USERS_PATH))
    if source_key in _migrated_sources or not os.path.exists(USERS_PATH):
        return
    data = _load_users()
    users = data.get("users") or {}
    if not users:
        _migrated_sources.add(source_key)
        return
    with database.connect() as connection:
        for user in users.values():
            if not all(user.get(key) for key in ("user_id", "username", "password_salt", "password_hash")):
                continue
            connection.execute(
                """
                INSERT OR IGNORE INTO users
                    (user_id, username, password_salt, password_hash, created_at, disabled, role)
                VALUES (?, ?, ?, ?, ?, 0, 'user')
                """,
                (
                    user["user_id"],
                    user["username"],
                    user["password_salt"],
                    user["password_hash"],
                    float(user.get("created_at") or time.time()),
                ),
            )
        admin_count = connection.execute(
            "SELECT COUNT(*) FROM users WHERE role = 'admin'"
        ).fetchone()[0]
        if admin_count == 0:
            first_user = connection.execute(
                "SELECT user_id FROM users ORDER BY created_at ASC, user_id ASC LIMIT 1"
            ).fetchone()
            if first_user:
                connection.execute(
                    "UPDATE users SET role = 'admin' WHERE user_id = ?",
                    (first_user["user_id"],),
                )
    _migrated_sources.add(source_key)


def _ensure_database():
    database.initialize()
    _migrate_legacy_users()


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
        "role": user.get("role", "user"),
        "disabled": bool(user.get("disabled", False)),
    }


def create_user(username, password):
    username = str(username or "").strip()
    password = str(password or "")
    if len(username) < 3 or len(username) > 40:
        return None, "用户名长度需要在 3 到 40 个字符之间。"
    if len(password) < 8:
        return None, "密码至少需要 8 个字符。"
    with _lock:
        _ensure_database()
        with database.connect() as connection:
            user_count = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        role = "admin" if user_count == 0 else "user"
        secret = _password_hash(password)
        user = {
            "user_id": secrets.token_hex(12),
            "username": username,
            "password_salt": secret["salt"],
            "password_hash": secret["hash"],
            "created_at": time.time(),
            "role": role,
            "disabled": False,
        }
        try:
            with database.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO users
                        (user_id, username, password_salt, password_hash, created_at, disabled, role)
                    VALUES (?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        user["user_id"],
                        user["username"],
                        user["password_salt"],
                        user["password_hash"],
                        user["created_at"],
                        user["role"],
                    ),
                )
        except sqlite3.IntegrityError:
            return None, "用户名已存在。"
        return _public_user(user), None


def verify_user(username, password):
    username = str(username or "").strip()
    password = str(password or "")
    with _lock:
        _ensure_database()
        with database.connect() as connection:
            row = connection.execute(
                """
                SELECT user_id, username, password_salt, password_hash, created_at, role, disabled
                FROM users
                WHERE username = ? COLLATE NOCASE AND disabled = 0
                """,
                (username,),
            ).fetchone()
        user = dict(row) if row else None
    if not user:
        return None
    expected = user.get("password_hash", "")
    actual = _password_hash(password, user.get("password_salt", ""))["hash"]
    if not hmac.compare_digest(expected, actual):
        return None
    return _public_user(user)


def login_retry_after(key):
    """Return remaining lock seconds for a client/user login key."""
    now = time.time()
    with _lock:
        blocked_until = float(_login_blocked_until.get(key, 0))
        if blocked_until <= now:
            _login_blocked_until.pop(key, None)
            return 0
        return max(1, int(blocked_until - now + 0.999))


def record_login_failure(key):
    """Record one failed login and return lock seconds when the limit is hit."""
    now = time.time()
    cutoff = now - LOGIN_WINDOW_SECONDS
    with _lock:
        failures = [stamp for stamp in _login_failures.get(key, []) if stamp >= cutoff]
        failures.append(now)
        _login_failures[key] = failures
        if len(failures) < LOGIN_MAX_FAILURES:
            return 0
        _login_failures.pop(key, None)
        _login_blocked_until[key] = now + LOGIN_LOCK_SECONDS
        return LOGIN_LOCK_SECONDS


def clear_login_failures(key):
    with _lock:
        _login_failures.pop(key, None)
        _login_blocked_until.pop(key, None)


def registration_retry_after(key):
    """Return seconds until this source may attempt another registration."""
    now = time.time()
    cutoff = now - REGISTER_WINDOW_SECONDS
    with _lock:
        attempts = [stamp for stamp in _registration_attempts.get(key, []) if stamp >= cutoff]
        if attempts:
            _registration_attempts[key] = attempts
        else:
            _registration_attempts.pop(key, None)
        if len(attempts) < REGISTER_MAX_ATTEMPTS:
            return 0
        return max(1, int(attempts[0] + REGISTER_WINDOW_SECONDS - now + 0.999))


def record_registration_attempt(key):
    now = time.time()
    cutoff = now - REGISTER_WINDOW_SECONDS
    with _lock:
        attempts = [stamp for stamp in _registration_attempts.get(key, []) if stamp >= cutoff]
        attempts.append(now)
        _registration_attempts[key] = attempts


def create_session(user):
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _lock:
        _ensure_database()
        with database.connect() as connection:
            connection.execute(
                """
                INSERT INTO auth_sessions (token_hash, user_id, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (_token_hash(token), user["user_id"], now, now + SESSION_TTL_SECONDS),
            )
    return token


def get_session_user(token):
    if not token:
        return None
    with _lock:
        _ensure_database()
        now = time.time()
        token_hash = _token_hash(token)
        with database.connect() as connection:
            connection.execute("DELETE FROM auth_sessions WHERE expires_at < ?", (now,))
            row = connection.execute(
                """
                SELECT users.user_id, users.username, users.created_at, users.role, users.disabled
                FROM auth_sessions
                JOIN users ON users.user_id = auth_sessions.user_id
                WHERE auth_sessions.token_hash = ?
                  AND auth_sessions.expires_at >= ?
                  AND users.disabled = 0
                """,
                (token_hash, now),
            ).fetchone()
        if not row:
            return None
        return dict(row)


def delete_session(token):
    if not token:
        return
    with _lock:
        _ensure_database()
        with database.connect() as connection:
            connection.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (_token_hash(token),))


def _token_hash(token):
    return hashlib.sha256(str(token).encode("utf-8")).hexdigest()


def clear_auth_state():
    with _lock:
        database.initialize()
        with database.connect() as connection:
            connection.execute("DELETE FROM auth_sessions")
        _login_failures.clear()
        _login_blocked_until.clear()
        _registration_attempts.clear()


def record_audit_event(action, actor_user_id=None, target_user_id=None, source_ip=None, details=None):
    with _lock:
        database.initialize()
        with database.connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_logs
                    (created_at, actor_user_id, action, target_user_id, source_ip, details_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    actor_user_id,
                    str(action),
                    target_user_id,
                    source_ip,
                    json.dumps(details or {}, ensure_ascii=False),
                ),
            )
            connection.execute(
                """
                DELETE FROM audit_logs
                WHERE audit_id NOT IN (
                    SELECT audit_id FROM audit_logs
                    ORDER BY audit_id DESC
                    LIMIT ?
                )
                """,
                (MAX_AUDIT_LOGS,),
            )


def list_users():
    with _lock:
        _ensure_database()
        with database.connect() as connection:
            rows = connection.execute(
                """
                SELECT users.user_id, users.username, users.created_at, users.role, users.disabled,
                       COUNT(auth_sessions.token_hash) AS active_sessions
                FROM users
                LEFT JOIN auth_sessions
                  ON auth_sessions.user_id = users.user_id
                 AND auth_sessions.expires_at >= ?
                GROUP BY users.user_id
                ORDER BY users.created_at ASC, users.user_id ASC
                """,
                (time.time(),),
            ).fetchall()
        return [
            {**_public_user(dict(row)), "active_sessions": int(row["active_sessions"] or 0)}
            for row in rows
        ]


def update_user_access(user_id, role=None, disabled=None):
    if role is not None and role not in {"admin", "user"}:
        return None, "角色只能是 admin 或 user。"
    with _lock:
        _ensure_database()
        with database.connect() as connection:
            row = connection.execute(
                "SELECT user_id, username, created_at, role, disabled FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if not row:
                return None, "账号不存在。"
            next_role = role if role is not None else row["role"]
            next_disabled = int(bool(disabled)) if disabled is not None else int(row["disabled"])
            if row["role"] == "admin" and (next_role != "admin" or next_disabled):
                other_admins = connection.execute(
                    """
                    SELECT COUNT(*) FROM users
                    WHERE role = 'admin' AND disabled = 0 AND user_id != ?
                    """,
                    (user_id,),
                ).fetchone()[0]
                if other_admins == 0:
                    return None, "系统必须保留至少一个可用管理员。"
            connection.execute(
                "UPDATE users SET role = ?, disabled = ? WHERE user_id = ?",
                (next_role, next_disabled, user_id),
            )
            if next_disabled:
                connection.execute("DELETE FROM auth_sessions WHERE user_id = ?", (user_id,))
            updated = connection.execute(
                "SELECT user_id, username, created_at, role, disabled FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return _public_user(dict(updated)), None


def list_audit_logs(limit=100):
    safe_limit = max(1, min(int(limit or 100), 500))
    with _lock:
        database.initialize()
        with database.connect() as connection:
            rows = connection.execute(
                """
                SELECT audit_logs.audit_id, audit_logs.created_at, audit_logs.action,
                       audit_logs.actor_user_id, audit_logs.target_user_id,
                       audit_logs.source_ip, audit_logs.details_json,
                       actor.username AS actor_username,
                       target.username AS target_username
                FROM audit_logs
                LEFT JOIN users AS actor ON actor.user_id = audit_logs.actor_user_id
                LEFT JOIN users AS target ON target.user_id = audit_logs.target_user_id
                ORDER BY audit_logs.created_at DESC, audit_logs.audit_id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["details"] = json.loads(item.pop("details_json") or "{}")
            except (TypeError, ValueError):
                item["details"] = {}
            result.append(item)
        return result
