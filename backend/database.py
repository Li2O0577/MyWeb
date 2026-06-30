"""SQLite storage shared by authentication and background task state."""
import os
import sqlite3
import threading


PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_configured_path = os.environ.get("MYWEB1_DB_PATH", os.path.join("backend", "runtime", "myweb1.db"))
DB_PATH = _configured_path if os.path.isabs(_configured_path) else os.path.join(PROJECT_DIR, _configured_path)

_init_lock = threading.RLock()
_initialized_paths = set()


def _normalized_path():
    return os.path.abspath(DB_PATH)


def _open_connection():
    path = _normalized_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    connection = sqlite3.connect(path, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize():
    """Create the current database schema once for each configured path."""
    path = _normalized_path()
    with _init_lock:
        if path in _initialized_paths and os.path.exists(path):
            return path
        with _open_connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    password_salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    disabled INTEGER NOT NULL DEFAULT 0,
                    role TEXT NOT NULL DEFAULT 'user'
                );

                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_auth_sessions_user
                    ON auth_sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_auth_sessions_expiry
                    ON auth_sessions(expires_at);

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    label TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    finished_at REAL,
                    duration_sec REAL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT,
                    result_payload_json TEXT,
                    error TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_tasks_created_at
                    ON tasks(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_tasks_status
                    ON tasks(status);

                CREATE TABLE IF NOT EXISTS audit_logs (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    actor_user_id TEXT,
                    action TEXT NOT NULL,
                    target_user_id TEXT,
                    source_ip TEXT,
                    details_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at
                    ON audit_logs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_audit_logs_actor
                    ON audit_logs(actor_user_id);
                """
            )
            user_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(users)").fetchall()
            }
            if "role" not in user_columns:
                connection.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
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
        _initialized_paths.add(path)
    return path


def connect():
    initialize()
    return _open_connection()
