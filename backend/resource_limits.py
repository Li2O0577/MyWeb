"""Environment-backed resource limits for the local multi-user server."""
import os


def _env_int(name, default, minimum=1):
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


MAX_UPLOAD_MB = _env_int("MYWEB1_MAX_UPLOAD_MB", 256)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_DATASET_ROWS = _env_int("MYWEB1_MAX_DATASET_ROWS", 500_000)
MAX_DATASET_COLUMNS = _env_int("MYWEB1_MAX_DATASET_COLUMNS", 1_000)
MAX_SESSIONS_PER_USER = _env_int("MYWEB1_MAX_SESSIONS_PER_USER", 10)
MAX_PENDING_TASKS_PER_USER = _env_int("MYWEB1_MAX_PENDING_TASKS_PER_USER", 5)

LOGIN_MAX_FAILURES = _env_int("MYWEB1_LOGIN_MAX_FAILURES", 5)
LOGIN_WINDOW_SECONDS = _env_int("MYWEB1_LOGIN_WINDOW_SECONDS", 300)
LOGIN_LOCK_SECONDS = _env_int("MYWEB1_LOGIN_LOCK_SECONDS", 900)
REGISTER_MAX_ATTEMPTS = _env_int("MYWEB1_REGISTER_MAX_ATTEMPTS", 5)
REGISTER_WINDOW_SECONDS = _env_int("MYWEB1_REGISTER_WINDOW_SECONDS", 3_600)
MAX_AUDIT_LOGS = _env_int("MYWEB1_MAX_AUDIT_LOGS", 10_000, minimum=100)
