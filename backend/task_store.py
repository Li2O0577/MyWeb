"""Queued background runner with SQLite-backed task status history."""
import json
import os
import threading
import time
import uuid

import database
from resource_limits import MAX_PENDING_TASKS_PER_USER

_tasks = {}
_lock = threading.RLock()
_queue = []
_running = set()
_cancel_requested = set()

GLOBAL_CONCURRENCY = int(os.environ.get("MYWEB1_GLOBAL_TASK_CONCURRENCY", "2"))
USER_CONCURRENCY = int(os.environ.get("MYWEB1_USER_TASK_CONCURRENCY", "1"))


def _now():
    return time.time()


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _json_dump(value):
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _json_load(value, fallback=None):
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _persist_task_locked(task):
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO tasks (
                task_id, kind, label, status, created_at, updated_at,
                finished_at, duration_sec, metadata_json, result_json,
                result_payload_json, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                kind = excluded.kind,
                label = excluded.label,
                status = excluded.status,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                finished_at = excluded.finished_at,
                duration_sec = excluded.duration_sec,
                metadata_json = excluded.metadata_json,
                result_json = excluded.result_json,
                result_payload_json = excluded.result_payload_json,
                error = excluded.error
            """,
            (
                task["task_id"],
                task["kind"],
                task["label"],
                task["status"],
                task["created_at"],
                task["updated_at"],
                task.get("finished_at"),
                task.get("duration_sec"),
                _json_dump(task.get("metadata") or {}),
                _json_dump(task.get("result")),
                _json_dump(task.get("result_payload")),
                task.get("error"),
            ),
        )


def restore_tasks():
    """Restore task history and settle work that a process restart interrupted."""
    with _lock:
        database.initialize()
        with database.connect() as connection:
            rows = connection.execute(
                """
                SELECT task_id, kind, label, status, created_at, updated_at,
                       finished_at, duration_sec, metadata_json, result_json,
                       result_payload_json, error
                FROM tasks
                ORDER BY created_at ASC
                """
            ).fetchall()
        _tasks.clear()
        _queue.clear()
        _running.clear()
        _cancel_requested.clear()
        interrupted = {"queued", "running", "cancelling"}
        now = _now()
        for row in rows:
            task = dict(row)
            task["metadata"] = _json_load(task.pop("metadata_json"), {})
            task["result"] = _json_load(task.pop("result_json"))
            task["result_payload"] = _json_load(task.pop("result_payload_json"))
            if task["status"] in interrupted:
                task["status"] = "failed"
                task["updated_at"] = now
                task["finished_at"] = now
                task["duration_sec"] = round(now - float(task["created_at"]), 3)
                task["result_payload"] = None
                task["error"] = "任务因服务重启而中断，请重新提交。"
                _persist_task_locked(task)
            _tasks[task["task_id"]] = task
        return len(_tasks)


def _public(task, include_payload=False):
    public = dict(task)
    if not include_payload:
        public.pop("result_payload", None)
    return public


def _compact_result(result):
    if not isinstance(result, dict):
        return None
    summary = {}
    for key in ("version_id", "task_type", "algorithm", "n_found", "target", "features"):
        if key in result:
            summary[key] = result[key]
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        summary["metrics"] = metrics
    else:
        metric_keys = ("r2", "mae", "rmse", "acc", "accuracy", "silhouette")
        picked = {key: result[key] for key in metric_keys if key in result}
        if picked:
            summary["metrics"] = picked
    return summary or None


def _user_id(task):
    return str((task.get("metadata") or {}).get("user_id") or "")


def _running_count_for_user(user_id):
    return sum(1 for task_id in _running if _user_id(_tasks.get(task_id) or {}) == str(user_id or ""))


def _pending_count_for_user(user_id, exclude_task_id=None):
    pending = {"queued", "running", "cancelling"}
    return sum(
        1
        for task_id, task in _tasks.items()
        if task_id != exclude_task_id
        and _user_id(task) == str(user_id or "")
        and task.get("status") in pending
    )


def _reject_task_locked(task, message):
    now = _now()
    task["status"] = "failed"
    task["updated_at"] = now
    task["finished_at"] = now
    task["duration_sec"] = round(now - float(task["created_at"]), 3)
    task["error"] = message
    _persist_task_locked(task)


def create_task(kind, label, metadata=None, status="running"):
    task_id = uuid.uuid4().hex
    now = _now()
    with _lock:
        task = {
            "task_id": task_id,
            "kind": kind,
            "label": label,
            "status": status,
            "created_at": now,
            "updated_at": now,
            "finished_at": None,
            "duration_sec": None,
            "metadata": metadata or {},
            "result": None,
            "result_payload": None,
            "error": None,
        }
        _tasks[task_id] = task
        _persist_task_locked(task)
    return task_id


def finish_task(task_id, result=None):
    with _lock:
        task = _tasks.get(task_id)
        if not task:
            return None
        now = _now()
        cancelled = task_id in _cancel_requested
        task["status"] = "succeeded"
        if cancelled:
            task["status"] = "cancelled"
        task["updated_at"] = now
        task["finished_at"] = now
        task["duration_sec"] = round(now - float(task["created_at"]), 3)
        task["result"] = None if cancelled else _compact_result(result)
        task["result_payload"] = None if cancelled else (result if isinstance(result, dict) else None)
        task["error"] = "任务已取消。" if cancelled else None
        _cancel_requested.discard(task_id)
        _running.discard(task_id)
        _persist_task_locked(task)
        _drain_queue_locked()
        return _public(task)


def fail_task(task_id, error):
    with _lock:
        task = _tasks.get(task_id)
        if not task:
            return None
        now = _now()
        task["status"] = "failed"
        task["updated_at"] = now
        task["finished_at"] = now
        task["duration_sec"] = round(now - float(task["created_at"]), 3)
        task["error"] = str(error or "任务失败")
        task["result_payload"] = None
        _cancel_requested.discard(task_id)
        _running.discard(task_id)
        _persist_task_locked(task)
        _drain_queue_locked()
        return _public(task)


def submit_task(task_id, work):
    """Queue work() in the background and settle the task with its result."""
    def runner():
        with _lock:
            task = _tasks.get(task_id)
            if not task or task_id in _cancel_requested:
                if task:
                    _settle_cancelled_locked(task_id)
                return None
            task["status"] = "running"
            task["updated_at"] = _now()
            _persist_task_locked(task)
        try:
            result = work()
        except Exception as exc:
            fail_task(task_id, exc)
            return None
        finish_task(task_id, result)
        return result

    with _lock:
        task = _tasks.get(task_id)
        if not task:
            return None
        user_id = _user_id(task)
        if user_id and _pending_count_for_user(user_id, exclude_task_id=task_id) >= MAX_PENDING_TASKS_PER_USER:
            _reject_task_locked(task, f"当前账号最多保留 {MAX_PENDING_TASKS_PER_USER} 个等待或运行中的任务。")
            return None
        task["status"] = "queued"
        task["updated_at"] = _now()
        _persist_task_locked(task)
        _queue.append((task_id, runner))
        _drain_queue_locked()
    return task_id


def start_inline_task(task_id):
    """Reserve queue capacity for a synchronous compatibility request."""
    with _lock:
        task = _tasks.get(task_id)
        if not task:
            return False
        user_id = _user_id(task)
        if user_id and _pending_count_for_user(user_id, exclude_task_id=task_id) >= MAX_PENDING_TASKS_PER_USER:
            _reject_task_locked(task, f"当前账号最多保留 {MAX_PENDING_TASKS_PER_USER} 个等待或运行中的任务。")
            return False
        if len(_running) >= max(1, GLOBAL_CONCURRENCY):
            _reject_task_locked(task, "服务器训练并发已满，请稍后重试或使用异步队列。")
            return False
        if user_id and _running_count_for_user(user_id) >= max(1, USER_CONCURRENCY):
            _reject_task_locked(task, "当前账号已有训练任务运行中，请稍后重试或使用异步队列。")
            return False
        _running.add(task_id)
        task["status"] = "running"
        task["updated_at"] = _now()
        _persist_task_locked(task)
        return True


def _settle_cancelled_locked(task_id):
    task = _tasks.get(task_id)
    if not task:
        return None
    now = _now()
    task["status"] = "cancelled"
    task["updated_at"] = now
    task["finished_at"] = now
    task["duration_sec"] = round(now - float(task["created_at"]), 3)
    task["result"] = None
    task["result_payload"] = None
    task["error"] = "任务已取消。"
    _cancel_requested.discard(task_id)
    _running.discard(task_id)
    _persist_task_locked(task)
    return _public(task)


def _can_start_locked(task_id):
    if len(_running) >= max(1, GLOBAL_CONCURRENCY):
        return False
    task = _tasks.get(task_id)
    if not task:
        return False
    user_id = _user_id(task)
    if user_id and _running_count_for_user(user_id) >= max(1, USER_CONCURRENCY):
        return False
    return True


def _drain_queue_locked():
    started = []
    remaining = []
    for task_id, runner in _queue:
        if task_id in _cancel_requested:
            _settle_cancelled_locked(task_id)
            continue
        if _can_start_locked(task_id):
            _running.add(task_id)
            started.append((task_id, runner))
        else:
            remaining.append((task_id, runner))
    _queue[:] = remaining
    for _task_id, runner in started:
        thread = threading.Thread(target=runner, name=f"myweb1-task-{_task_id[:8]}", daemon=True)
        thread.start()


def cancel_task(task_id, user_id=None):
    with _lock:
        task = _tasks.get(task_id)
        if not task:
            return None
        if user_id is not None and _user_id(task) != str(user_id):
            return None
        if task["status"] in {"succeeded", "failed", "cancelled"}:
            return _public(task)
        _cancel_requested.add(task_id)
        if task["status"] == "queued":
            _queue[:] = [(tid, runner) for tid, runner in _queue if tid != task_id]
            result = _settle_cancelled_locked(task_id)
            _drain_queue_locked()
            return result
        task["status"] = "cancelling"
        task["updated_at"] = _now()
        _persist_task_locked(task)
        return _public(task)


def get_task(task_id, include_payload=False, user_id=None):
    with _lock:
        task = _tasks.get(task_id)
        if task and user_id is not None and _user_id(task) != str(user_id):
            return None
        return _public(task, include_payload=include_payload) if task else None


def list_tasks(limit=20, kind=None, user_id=None):
    with _lock:
        items = list(_tasks.values())
        if kind:
            items = [task for task in items if task.get("kind") == kind]
        if user_id is not None:
            items = [task for task in items if _user_id(task) == str(user_id)]
        items.sort(key=lambda task: task.get("created_at", 0), reverse=True)
        return [_public(task) for task in items[: max(1, min(int(limit or 20), 100))]]


def pending_task_count(user_id=None):
    with _lock:
        if user_id is None:
            return sum(1 for task in _tasks.values() if task.get("status") in {"queued", "running", "cancelling"})
        return _pending_count_for_user(user_id)


def clear_tasks():
    with _lock:
        _tasks.clear()
        _queue.clear()
        _running.clear()
        _cancel_requested.clear()
        database.initialize()
        with database.connect() as connection:
            connection.execute("DELETE FROM tasks")


restore_tasks()
