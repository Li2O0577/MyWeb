"""Helpers for model training routes that can run sync or background tasks."""
from flask import jsonify

from routes._responses import service_error
from task_store import fail_task, finish_task, start_inline_task, submit_task


def wants_async_training(request, data):
    """Return whether the caller requested background training."""
    return request.args.get("async") == "1" or data.get("async") is True


def run_or_submit_training(task_id, run_training, run_async, generic_error):
    """Run a training callable now, or submit it as a background task."""
    def guarded_training():
        try:
            return run_training()
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(generic_error) from exc

    if run_async:
        if not submit_task(task_id, guarded_training):
            return service_error(
                "TASK_QUOTA_EXCEEDED",
                "当前账号等待或运行中的训练任务过多，请等待任务完成或取消旧任务后再试。",
                429,
            )
        return jsonify({"task_id": task_id, "status": "queued", "async": True}), 202

    if not start_inline_task(task_id):
        return service_error(
            "TASK_CAPACITY_EXCEEDED",
            "当前训练并发或账号任务配额已满，请稍后重试。",
            429,
        )

    try:
        result = guarded_training()
    except ValueError as exc:
        fail_task(task_id, exc)
        return service_error("TRAINING_FAILED", str(exc), 400)
    except RuntimeError as exc:
        fail_task(task_id, exc)
        return service_error("TRAINING_FAILED", str(exc), 400)
    finish_task(task_id, result)
    return jsonify(result)
