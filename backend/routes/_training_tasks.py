"""Helpers for model training routes that can run sync or background tasks."""
from flask import jsonify

from routes._responses import service_error
from task_store import fail_task, finish_task, submit_task


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
        submit_task(task_id, guarded_training)
        return jsonify({"task_id": task_id, "status": "running", "async": True}), 202

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
