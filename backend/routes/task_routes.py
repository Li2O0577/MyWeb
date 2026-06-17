"""Task status API routes."""
from flask import Blueprint, jsonify

from routes._auth import current_user_id
from routes._responses import api_error
from task_store import cancel_task, get_task, list_tasks

task_bp = Blueprint("tasks", __name__)


@task_bp.route("", methods=["GET"])
def tasks():
    return jsonify({"tasks": list_tasks(limit=30, user_id=current_user_id())})


@task_bp.route("/<task_id>", methods=["GET"])
def task_detail(task_id):
    task = get_task(task_id, include_payload=True, user_id=current_user_id())
    if task is None:
        return api_error("TASK_NOT_FOUND", "没有找到对应任务", 404)
    return jsonify(task)


@task_bp.route("/<task_id>/cancel", methods=["POST"])
def cancel(task_id):
    task = cancel_task(task_id, user_id=current_user_id())
    if task is None:
        return api_error("TASK_NOT_FOUND", "没有找到对应任务", 404)
    return jsonify(task)
