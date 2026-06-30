"""Administrator-only account management and audit APIs."""
from flask import Blueprint, jsonify, request

from auth_store import list_audit_logs, list_users, record_audit_event, update_user_access
from routes._auth import current_user, require_admin
from routes._responses import api_error


admin_bp = Blueprint("admin", __name__)


@admin_bp.route("/users", methods=["GET"])
@require_admin
def users_index():
    return jsonify({"users": list_users()})


@admin_bp.route("/users/<user_id>", methods=["PATCH"])
@require_admin
def users_update(user_id):
    actor = current_user()
    data = request.json or {}
    if user_id == actor["user_id"] and (
        data.get("disabled") is True or data.get("role") == "user"
    ):
        return api_error("SELF_ACCESS_CHANGE_DENIED", "不能禁用或降级当前登录的管理员账号。", 400)
    role = data.get("role") if "role" in data else None
    disabled = data.get("disabled") if "disabled" in data else None
    if disabled is not None and not isinstance(disabled, bool):
        return api_error("INVALID_FIELD", "disabled 必须是布尔值。", 400)
    user, error = update_user_access(user_id, role=role, disabled=disabled)
    if error:
        status = 404 if error == "账号不存在。" else 400
        return api_error("ACCOUNT_UPDATE_FAILED", error, status)
    record_audit_event(
        "admin.user_access_updated",
        actor_user_id=actor["user_id"],
        target_user_id=user_id,
        source_ip=request.remote_addr,
        details={"role": user["role"], "disabled": user["disabled"]},
    )
    return jsonify({"user": user})


@admin_bp.route("/audit-logs", methods=["GET"])
@require_admin
def audit_logs_index():
    try:
        limit = int(request.args.get("limit", 100))
    except (TypeError, ValueError):
        return api_error("INVALID_FIELD", "limit 必须是整数。", 400)
    return jsonify({"audit_logs": list_audit_logs(limit=limit)})
