"""Shared JSON error responses for API routes."""
from flask import jsonify


def api_error(code, message, status=400, detail=None):
    payload = {"error": {"code": code, "message": message}}
    if detail:
        payload["error"]["detail"] = detail
    return jsonify(payload), status


def missing_field(field):
    return api_error("MISSING_FIELD", f"缺少必要字段：{field}", 400)


def service_error(code, message, status=400):
    return api_error(code, message, status)


def session_expired():
    return api_error("SESSION_EXPIRED", "数据 session 不存在或已过期，请重新上传数据。", 404)
