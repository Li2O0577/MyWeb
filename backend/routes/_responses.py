"""Consistent API response helpers."""
from flask import jsonify


def api_error(code, message, status=400, detail=None):
    payload = {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "detail": detail or "",
        },
    }
    return jsonify(payload), status


def missing_field(field):
    return api_error(
        "MISSING_FIELD",
        f"缺少必填参数：{field}",
        400,
        detail=f"请求中没有提供必填参数 `{field}`。",
    )


def missing_fields(fields):
    fields_text = ", ".join(fields)
    return api_error(
        "MISSING_FIELD",
        f"缺少必填参数：{fields_text}",
        400,
        detail=f"请求中没有提供这些必填参数：{fields_text}。",
    )


def session_expired():
    return api_error(
        "SESSION_EXPIRED",
        "当前数据会话已失效",
        404,
        detail="请重新上传数据，或点击页面中的“重新同步当前数据到后端”后再试。",
    )


def service_error(code, message, status=400):
    return api_error(code, message, status, detail=message)


def error_event(code, message, detail=None):
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "detail": detail or "",
        },
    }
