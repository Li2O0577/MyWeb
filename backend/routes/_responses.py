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
        f"Missing required field: {field}",
        400,
        detail=f"Required request field `{field}` was not provided.",
    )


def missing_fields(fields):
    fields_text = ", ".join(fields)
    return api_error(
        "MISSING_FIELD",
        f"Missing required fields: {fields_text}",
        400,
        detail=f"Required request fields were not provided: {fields_text}.",
    )


def session_expired():
    return api_error(
        "SESSION_EXPIRED",
        "Session not found or expired",
        404,
        detail="Upload or sync the current dataset again before running this operation.",
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
