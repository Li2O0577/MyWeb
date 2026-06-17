"""Route helpers for account authentication and ownership checks."""
from functools import wraps

from flask import request

from auth_store import COOKIE_NAME, get_session_user
from routes._responses import api_error


def current_user():
    return get_session_user(request.cookies.get(COOKIE_NAME))


def current_user_id():
    user = current_user()
    return user.get("user_id") if user else None


def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_user():
            return api_error("UNAUTHENTICATED", "请先登录。", 401)
        return fn(*args, **kwargs)

    return wrapper
