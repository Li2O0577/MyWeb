"""Authentication API routes."""
from flask import Blueprint, jsonify, request

from auth_store import (
    COOKIE_NAME,
    clear_login_failures,
    create_session,
    create_user,
    delete_session,
    login_retry_after,
    record_registration_attempt,
    record_login_failure,
    registration_retry_after,
    verify_user,
)
from routes._auth import current_user
from routes._responses import api_error

auth_bp = Blueprint("auth", __name__)


def _credentials():
    data = request.json or {}
    return str(data.get("username") or "").strip(), str(data.get("password") or "")


def _login_key(username):
    return f"{request.remote_addr or 'unknown'}:{str(username or '').strip().lower()}"


def _rate_limit_response(retry_after):
    response, status = api_error(
        "LOGIN_RATE_LIMITED",
        f"登录失败次数过多，请在 {retry_after} 秒后重试。",
        429,
    )
    response.headers["Retry-After"] = str(retry_after)
    return response, status


def _registration_rate_limit_response(retry_after):
    response, status = api_error(
        "REGISTER_RATE_LIMITED",
        f"注册请求过于频繁，请在 {retry_after} 秒后重试。",
        429,
    )
    response.headers["Retry-After"] = str(retry_after)
    return response, status


def _set_auth_cookie(response, token):
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        samesite="Lax",
        secure=False,
        max_age=7 * 24 * 60 * 60,
    )
    return response


@auth_bp.route("/register", methods=["POST"])
def register():
    username, password = _credentials()
    source_key = request.remote_addr or "unknown"
    retry_after = registration_retry_after(source_key)
    if retry_after:
        return _registration_rate_limit_response(retry_after)
    record_registration_attempt(source_key)
    user, err = create_user(username, password)
    if err:
        return api_error("REGISTER_FAILED", err, 400)
    token = create_session(user)
    return _set_auth_cookie(jsonify({"user": user}), token)


@auth_bp.route("/login", methods=["POST"])
def login():
    username, password = _credentials()
    login_key = _login_key(username)
    retry_after = login_retry_after(login_key)
    if retry_after:
        return _rate_limit_response(retry_after)
    user = verify_user(username, password)
    if not user:
        retry_after = record_login_failure(login_key)
        if retry_after:
            return _rate_limit_response(retry_after)
        return api_error("LOGIN_FAILED", "用户名或密码不正确。", 401)
    clear_login_failures(login_key)
    token = create_session(user)
    return _set_auth_cookie(jsonify({"user": user}), token)


@auth_bp.route("/logout", methods=["POST"])
def logout():
    delete_session(request.cookies.get(COOKIE_NAME))
    response = jsonify({"status": "logged_out"})
    response.delete_cookie(COOKIE_NAME)
    return response


@auth_bp.route("/me", methods=["GET"])
def me():
    user = current_user()
    if not user:
        return api_error("UNAUTHENTICATED", "请先登录。", 401)
    return jsonify({"user": user})
