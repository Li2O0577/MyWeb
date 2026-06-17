"""Authentication API routes."""
from flask import Blueprint, jsonify, request

from auth_store import COOKIE_NAME, create_session, create_user, delete_session, verify_user
from routes._auth import current_user
from routes._responses import api_error

auth_bp = Blueprint("auth", __name__)


def _credentials():
    data = request.json or {}
    return str(data.get("username") or "").strip(), str(data.get("password") or "")


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
    user, err = create_user(username, password)
    if err:
        return api_error("REGISTER_FAILED", err, 400)
    token = create_session(user)
    return _set_auth_cookie(jsonify({"user": user}), token)


@auth_bp.route("/login", methods=["POST"])
def login():
    username, password = _credentials()
    user = verify_user(username, password)
    if not user:
        return api_error("LOGIN_FAILED", "用户名或密码不正确。", 401)
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
