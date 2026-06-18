"""Flask backend for ML training & inference."""
import os
import threading
from flask import Flask, request, send_from_directory
from flask_cors import CORS
from routes._auth import current_user
from routes._responses import api_error
from resource_limits import (
    MAX_DATASET_COLUMNS,
    MAX_DATASET_ROWS,
    MAX_PENDING_TASKS_PER_USER,
    MAX_SESSIONS_PER_USER,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_MB,
)
from session_store import active_session_count, cleanup_expired, recent_sessions, restore_sessions

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def _cors_origins():
    raw = os.environ.get(
        "CORS_ORIGINS",
        "http://127.0.0.1:5173,http://localhost:5173",
    )
    if raw.strip() == "*":
        return "*"
    return [item.strip() for item in raw.split(",") if item.strip()]


CORS(app, resources={r"/api/*": {"origins": _cors_origins()}}, supports_credentials=True)

FRONTEND_DIST = os.path.abspath(
    os.environ.get(
        "FRONTEND_DIST",
        os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"),
    )
)

restore_sessions()

# Clean up registry entries whose model files have been manually deleted
from models.registry import cleanup_orphaned_versions
cleanup_orphaned_versions()


# ── Periodic cleanup ──
def _schedule_cleanup(interval=300):
    """Run cleanup on a recurring timer. Each cycle is exception-isolated so a
    single failure cannot kill the entire cleanup chain."""
    try:
        cleanup_expired()
    except Exception:
        pass
    t = threading.Timer(interval, _schedule_cleanup, args=[interval])
    t.daemon = True
    t.start()


_schedule_cleanup()


# ── Register blueprints ──
from routes.auth_routes import auth_bp
from routes.data_routes import data_bp
from routes.regression_routes import reg_bp
from routes.classification_routes import cls_bp
from routes.diy_mlp_routes import diy_bp
from routes.decision_tree_routes import dt_bp
from routes.clustering_routes import cluster_bp
from routes.llm_routes import llm_bp
from routes.task_routes import task_bp

app.register_blueprint(auth_bp, url_prefix="/api/auth")
app.register_blueprint(data_bp, url_prefix="/api/data")
app.register_blueprint(reg_bp, url_prefix="/api/regression")
app.register_blueprint(cls_bp, url_prefix="/api/classification")
app.register_blueprint(diy_bp, url_prefix="/api/diy_mlp")
app.register_blueprint(dt_bp, url_prefix="/api/decision_tree")
app.register_blueprint(cluster_bp, url_prefix="/api/clustering")
app.register_blueprint(llm_bp, url_prefix="/api/llm")
app.register_blueprint(task_bp, url_prefix="/api/tasks")


@app.before_request
def require_api_login():
    if not request.path.startswith("/api/"):
        return None
    if request.path.startswith("/api/auth/"):
        return None
    if current_user():
        return None
    return api_error("UNAUTHENTICATED", "请先登录。", 401)


@app.route("/api/health")
def health():
    try:
        user = current_user()
        user_id = user.get("user_id") if user else None
        n_sessions = active_session_count(user_id=user_id)
        from task_store import pending_task_count
        n_pending_tasks = pending_task_count(user_id=user_id)
        from models.registry import get_active_version, get_registry
        reg = get_registry()
        model_files = {}
        for mt in ["regression", "classification", "diy_mlp", "decision_tree", "clustering"]:
            active = get_active_version(mt, user_id=user_id)
            versions = [
                item
                for item in reg.get(mt, {}).get("versions", {}).values()
                if str(item.get("user_id") or "") == str(user_id)
            ]
            n_versions = len(versions)
            model_files[mt] = active is not None
            model_files[f"{mt}_versions"] = n_versions
        return {
            "status": "ok",
            "active_sessions": n_sessions,
            "resource_usage": {
                "active_sessions": n_sessions,
                "pending_tasks": n_pending_tasks,
            },
            "resource_limits": {
                "max_sessions_per_user": MAX_SESSIONS_PER_USER,
                "max_pending_tasks_per_user": MAX_PENDING_TASKS_PER_USER,
                "max_upload_mb": MAX_UPLOAD_MB,
                "max_dataset_rows": MAX_DATASET_ROWS,
                "max_dataset_columns": MAX_DATASET_COLUMNS,
            },
            "recent_sessions": recent_sessions(user_id=user_id),
            "saved_models": model_files,
            "user": user,
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "active_sessions": 0,
            "resource_usage": {},
            "resource_limits": {},
            "recent_sessions": [],
            "saved_models": {},
        }


def _frontend_available():
    return os.path.exists(os.path.join(FRONTEND_DIST, "index.html"))


@app.route("/")
def serve_frontend_index():
    if not _frontend_available():
        return api_error(
            "FRONTEND_NOT_BUILT",
            "React 前端尚未构建",
            404,
            detail="请先运行 frontend/npm run build，或在开发模式使用 Vite dev server。",
        )
    return send_from_directory(FRONTEND_DIST, "index.html")


@app.route("/<path:path>")
def serve_frontend_asset(path):
    if path.startswith("api/"):
        return api_error("NOT_FOUND", "没有找到对应的后端接口", 404)
    if not _frontend_available():
        return api_error(
            "FRONTEND_NOT_BUILT",
            "React 前端尚未构建",
            404,
            detail="请先运行 frontend/npm run build，或在开发模式使用 Vite dev server。",
        )
    asset_path = os.path.join(FRONTEND_DIST, path)
    if os.path.isfile(asset_path):
        return send_from_directory(FRONTEND_DIST, path)
    return send_from_directory(FRONTEND_DIST, "index.html")


@app.errorhandler(404)
def not_found(_err):
    return api_error("NOT_FOUND", "没有找到对应的后端接口", 404)


@app.errorhandler(413)
def too_large(_err):
    return api_error(
        "PAYLOAD_TOO_LARGE",
        "上传文件超过 256 MB 限制",
        413,
        detail="请先压缩、拆分文件，或减少数据量后再上传。",
    )


@app.errorhandler(500)
def internal_error(err):
    detail = str(err) if os.environ.get("FLASK_DEBUG", "0") == "1" else "请查看后端终端日志获取详细原因。"
    return api_error("INTERNAL_ERROR", "后端处理时出现内部错误", 500, detail)


if __name__ == "__main__":
    # debug=True but use_reloader=False: keep debug info + interactive
    # debugger but avoid the watchdog reloader bug on Windows (duplicate
    # listener processes that accept TCP but never respond)
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5001"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug, use_reloader=False)
