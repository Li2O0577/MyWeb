"""Flask backend for ML training & inference."""
import os
import threading
from flask import Flask
from flask_cors import CORS
from routes._responses import api_error
from session_store import active_session_count, cleanup_expired, recent_sessions, restore_sessions

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024  # 256 MB per request
CORS(app)

restore_sessions()


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
from routes.data_routes import data_bp
from routes.regression_routes import reg_bp
from routes.classification_routes import cls_bp
from routes.diy_mlp_routes import diy_bp
from routes.decision_tree_routes import dt_bp
from routes.clustering_routes import cluster_bp
from routes.llm_routes import llm_bp

app.register_blueprint(data_bp, url_prefix="/api/data")
app.register_blueprint(reg_bp, url_prefix="/api/regression")
app.register_blueprint(cls_bp, url_prefix="/api/classification")
app.register_blueprint(diy_bp, url_prefix="/api/diy_mlp")
app.register_blueprint(dt_bp, url_prefix="/api/decision_tree")
app.register_blueprint(cluster_bp, url_prefix="/api/clustering")
app.register_blueprint(llm_bp, url_prefix="/api/llm")


@app.route("/api/health")
def health():
    n_sessions = active_session_count()
    from models.registry import get_active_version, get_registry
    reg = get_registry()
    model_files = {}
    for mt in ["regression", "classification", "diy_mlp", "decision_tree", "clustering"]:
        active = reg.get(mt, {}).get("active")
        n_versions = len(reg.get(mt, {}).get("versions", {}))
        model_files[mt] = active is not None
        model_files[f"{mt}_versions"] = n_versions
    return {
        "status": "ok",
        "active_sessions": n_sessions,
        "recent_sessions": recent_sessions(),
        "saved_models": model_files,
    }


@app.errorhandler(404)
def not_found(_err):
    return api_error("NOT_FOUND", "API endpoint not found", 404)


@app.errorhandler(413)
def too_large(_err):
    return api_error(
        "PAYLOAD_TOO_LARGE",
        "Uploaded file exceeds the 256 MB size limit",
        413,
        detail="Please split the file or reduce its size before uploading.",
    )


@app.errorhandler(500)
def internal_error(err):
    return api_error("INTERNAL_ERROR", "Internal server error", 500, str(err))


if __name__ == "__main__":
    # debug=True but use_reloader=False: keep debug info + interactive
    # debugger but avoid the watchdog reloader bug on Windows (duplicate
    # listener processes that accept TCP but never respond)
    port = int(os.environ.get("FLASK_PORT", "5001"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(port=port, debug=debug, use_reloader=False)
