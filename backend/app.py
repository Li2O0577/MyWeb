"""Flask backend for ML training & inference."""
import os
import threading
from flask import Flask
from flask_cors import CORS
from routes._responses import api_error
from session_store import active_session_count, cleanup_expired, recent_sessions, restore_sessions

app = Flask(__name__)
CORS(app)

restore_sessions()


# ── Periodic cleanup ──
def _cleanup_expired():
    cleanup_expired()


def _start_cleanup_timer():
    _cleanup_expired()
    t = threading.Timer(300, _start_cleanup_timer)
    t.daemon = True
    t.start()


_start_cleanup_timer()


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
    # Check which models exist
    import os as _os
    models_dir = _os.path.join(_os.path.dirname(__file__), "models")
    model_files = {
        "regression": _os.path.exists(_os.path.join(models_dir, "reg_best_model.pth")),
        "classification": _os.path.exists(_os.path.join(models_dir, "cls_best_model.pth")),
        "diy_mlp": _os.path.exists(_os.path.join(models_dir, "diy_best_model.pth")),
        "decision_tree": _os.path.exists(_os.path.join(models_dir, "dt_model.pkl")),
        "clustering": _os.path.exists(_os.path.join(models_dir, "cluster_model.pkl")),
    }
    return {
        "status": "ok",
        "active_sessions": n_sessions,
        "recent_sessions": recent_sessions(),
        "saved_models": model_files,
    }


@app.errorhandler(404)
def not_found(_err):
    return api_error("NOT_FOUND", "API endpoint not found", 404)


@app.errorhandler(500)
def internal_error(err):
    return api_error("INTERNAL_ERROR", "Internal server error", 500, str(err))


if __name__ == "__main__":
    # debug=True but use_reloader=False: keep debug info + interactive
    # debugger but avoid the watchdog reloader bug on Windows (duplicate
    # listener processes that accept TCP but never respond)
    port = int(os.environ.get("FLASK_PORT", "5001"))
    app.run(port=port, debug=True, use_reloader=False)
