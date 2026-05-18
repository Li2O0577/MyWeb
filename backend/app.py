"""Flask backend for ML training & inference."""
import uuid
import time
import threading
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── In-memory session store ──
_sessions = {}
_sessions_lock = threading.Lock()
SESSION_TTL = 3600  # 1 hour


def create_session(df):
    sid = str(uuid.uuid4())[:8]
    with _sessions_lock:
        _sessions[sid] = {"df": df, "at": time.time()}
    return sid


def get_session(sid):
    with _sessions_lock:
        s = _sessions.get(sid)
        if s and time.time() - s["at"] < SESSION_TTL:
            s["at"] = time.time()
            return s["df"]
        if s:
            del _sessions[sid]
    return None


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
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(port=5000, debug=True)
