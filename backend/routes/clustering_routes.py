"""Clustering training & prediction API routes."""
from flask import Blueprint, request, jsonify
from routes._helpers import coerce_columns_like
from routes._responses import missing_field, missing_fields, session_expired, service_error
from services.clustering_service import train, elbow, predict_one
from session_store import get_session

cluster_bp = Blueprint("clustering", __name__)


def _require(data, *keys):
    for k in keys:
        if k not in data:
            return missing_field(k)
    return None


@cluster_bp.route("/train", methods=["POST"])
def train_model():
    data = request.json or {}
    if err := _require(data, "session_id", "feature_cols", "algorithm", "params"):
        return err
    sid = data["session_id"]
    df = get_session(sid)
    if df is None:
        return session_expired()

    feature_cols = coerce_columns_like(df, data["feature_cols"])

    result, err = train(df, feature_cols, data["algorithm"], data["params"])
    if err:
        return service_error("TRAINING_FAILED", err, 400)
    return jsonify(result)


@cluster_bp.route("/elbow", methods=["POST"])
def elbow_method():
    data = request.json or {}
    if "session_id" not in data or "feature_cols" not in data or "max_k" not in data:
        return missing_fields(["session_id", "feature_cols", "max_k"])
    df = get_session(data["session_id"])
    if df is None:
        return session_expired()

    feature_cols = coerce_columns_like(df, data["feature_cols"])

    result = elbow(df, feature_cols, data["max_k"])
    return jsonify(result)


@cluster_bp.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    if "features" not in data:
        return missing_field("features")
    result, err = predict_one(data["features"])
    if err:
        return service_error("PREDICTION_FAILED", err, 400)
    return jsonify(result)


@cluster_bp.route("/status", methods=["GET"])
def model_status():
    import os, json
    base = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(base, "models", "cluster_config.json")
    model_path = os.path.join(base, "models", "cluster_model.pkl")
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        return jsonify({"has_model": False})
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return jsonify({"has_model": True, "features": config.get("features", []),
                    "algorithm": config.get("algorithm"), "n_clusters": config.get("n_clusters"),
                    "silhouette": config.get("silhouette")})


@cluster_bp.route("/clear", methods=["POST"])
def clear():
    import os
    for f in ["models/cluster_model.pkl", "models/cluster_scaler.pkl", "models/cluster_config.json"]:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f)
        if os.path.exists(path):
            os.remove(path)
    return jsonify({"status": "cleared"})
