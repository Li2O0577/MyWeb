"""Clustering training & prediction API routes."""
from flask import Blueprint, request, jsonify
from services.clustering_service import train, elbow, predict_one
from app import get_session

cluster_bp = Blueprint("clustering", __name__)


def _require(data, *keys):
    for k in keys:
        if k not in data:
            return jsonify({"error": f"Missing required field: {k}"}), 400
    return None


@cluster_bp.route("/train", methods=["POST"])
def train_model():
    data = request.json or {}
    if err := _require(data, "session_id", "feature_cols", "algorithm", "params"):
        return err
    sid = data["session_id"]
    df = get_session(sid)
    if df is None:
        return jsonify({"error": "Session not found or expired"}), 404

    result, err = train(df, data["feature_cols"], data["algorithm"], data["params"])
    if err:
        return jsonify({"error": err}), 400
    return jsonify(result)


@cluster_bp.route("/elbow", methods=["POST"])
def elbow_method():
    data = request.json
    sid = data.get("session_id")
    df = get_session(sid)
    if df is None:
        return jsonify({"error": "Session not found or expired"}), 404

    result = elbow(df, data["feature_cols"], data["max_k"])
    return jsonify(result)


@cluster_bp.route("/predict", methods=["POST"])
def predict():
    data = request.json
    result, err = predict_one(data["features"])
    if err:
        return jsonify({"error": err}), 400
    return jsonify(result)


@cluster_bp.route("/clear", methods=["POST"])
def clear():
    import os
    for f in ["models/cluster_model.pkl", "models/cluster_scaler.pkl", "models/cluster_config.json"]:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f)
        if os.path.exists(path):
            os.remove(path)
    return jsonify({"status": "cleared"})
