"""DIY MLP training & prediction API routes."""
from flask import Blueprint, request, jsonify
from services.diy_mlp_service import train, predict_one, predict_batch
from app import get_session

diy_bp = Blueprint("diy_mlp", __name__)


def _require(data, *keys):
    for k in keys:
        if k not in data:
            return jsonify({"error": f"Missing required field: {k}"}), 400
    return None


@diy_bp.route("/train", methods=["POST"])
def train_model():
    data = request.json or {}
    if err := _require(data, "session_id", "target_col", "feature_cols", "layers", "task_type", "n_classes", "learning_rate", "optimizer", "epochs", "batch_size", "val_split", "patience"):
        return err
    sid = data["session_id"]
    df = get_session(sid)
    if df is None:
        return jsonify({"error": "Session not found or expired"}), 404

    result = train(
        df, data["target_col"], data["feature_cols"], data["layers"],
        data["task_type"], data["n_classes"], data["learning_rate"],
        data["optimizer"], data["epochs"], data["batch_size"],
        data["val_split"], data["patience"], data.get("device", "cpu")
    )
    return jsonify(result)


@diy_bp.route("/predict", methods=["POST"])
def predict():
    data = request.json
    result, err = predict_one(data["features"], data.get("device", "cpu"))
    if err:
        return jsonify({"error": err}), 404
    return jsonify(result)


@diy_bp.route("/batch_predict", methods=["POST"])
def batch_predict():
    data = request.json
    result, err = predict_batch(data["rows"], data.get("device", "cpu"))
    if err:
        return jsonify({"error": err}), 404
    return jsonify(result)


@diy_bp.route("/clear", methods=["POST"])
def clear():
    import os
    for f in ["models/diy_best_model.pth", "models/diy_scaler.pkl", "models/diy_config.json"]:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f)
        if os.path.exists(path):
            os.remove(path)
    return jsonify({"status": "cleared"})
