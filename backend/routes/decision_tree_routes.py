"""Decision tree training & prediction API routes."""
from flask import Blueprint, request, jsonify
from services.decision_tree_service import train, predict_one
from app import get_session

dt_bp = Blueprint("decision_tree", __name__)


def _require(data, *keys):
    for k in keys:
        if k not in data:
            return jsonify({"error": f"Missing required field: {k}"}), 400
    return None


@dt_bp.route("/train", methods=["POST"])
def train_model():
    data = request.json or {}
    if err := _require(data, "session_id", "target_col", "feature_cols", "task_type", "criterion", "max_depth"):
        return err
    sid = data["session_id"]
    df = get_session(sid)
    if df is None:
        return jsonify({"error": "Session not found or expired"}), 404

    result = train(df, data["target_col"], data["feature_cols"],
                   data["task_type"], data["criterion"], data["max_depth"])
    return jsonify(result)


@dt_bp.route("/predict", methods=["POST"])
def predict():
    data = request.json
    result, err = predict_one(data["input_dict"], data["task_type"])
    if err:
        return jsonify({"error": err}), 404
    return jsonify(result)


@dt_bp.route("/clear", methods=["POST"])
def clear():
    import os
    for f in ["models/dt_model.pkl", "models/dt_config.json"]:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f)
        if os.path.exists(path):
            os.remove(path)
    return jsonify({"status": "cleared"})
