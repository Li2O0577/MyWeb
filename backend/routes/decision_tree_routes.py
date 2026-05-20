"""Decision tree training & prediction API routes."""
from flask import Blueprint, request, jsonify
from routes._helpers import coerce_column_like, coerce_columns_like
from routes._responses import missing_field, session_expired, service_error
from services.decision_tree_service import train, predict_one
from services.training_validation import (
    validate_classification_training,
    validate_regression_training,
)
from session_store import get_session

dt_bp = Blueprint("decision_tree", __name__)


def _require(data, *keys):
    for k in keys:
        if k not in data:
            return missing_field(k)
    return None


@dt_bp.route("/train", methods=["POST"])
def train_model():
    data = request.json or {}
    if err := _require(data, "session_id", "target_col", "feature_cols", "task_type", "criterion", "max_depth"):
        return err
    sid = data["session_id"]
    df = get_session(sid)
    if df is None:
        return session_expired()

    target_col = coerce_column_like(df, data["target_col"])
    feature_cols = coerce_columns_like(df, data["feature_cols"])
    task_type = data["task_type"]
    if task_type == "classification":
        err = validate_classification_training(
            df, target_col, feature_cols, require_numeric_features=False, min_clean_rows=5
        )
    else:
        err = validate_regression_training(
            df, target_col, feature_cols, require_numeric_features=False, min_clean_rows=5
        )
    if err:
        return service_error("INPUT_VALIDATION_FAILED", err["error"], 400)

    result = train(df, target_col, feature_cols,
                   task_type, data["criterion"], data["max_depth"])
    if isinstance(result, dict) and "error" in result:
        return service_error("TRAINING_FAILED", result["error"], 400)
    return jsonify(result)


@dt_bp.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    if "input_dict" not in data or "task_type" not in data:
        return missing_field("input_dict or task_type")
    result, err = predict_one(data["input_dict"], data["task_type"])
    if err:
        return service_error("MODEL_NOT_FOUND", err, 404)
    return jsonify(result)


@dt_bp.route("/status", methods=["GET"])
def model_status():
    import os, json
    base = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(base, "models", "dt_config.json")
    model_path = os.path.join(base, "models", "dt_model.pkl")
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        return jsonify({"has_model": False})
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return jsonify({"has_model": True, "features": config.get("features", []),
                    "target": config.get("target", ""), "task_type": config.get("task_type"),
                    "criterion": config.get("criterion"), "max_depth": config.get("max_depth")})


@dt_bp.route("/clear", methods=["POST"])
def clear():
    import os
    for f in ["models/dt_model.pkl", "models/dt_config.json"]:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f)
        if os.path.exists(path):
            os.remove(path)
    return jsonify({"status": "cleared"})
