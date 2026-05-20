"""Regression training & prediction API routes."""
from flask import Blueprint, request, jsonify
from routes._helpers import coerce_column_like, coerce_columns_like
from routes._responses import missing_field, session_expired, service_error
from services.regression_service import train, predict_one, predict_batch
from services.training_validation import validate_regression_training
from session_store import get_session

reg_bp = Blueprint("regression", __name__)


def _require(data, *keys):
    for k in keys:
        if k not in data:
            return missing_field(k)
    return None


@reg_bp.route("/train", methods=["POST"])
def train_model():
    data = request.json or {}
    if err := _require(data, "session_id", "target_col", "feature_cols", "learning_rate", "epochs", "batch_size"):
        return err
    sid = data["session_id"]
    df = get_session(sid)
    if df is None:
        return session_expired()

    target_col = coerce_column_like(df, data["target_col"])
    feature_cols = coerce_columns_like(df, data["feature_cols"])
    if err := validate_regression_training(
        df, target_col, feature_cols, batch_size=data["batch_size"]
    ):
        return service_error("INPUT_VALIDATION_FAILED", err["error"], 400)

    n_samples = len(df)
    n_features = len(feature_cols)

    if n_samples < 500:
        h1, h2, dr = max(8, n_features), max(4, n_features // 2), 0.1
    elif n_samples < 5000:
        h1, h2, dr = n_features * 2, n_features, 0.2
    else:
        h1, h2, dr = n_features * 3, n_features * 2, 0.3

    result = train(df, target_col, feature_cols, h1, h2, dr,
                   data["learning_rate"], data["epochs"], data["batch_size"],
                   data.get("device", "cpu"))
    if "error" in result:
        return service_error("TRAINING_FAILED", result["error"], 400)
    return jsonify(result)


@reg_bp.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    if "features" not in data:
        return missing_field("features")
    result, err = predict_one(data["features"], data.get("device", "cpu"))
    if err:
        return service_error("MODEL_NOT_FOUND", err, 404)
    return jsonify(result)


@reg_bp.route("/batch_predict", methods=["POST"])
def batch_predict():
    data = request.json or {}
    if "rows" not in data:
        return missing_field("rows")
    result, err = predict_batch(data["rows"], data.get("device", "cpu"))
    if err:
        return service_error("MODEL_NOT_FOUND", err, 404)
    return jsonify(result)


@reg_bp.route("/status", methods=["GET"])
def model_status():
    import os, json
    base = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(base, "models", "reg_config.json")
    model_path = os.path.join(base, "models", "reg_best_model.pth")
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        return jsonify({"has_model": False})
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return jsonify({"has_model": True, "features": config.get("features", []),
                    "target": config.get("target", ""), "hidden1": config.get("hidden1"),
                    "hidden2": config.get("hidden2"), "dropout_rate": config.get("dropout_rate"),
                    "r2": config.get("r2"), "mae": config.get("mae"), "rmse": config.get("rmse")})


@reg_bp.route("/clear", methods=["POST"])
def clear():
    import os
    for f in ["models/reg_best_model.pth", "models/reg_scaler.pkl", "models/reg_config.json"]:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f)
        if os.path.exists(path):
            os.remove(path)
    return jsonify({"status": "cleared"})
