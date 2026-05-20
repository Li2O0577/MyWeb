"""DIY MLP training & prediction API routes."""
from flask import Blueprint, request, jsonify
from routes._helpers import coerce_column_like, coerce_columns_like
from routes._responses import missing_field, session_expired, service_error
from services.diy_mlp_service import train, predict_one, predict_batch
from services.training_validation import validate_mlp_training
from session_store import get_session

diy_bp = Blueprint("diy_mlp", __name__)


def _require(data, *keys):
    for k in keys:
        if k not in data:
            return missing_field(k)
    return None


@diy_bp.route("/train", methods=["POST"])
def train_model():
    data = request.json or {}
    if err := _require(data, "session_id", "target_col", "feature_cols", "layers", "task_type", "n_classes", "learning_rate", "optimizer", "epochs", "batch_size", "val_split", "patience"):
        return err
    sid = data["session_id"]
    df = get_session(sid)
    if df is None:
        return session_expired()

    target_col = coerce_column_like(df, data["target_col"])
    feature_cols = coerce_columns_like(df, data["feature_cols"])
    if err := validate_mlp_training(
        df, target_col, feature_cols,
        task_type=data["task_type"],
        batch_size=data["batch_size"],
        val_split=data["val_split"],
        n_classes=data["n_classes"],
    ):
        return service_error("INPUT_VALIDATION_FAILED", err["error"], 400)

    result = train(
        df, target_col, feature_cols, data["layers"],
        data["task_type"], data["n_classes"], data["learning_rate"],
        data["optimizer"], data["epochs"], data["batch_size"],
        data["val_split"], data["patience"], data.get("device", "cpu")
    )
    if "error" in result:
        return service_error("TRAINING_FAILED", result["error"], 400)
    return jsonify(result)


@diy_bp.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    if "features" not in data:
        return missing_field("features")
    result, err = predict_one(data["features"], data.get("device", "cpu"))
    if err:
        return service_error("MODEL_NOT_FOUND", err, 404)
    return jsonify(result)


@diy_bp.route("/batch_predict", methods=["POST"])
def batch_predict():
    data = request.json or {}
    if "rows" not in data:
        return missing_field("rows")
    result, err = predict_batch(data["rows"], data.get("device", "cpu"))
    if err:
        return service_error("MODEL_NOT_FOUND", err, 404)
    return jsonify(result)


@diy_bp.route("/status", methods=["GET"])
def model_status():
    import os, json
    base = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(base, "models", "diy_config.json")
    model_path = os.path.join(base, "models", "diy_best_model.pth")
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        return jsonify({"has_model": False})
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return jsonify({"has_model": True, "features": config.get("features", []),
                    "target": config.get("target", ""), "task": config.get("task"),
                    "n_classes": config.get("n_classes"), "layers": config.get("layers"),
                    "reverse_label_map": config.get("reverse_label_map", {})})


@diy_bp.route("/clear", methods=["POST"])
def clear():
    import os
    for f in ["models/diy_best_model.pth", "models/diy_scaler.pkl", "models/diy_config.json"]:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f)
        if os.path.exists(path):
            os.remove(path)
    return jsonify({"status": "cleared"})
