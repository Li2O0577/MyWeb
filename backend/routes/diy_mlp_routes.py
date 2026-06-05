"""DIY MLP training & prediction API routes."""
from flask import Blueprint, request, jsonify
from routes._helpers import (
    coerce_column_like, coerce_columns_like, numeric_prediction_error,
    prediction_exception_message, prediction_service_error,
)
from routes._responses import missing_field, session_expired, service_error
from routes._versioning import setup_version_routes
from services.diy_mlp_service import train, predict_one, predict_batch
from services.training_validation import validate_mlp_training
from session_store import get_session, get_session_meta

diy_bp = Blueprint("diy_mlp", __name__)


@diy_bp.route("/train", methods=["POST"])
def train_model():
    data = request.json or {}
    for k in ("session_id", "target_col", "feature_cols", "layers", "task_type",
              "n_classes", "learning_rate", "optimizer", "epochs", "batch_size",
              "val_split", "patience"):
        if k not in data:
            return missing_field(k)
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

    meta = get_session_meta(sid)
    dataset_name = meta.get("source_name", "") if meta else ""

    result, err = train(
        df, target_col, feature_cols, data["layers"],
        data["task_type"], data["n_classes"], data["learning_rate"],
        data["optimizer"], data["epochs"], data["batch_size"],
        data["val_split"], data["patience"], data.get("device", "cpu"),
        dataset_name=dataset_name, session_id=sid
    )
    if err:
        return service_error("TRAINING_FAILED", err, 400)
    return jsonify(result)


@diy_bp.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    if "features" not in data:
        return missing_field("features")
    if err := numeric_prediction_error(data["features"], "DIY MLP 预测输入"):
        return service_error("PREDICTION_FAILED", err, 400)
    try:
        result, err = predict_one(data["features"], data.get("device", "cpu"),
                                  version_id=data.get("version_id"))
    except Exception:
        return service_error("PREDICTION_FAILED", prediction_exception_message(), 400)
    if err:
        return prediction_service_error(err)
    return jsonify(result)


@diy_bp.route("/batch_predict", methods=["POST"])
def batch_predict():
    data = request.json or {}
    if "rows" not in data:
        return missing_field("rows")
    if err := numeric_prediction_error(data["rows"], "DIY MLP 批量预测输入", batch=True):
        return service_error("PREDICTION_FAILED", err, 400)
    try:
        result, err = predict_batch(data["rows"], data.get("device", "cpu"),
                                    version_id=data.get("version_id"))
    except Exception:
        return service_error("PREDICTION_FAILED", prediction_exception_message(), 400)
    if err:
        return prediction_service_error(err)
    return jsonify(result)


setup_version_routes(diy_bp, "diy_mlp")
