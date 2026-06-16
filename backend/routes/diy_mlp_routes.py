"""DIY MLP training & prediction API routes."""
from flask import Blueprint, jsonify, request

from routes._helpers import (
    coerce_column_like,
    coerce_columns_like,
    numeric_prediction_error,
    prediction_exception_message,
    prediction_service_error,
)
from routes._responses import missing_field, service_error, session_expired
from routes._versioning import setup_version_routes
from services.diy_mlp_service import predict_batch, predict_one, train
from session_store import get_session, get_session_meta

diy_bp = Blueprint("diy_mlp", __name__)
setup_version_routes(diy_bp, "diy_mlp")


@diy_bp.route("/train", methods=["POST"])
def train_route():
    data = request.json or {}
    for field in ("session_id", "target_col", "feature_cols", "task_type", "learning_rate", "epochs", "batch_size"):
        if field not in data:
            return missing_field(field)

    df = get_session(data["session_id"])
    if df is None:
        return session_expired()

    target_col = coerce_column_like(df, data["target_col"])
    feature_cols = coerce_columns_like(df, data["feature_cols"])
    layers = data.get("layers") or data.get("layers_config") or [
        {"neurons": 32, "activation": "ReLU", "bn": False, "dropout": 0.1}
    ]
    try:
        result, err = train(
            df,
            target_col,
            feature_cols,
            layers,
            data["task_type"],
            int(data.get("n_classes", 2)),
            float(data["learning_rate"]),
            data.get("optimizer", "Adam"),
            int(data["epochs"]),
            int(data["batch_size"]),
            float(data.get("val_split", 0.2)),
            int(data.get("patience", 10)),
            data.get("device", "cpu"),
            dataset_name=(get_session_meta(data["session_id"]) or {}).get("source_name", ""),
            session_id=data["session_id"],
        )
    except Exception:
        return service_error("TRAINING_FAILED", "模型训练失败，请检查网络结构、字段类型和训练参数。", 400)
    if err:
        return service_error("TRAINING_FAILED", err, 400)
    return jsonify(result)


@diy_bp.route("/predict", methods=["POST"])
def predict_route():
    data = request.json or {}
    if "features" not in data:
        return missing_field("features")
    if err := numeric_prediction_error(data["features"], "DIY MLP 预测输入"):
        return service_error("PREDICTION_FAILED", err, 400)
    try:
        result, err = predict_one(data["features"], data.get("device", "cpu"), data.get("version_id"))
    except Exception:
        return service_error("PREDICTION_FAILED", prediction_exception_message(), 400)
    if err:
        return prediction_service_error(err)
    return jsonify(result)


@diy_bp.route("/batch_predict", methods=["POST"])
def batch_predict_route():
    data = request.json or {}
    if "rows" not in data:
        return missing_field("rows")
    if err := numeric_prediction_error(data["rows"], "DIY MLP 批量预测输入", batch=True):
        return service_error("PREDICTION_FAILED", err, 400)
    try:
        result, err = predict_batch(data["rows"], data.get("device", "cpu"), data.get("version_id"))
    except Exception:
        return service_error("PREDICTION_FAILED", prediction_exception_message(), 400)
    if err:
        return prediction_service_error(err)
    return jsonify(result)
