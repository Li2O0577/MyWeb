"""Classification training & prediction API routes."""
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
from services.classification_service import predict_batch, predict_one, train
from session_store import get_session, get_session_meta

cls_bp = Blueprint("classification", __name__)
setup_version_routes(cls_bp, "classification")


def _classification_validation_error(df, target_col):
    clean = df[[target_col]].dropna()
    n_rows = len(clean)
    n_classes = clean[target_col].nunique()
    if n_rows and n_classes >= max(5, int(n_rows * 0.8)):
        return f"分类目标类别数量 {n_classes} 过于接近样本数 {n_rows}，更像 ID 或连续标签，请更换目标列。"
    return None


@cls_bp.route("/train", methods=["POST"])
def train_route():
    data = request.json or {}
    for field in ("session_id", "target_col", "feature_cols", "learning_rate", "epochs", "batch_size"):
        if field not in data:
            return missing_field(field)

    df = get_session(data["session_id"])
    if df is None:
        return session_expired()

    target_col = coerce_column_like(df, data["target_col"])
    feature_cols = coerce_columns_like(df, data["feature_cols"])
    if err := _classification_validation_error(df, target_col):
        return service_error("INPUT_VALIDATION_FAILED", err, 400)

    try:
        result, err = train(
            df,
            target_col,
            feature_cols,
            int(data.get("hidden1", 64)),
            int(data.get("hidden2", 32)),
            float(data.get("dropout_rate", 0.2)),
            float(data["learning_rate"]),
            int(data["epochs"]),
            int(data["batch_size"]),
            data.get("device", "cpu"),
            dataset_name=(get_session_meta(data["session_id"]) or {}).get("source_name", ""),
            session_id=data["session_id"],
        )
    except Exception:
        return service_error("TRAINING_FAILED", "模型训练失败，请检查字段类型和训练参数。", 400)
    if err:
        return service_error("TRAINING_FAILED", err, 400)
    return jsonify(result)


@cls_bp.route("/predict", methods=["POST"])
def predict_route():
    data = request.json or {}
    if "features" not in data:
        return missing_field("features")
    if err := numeric_prediction_error(data["features"], "分类预测输入"):
        return service_error("PREDICTION_FAILED", err, 400)
    try:
        result, err = predict_one(data["features"], data.get("device", "cpu"), data.get("version_id"))
    except Exception:
        return service_error("PREDICTION_FAILED", prediction_exception_message(), 400)
    if err:
        return prediction_service_error(err)
    return jsonify(result)


@cls_bp.route("/batch_predict", methods=["POST"])
def batch_predict_route():
    data = request.json or {}
    if "rows" not in data:
        return missing_field("rows")
    if err := numeric_prediction_error(data["rows"], "分类批量预测输入", batch=True):
        return service_error("PREDICTION_FAILED", err, 400)
    try:
        result, err = predict_batch(data["rows"], data.get("device", "cpu"), data.get("version_id"))
    except Exception:
        return service_error("PREDICTION_FAILED", prediction_exception_message(), 400)
    if err:
        return prediction_service_error(err)
    return jsonify(result)
