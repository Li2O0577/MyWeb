"""Regression training & prediction API routes."""
from flask import Blueprint, jsonify, request

from routes._auth import current_user_id
from routes._helpers import (
    coerce_column_like,
    coerce_columns_like,
    numeric_prediction_error,
    prediction_exception_message,
    prediction_service_error,
)
from routes._responses import missing_field, service_error, session_expired
from routes._training_tasks import run_or_submit_training, wants_async_training
from routes._versioning import setup_version_routes
from services.regression_service import predict_batch, predict_one, train
from session_store import get_session, get_session_meta
from task_store import create_task

reg_bp = Blueprint("regression", __name__)
setup_version_routes(reg_bp, "regression")


@reg_bp.route("/train", methods=["POST"])
def train_route():
    data = request.json or {}
    for field in ("session_id", "target_col", "feature_cols", "learning_rate", "epochs", "batch_size"):
        if field not in data:
            return missing_field(field)

    user_id = current_user_id()
    df = get_session(data["session_id"], user_id=user_id)
    if df is None:
        return session_expired()

    target_col = coerce_column_like(df, data["target_col"])
    feature_cols = coerce_columns_like(df, data["feature_cols"])
    meta = get_session_meta(data["session_id"], user_id=user_id) or {}
    task_id = create_task("training", "回归训练", {
        "model_type": "regression",
        "session_id": data["session_id"],
        "dataset_name": meta.get("source_name", ""),
        "user_id": user_id,
        "target_col": str(target_col),
        "feature_count": len(feature_cols),
    })
    df_snapshot = df.copy(deep=True)

    def run_training():
        result, err = train(
            df_snapshot,
            target_col,
            feature_cols,
            int(data.get("hidden1", 64)),
            int(data.get("hidden2", 32)),
            float(data.get("dropout_rate", 0.2)),
            float(data["learning_rate"]),
            int(data["epochs"]),
            int(data["batch_size"]),
            data.get("device", "cpu"),
            dataset_name=meta.get("source_name", ""),
            session_id=data["session_id"],
            user_id=user_id,
        )
        if err:
            raise ValueError(err)
        result["task_id"] = task_id
        return result

    return run_or_submit_training(
        task_id,
        run_training,
        wants_async_training(request, data),
        "模型训练失败，请检查字段类型和训练参数。",
    )


@reg_bp.route("/predict", methods=["POST"])
def predict_route():
    data = request.json or {}
    if "features" not in data:
        return missing_field("features")
    if err := numeric_prediction_error(data["features"], "回归预测输入"):
        return service_error("PREDICTION_FAILED", err, 400)
    try:
        result, err = predict_one(data["features"], data.get("device", "cpu"), data.get("version_id"), user_id=current_user_id())
    except Exception:
        return service_error("PREDICTION_FAILED", prediction_exception_message(), 400)
    if err:
        return prediction_service_error(err)
    return jsonify(result)


@reg_bp.route("/batch_predict", methods=["POST"])
def batch_predict_route():
    data = request.json or {}
    if "rows" not in data:
        return missing_field("rows")
    if err := numeric_prediction_error(data["rows"], "回归批量预测输入", batch=True):
        return service_error("PREDICTION_FAILED", err, 400)
    try:
        result, err = predict_batch(data["rows"], data.get("device", "cpu"), data.get("version_id"), user_id=current_user_id())
    except Exception:
        return service_error("PREDICTION_FAILED", prediction_exception_message(), 400)
    if err:
        return prediction_service_error(err)
    return jsonify(result)
