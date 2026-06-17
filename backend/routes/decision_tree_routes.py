"""Decision tree training & prediction API routes."""
from flask import Blueprint, jsonify, request

from routes._auth import current_user_id
from routes._helpers import coerce_column_like, coerce_columns_like, prediction_exception_message
from routes._responses import missing_field, service_error, session_expired
from routes._training_tasks import run_or_submit_training, wants_async_training
from routes._versioning import setup_version_routes
from services.decision_tree_service import predict_batch, predict_one, train
from session_store import get_session, get_session_meta
from task_store import create_task

dt_bp = Blueprint("decision_tree", __name__)
setup_version_routes(dt_bp, "decision_tree")


@dt_bp.route("/train", methods=["POST"])
def train_route():
    data = request.json or {}
    for field in ("session_id", "target_col", "feature_cols", "task_type"):
        if field not in data:
            return missing_field(field)

    user_id = current_user_id()
    df = get_session(data["session_id"], user_id=user_id)
    if df is None:
        return session_expired()

    task_type = data["task_type"]
    criterion = data.get("criterion") or ("gini" if task_type == "classification" else "squared_error")
    max_depth = data.get("max_depth")
    max_depth = None if max_depth in ("", None) else int(max_depth)
    target_col = coerce_column_like(df, data["target_col"])
    feature_cols = coerce_columns_like(df, data["feature_cols"])

    meta = get_session_meta(data["session_id"], user_id=user_id) or {}
    task_id = create_task("training", "决策树训练", {
        "model_type": "decision_tree",
        "session_id": data["session_id"],
        "dataset_name": meta.get("source_name", ""),
        "user_id": user_id,
        "target_col": str(target_col),
        "feature_count": len(feature_cols),
        "task_type": task_type,
    })
    df_snapshot = df.copy(deep=True)

    def run_training():
        result, err = train(
            df_snapshot,
            target_col,
            feature_cols,
            task_type,
            criterion,
            max_depth,
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
        "决策树训练失败，请检查字段类型和训练参数。",
    )


@dt_bp.route("/predict", methods=["POST"])
def predict_route():
    data = request.json or {}
    for field in ("input_dict", "task_type"):
        if field not in data:
            return missing_field(field)
    try:
        result, err = predict_one(data["input_dict"], data["task_type"], data.get("version_id"), user_id=current_user_id())
    except Exception:
        return service_error("PREDICTION_FAILED", prediction_exception_message(), 400)
    if err:
        code, message = err if isinstance(err, tuple) else ("PREDICTION_FAILED", err)
        return service_error(code, message, 404 if code == "MODEL_NOT_FOUND" else 400)
    return jsonify(result)


@dt_bp.route("/batch_predict", methods=["POST"])
def batch_predict_route():
    data = request.json or {}
    for field in ("rows", "task_type"):
        if field not in data:
            return missing_field(field)
    try:
        result, err = predict_batch(data["rows"], data["task_type"], data.get("version_id"), user_id=current_user_id())
    except Exception:
        return service_error("PREDICTION_FAILED", prediction_exception_message(), 400)
    if err:
        code, message = err if isinstance(err, tuple) else ("PREDICTION_FAILED", err)
        return service_error(code, message, 404 if code == "MODEL_NOT_FOUND" else 400)
    return jsonify(result)
