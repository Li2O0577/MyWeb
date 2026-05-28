"""Decision tree training & prediction API routes."""
from flask import Blueprint, request, jsonify
from routes._helpers import coerce_column_like, coerce_columns_like
from routes._responses import missing_field, session_expired, service_error
from routes._versioning import setup_version_routes
from services.decision_tree_service import train, predict_one
from services.training_validation import (
    validate_classification_training,
    validate_regression_training,
)
from session_store import get_session, get_session_meta

dt_bp = Blueprint("decision_tree", __name__)


@dt_bp.route("/train", methods=["POST"])
def train_model():
    data = request.json or {}
    for k in ("session_id", "target_col", "feature_cols", "task_type", "criterion", "max_depth"):
        if k not in data:
            return missing_field(k)
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

    meta = get_session_meta(sid)
    dataset_name = meta.get("source_name", "") if meta else ""

    result, err = train(df, target_col, feature_cols,
                   task_type, data["criterion"], data["max_depth"],
                   dataset_name=dataset_name, session_id=sid)
    if err:
        return service_error("TRAINING_FAILED", err, 400)
    return jsonify(result)


@dt_bp.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    if "input_dict" not in data or "task_type" not in data:
        return missing_field("input_dict or task_type")
    result, err = predict_one(data["input_dict"], data["task_type"],
                              version_id=data.get("version_id"))
    if err:
        code = "MODEL_NOT_FOUND" if "没有找到已保存" in err else "PREDICTION_FAILED"
        status = 404 if code == "MODEL_NOT_FOUND" else 400
        return service_error(code, err, status)
    return jsonify(result)


setup_version_routes(dt_bp, "decision_tree")
