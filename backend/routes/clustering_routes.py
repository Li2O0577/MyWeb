"""Clustering training & prediction API routes."""
from flask import Blueprint, request, jsonify
from routes._helpers import coerce_columns_like, numeric_prediction_error, prediction_exception_message
from routes._responses import missing_field, missing_fields, session_expired, service_error
from routes._versioning import setup_version_routes
from services.clustering_service import train, elbow, predict_one
from services.training_validation import validate_feature_training
from session_store import get_session, get_session_meta

cluster_bp = Blueprint("clustering", __name__)


@cluster_bp.route("/train", methods=["POST"])
def train_model():
    data = request.json or {}
    for k in ("session_id", "feature_cols", "algorithm", "params"):
        if k not in data:
            return missing_field(k)
    sid = data["session_id"]
    df = get_session(sid)
    if df is None:
        return session_expired()

    feature_cols = coerce_columns_like(df, data["feature_cols"])
    if err := validate_feature_training(df, feature_cols):
        return service_error("INPUT_VALIDATION_FAILED", err["error"], 400)

    meta = get_session_meta(sid)
    dataset_name = meta.get("source_name", "") if meta else ""

    result, err = train(df, feature_cols, data["algorithm"], data["params"],
                        dataset_name=dataset_name, session_id=sid)
    if err:
        return service_error("TRAINING_FAILED", err, 400)
    return jsonify(result)


@cluster_bp.route("/elbow", methods=["POST"])
def elbow_method():
    data = request.json or {}
    if "session_id" not in data or "feature_cols" not in data or "max_k" not in data:
        return missing_fields(["session_id", "feature_cols", "max_k"])
    df = get_session(data["session_id"])
    if df is None:
        return session_expired()

    feature_cols = coerce_columns_like(df, data["feature_cols"])
    if err := validate_feature_training(df, feature_cols):
        return service_error("INPUT_VALIDATION_FAILED", err["error"], 400)

    result = elbow(df, feature_cols, data["max_k"])
    return jsonify(result)


@cluster_bp.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    if "features" not in data:
        return missing_field("features")
    if err := numeric_prediction_error(data["features"], "聚类预测输入"):
        return service_error("PREDICTION_FAILED", err, 400)
    try:
        result, err = predict_one(data["features"], version_id=data.get("version_id"))
    except Exception:
        return service_error("PREDICTION_FAILED", prediction_exception_message(), 400)
    if err:
        return service_error("PREDICTION_FAILED", err, 400)
    return jsonify(result)


setup_version_routes(cluster_bp, "clustering")
