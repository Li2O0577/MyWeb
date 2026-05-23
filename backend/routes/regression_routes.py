"""Regression training & prediction API routes."""
from flask import Blueprint, request, jsonify
from routes._helpers import coerce_column_like, coerce_columns_like
from routes._responses import missing_field, session_expired, service_error
from routes._versioning import setup_version_routes
from services.regression_service import train, predict_one, predict_batch
from services.training_validation import validate_regression_training
from session_store import get_session, get_session_meta

reg_bp = Blueprint("regression", __name__)


@reg_bp.route("/train", methods=["POST"])
def train_model():
    data = request.json or {}
    for k in ("session_id", "target_col", "feature_cols", "learning_rate", "epochs", "batch_size"):
        if k not in data:
            return missing_field(k)
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

    meta = get_session_meta(sid)
    dataset_name = meta.get("source_name", "") if meta else ""

    result, err = train(df, target_col, feature_cols, h1, h2, dr,
                   data["learning_rate"], data["epochs"], data["batch_size"],
                   data.get("device", "cpu"),
                   dataset_name=dataset_name, session_id=sid)
    if err:
        return service_error("TRAINING_FAILED", err, 400)
    return jsonify(result)


@reg_bp.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    if "features" not in data:
        return missing_field("features")
    result, err = predict_one(data["features"], data.get("device", "cpu"),
                              version_id=data.get("version_id"))
    if err:
        return service_error("MODEL_NOT_FOUND", err, 404)
    return jsonify(result)


@reg_bp.route("/batch_predict", methods=["POST"])
def batch_predict():
    data = request.json or {}
    if "rows" not in data:
        return missing_field("rows")
    result, err = predict_batch(data["rows"], data.get("device", "cpu"),
                                version_id=data.get("version_id"))
    if err:
        return service_error("MODEL_NOT_FOUND", err, 404)
    return jsonify(result)


setup_version_routes(reg_bp, "regression")
