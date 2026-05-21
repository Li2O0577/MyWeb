"""Regression training & prediction API routes."""
from flask import Blueprint, request, jsonify
from routes._helpers import coerce_column_like, coerce_columns_like
from routes._responses import missing_field, session_expired, service_error
from services.regression_service import train, predict_one, predict_batch
from services.training_validation import validate_regression_training
from session_store import get_session, get_session_meta
from models.registry import (
    list_versions, get_version, get_active_version, activate_version, delete_version
)

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

    meta = get_session_meta(sid)
    dataset_name = meta.get("source_name", "") if meta else ""

    result = train(df, target_col, feature_cols, h1, h2, dr,
                   data["learning_rate"], data["epochs"], data["batch_size"],
                   data.get("device", "cpu"),
                   dataset_name=dataset_name, session_id=sid)
    if "error" in result:
        return service_error("TRAINING_FAILED", result["error"], 400)
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


@reg_bp.route("/status", methods=["GET"])
def model_status():
    vid = get_active_version("regression")
    if not vid:
        return jsonify({"has_model": False})
    meta = get_version("regression", vid)
    if not meta:
        return jsonify({"has_model": False})
    return jsonify({"has_model": True, "version_id": vid,
                    "features": meta.get("features", []),
                    "target": meta.get("target", ""),
                    "metrics": meta.get("metrics", {}),
                    "params": meta.get("params", {}),
                    "created_at": meta.get("created_at", ""),
                    "dataset_name": meta.get("dataset_name", "")})


@reg_bp.route("/versions", methods=["GET"])
def list_model_versions():
    versions = list_versions("regression")
    active = get_active_version("regression")
    return jsonify({"versions": versions, "active": active})


@reg_bp.route("/version/<version_id>", methods=["GET"])
def get_version_detail(version_id):
    meta = get_version("regression", version_id)
    if not meta:
        return service_error("VERSION_NOT_FOUND", "Version not found", 404)
    return jsonify({"version_id": version_id, **meta})


@reg_bp.route("/activate", methods=["POST"])
def activate():
    data = request.json or {}
    if "version_id" not in data:
        return missing_field("version_id")
    ok = activate_version("regression", data["version_id"])
    if not ok:
        return service_error("VERSION_NOT_FOUND", "Version not found", 404)
    return jsonify({"status": "activated", "version_id": data["version_id"]})


@reg_bp.route("/version/<version_id>", methods=["DELETE"])
def delete_model_version(version_id):
    ok = delete_version("regression", version_id)
    if not ok:
        return service_error("VERSION_NOT_FOUND", "Version not found", 404)
    return jsonify({"status": "deleted"})


@reg_bp.route("/clear", methods=["POST"])
def clear():
    vid = get_active_version("regression")
    if vid:
        delete_version("regression", vid)
    return jsonify({"status": "cleared"})
