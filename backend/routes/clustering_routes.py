"""Clustering training & prediction API routes."""
from flask import Blueprint, request, jsonify
from routes._helpers import coerce_columns_like
from routes._responses import missing_field, missing_fields, session_expired, service_error
from services.clustering_service import train, elbow, predict_one
from services.training_validation import validate_feature_training
from session_store import get_session, get_session_meta
from models.registry import (
    list_versions, get_version, get_active_version, activate_version, delete_version
)

cluster_bp = Blueprint("clustering", __name__)


def _require(data, *keys):
    for k in keys:
        if k not in data:
            return missing_field(k)
    return None


@cluster_bp.route("/train", methods=["POST"])
def train_model():
    data = request.json or {}
    if err := _require(data, "session_id", "feature_cols", "algorithm", "params"):
        return err
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
    result, err = predict_one(data["features"], version_id=data.get("version_id"))
    if err:
        return service_error("PREDICTION_FAILED", err, 400)
    return jsonify(result)


@cluster_bp.route("/status", methods=["GET"])
def model_status():
    vid = get_active_version("clustering")
    if not vid:
        return jsonify({"has_model": False})
    meta = get_version("clustering", vid)
    if not meta:
        return jsonify({"has_model": False})
    return jsonify({"has_model": True, "version_id": vid,
                    "features": meta.get("features", []),
                    "target": meta.get("target", ""),
                    "metrics": meta.get("metrics", {}),
                    "params": meta.get("params", {}),
                    "created_at": meta.get("created_at", ""),
                    "dataset_name": meta.get("dataset_name", "")})


@cluster_bp.route("/versions", methods=["GET"])
def list_model_versions():
    versions = list_versions("clustering")
    active = get_active_version("clustering")
    return jsonify({"versions": versions, "active": active})


@cluster_bp.route("/version/<version_id>", methods=["GET"])
def get_version_detail(version_id):
    meta = get_version("clustering", version_id)
    if not meta:
        return service_error("VERSION_NOT_FOUND", "Version not found", 404)
    return jsonify({"version_id": version_id, **meta})


@cluster_bp.route("/activate", methods=["POST"])
def activate():
    data = request.json or {}
    if "version_id" not in data:
        return missing_field("version_id")
    ok = activate_version("clustering", data["version_id"])
    if not ok:
        return service_error("VERSION_NOT_FOUND", "Version not found", 404)
    return jsonify({"status": "activated", "version_id": data["version_id"]})


@cluster_bp.route("/version/<version_id>", methods=["DELETE"])
def delete_model_version(version_id):
    ok = delete_version("clustering", version_id)
    if not ok:
        return service_error("VERSION_NOT_FOUND", "Version not found", 404)
    return jsonify({"status": "deleted"})


@cluster_bp.route("/clear", methods=["POST"])
def clear():
    vid = get_active_version("clustering")
    if vid:
        delete_version("clustering", vid)
    return jsonify({"status": "cleared"})
