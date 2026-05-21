"""Decision tree training & prediction API routes."""
from flask import Blueprint, request, jsonify
from routes._helpers import coerce_column_like, coerce_columns_like
from routes._responses import missing_field, session_expired, service_error
from services.decision_tree_service import train, predict_one
from services.training_validation import (
    validate_classification_training,
    validate_regression_training,
)
from session_store import get_session, get_session_meta
from models.registry import (
    list_versions, get_version, get_active_version, activate_version, delete_version
)

dt_bp = Blueprint("decision_tree", __name__)


def _require(data, *keys):
    for k in keys:
        if k not in data:
            return missing_field(k)
    return None


@dt_bp.route("/train", methods=["POST"])
def train_model():
    data = request.json or {}
    if err := _require(data, "session_id", "target_col", "feature_cols", "task_type", "criterion", "max_depth"):
        return err
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

    result = train(df, target_col, feature_cols,
                   task_type, data["criterion"], data["max_depth"],
                   dataset_name=dataset_name, session_id=sid)
    if isinstance(result, dict) and "error" in result:
        return service_error("TRAINING_FAILED", result["error"], 400)
    return jsonify(result)


@dt_bp.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    if "input_dict" not in data or "task_type" not in data:
        return missing_field("input_dict or task_type")
    result, err = predict_one(data["input_dict"], data["task_type"],
                              version_id=data.get("version_id"))
    if err:
        return service_error("MODEL_NOT_FOUND", err, 404)
    return jsonify(result)


@dt_bp.route("/status", methods=["GET"])
def model_status():
    vid = get_active_version("decision_tree")
    if not vid:
        return jsonify({"has_model": False})
    meta = get_version("decision_tree", vid)
    if not meta:
        return jsonify({"has_model": False})
    return jsonify({"has_model": True, "version_id": vid,
                    "features": meta.get("features", []),
                    "target": meta.get("target", ""),
                    "metrics": meta.get("metrics", {}),
                    "params": meta.get("params", {}),
                    "created_at": meta.get("created_at", ""),
                    "dataset_name": meta.get("dataset_name", "")})


@dt_bp.route("/versions", methods=["GET"])
def list_model_versions():
    versions = list_versions("decision_tree")
    active = get_active_version("decision_tree")
    return jsonify({"versions": versions, "active": active})


@dt_bp.route("/version/<version_id>", methods=["GET"])
def get_version_detail(version_id):
    meta = get_version("decision_tree", version_id)
    if not meta:
        return service_error("VERSION_NOT_FOUND", "Version not found", 404)
    return jsonify({"version_id": version_id, **meta})


@dt_bp.route("/activate", methods=["POST"])
def activate():
    data = request.json or {}
    if "version_id" not in data:
        return missing_field("version_id")
    ok = activate_version("decision_tree", data["version_id"])
    if not ok:
        return service_error("VERSION_NOT_FOUND", "Version not found", 404)
    return jsonify({"status": "activated", "version_id": data["version_id"]})


@dt_bp.route("/version/<version_id>", methods=["DELETE"])
def delete_model_version(version_id):
    ok = delete_version("decision_tree", version_id)
    if not ok:
        return service_error("VERSION_NOT_FOUND", "Version not found", 404)
    return jsonify({"status": "deleted"})


@dt_bp.route("/clear", methods=["POST"])
def clear():
    vid = get_active_version("decision_tree")
    if vid:
        delete_version("decision_tree", vid)
    return jsonify({"status": "cleared"})
