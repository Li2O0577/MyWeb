"""DIY MLP training & prediction API routes."""
from flask import Blueprint, request, jsonify
from routes._helpers import coerce_column_like, coerce_columns_like
from routes._responses import missing_field, session_expired, service_error
from services.diy_mlp_service import train, predict_one, predict_batch
from services.training_validation import validate_mlp_training
from session_store import get_session, get_session_meta
from models.registry import (
    list_versions, get_version, get_active_version, activate_version, delete_version
)

diy_bp = Blueprint("diy_mlp", __name__)


def _require(data, *keys):
    for k in keys:
        if k not in data:
            return missing_field(k)
    return None


@diy_bp.route("/train", methods=["POST"])
def train_model():
    data = request.json or {}
    if err := _require(data, "session_id", "target_col", "feature_cols", "layers", "task_type", "n_classes", "learning_rate", "optimizer", "epochs", "batch_size", "val_split", "patience"):
        return err
    sid = data["session_id"]
    df = get_session(sid)
    if df is None:
        return session_expired()

    target_col = coerce_column_like(df, data["target_col"])
    feature_cols = coerce_columns_like(df, data["feature_cols"])
    if err := validate_mlp_training(
        df, target_col, feature_cols,
        task_type=data["task_type"],
        batch_size=data["batch_size"],
        val_split=data["val_split"],
        n_classes=data["n_classes"],
    ):
        return service_error("INPUT_VALIDATION_FAILED", err["error"], 400)

    meta = get_session_meta(sid)
    dataset_name = meta.get("source_name", "") if meta else ""

    result = train(
        df, target_col, feature_cols, data["layers"],
        data["task_type"], data["n_classes"], data["learning_rate"],
        data["optimizer"], data["epochs"], data["batch_size"],
        data["val_split"], data["patience"], data.get("device", "cpu"),
        dataset_name=dataset_name, session_id=sid
    )
    if "error" in result:
        return service_error("TRAINING_FAILED", result["error"], 400)
    return jsonify(result)


@diy_bp.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    if "features" not in data:
        return missing_field("features")
    result, err = predict_one(data["features"], data.get("device", "cpu"),
                              version_id=data.get("version_id"))
    if err:
        return service_error("MODEL_NOT_FOUND", err, 404)
    return jsonify(result)


@diy_bp.route("/batch_predict", methods=["POST"])
def batch_predict():
    data = request.json or {}
    if "rows" not in data:
        return missing_field("rows")
    result, err = predict_batch(data["rows"], data.get("device", "cpu"),
                                version_id=data.get("version_id"))
    if err:
        return service_error("MODEL_NOT_FOUND", err, 404)
    return jsonify(result)


@diy_bp.route("/status", methods=["GET"])
def model_status():
    vid = get_active_version("diy_mlp")
    if not vid:
        return jsonify({"has_model": False})
    meta = get_version("diy_mlp", vid)
    if not meta:
        return jsonify({"has_model": False})
    return jsonify({"has_model": True, "version_id": vid,
                    "features": meta.get("features", []),
                    "target": meta.get("target", ""),
                    "metrics": meta.get("metrics", {}),
                    "params": meta.get("params", {}),
                    "created_at": meta.get("created_at", ""),
                    "dataset_name": meta.get("dataset_name", "")})


@diy_bp.route("/versions", methods=["GET"])
def list_model_versions():
    versions = list_versions("diy_mlp")
    active = get_active_version("diy_mlp")
    return jsonify({"versions": versions, "active": active})


@diy_bp.route("/version/<version_id>", methods=["GET"])
def get_version_detail(version_id):
    meta = get_version("diy_mlp", version_id)
    if not meta:
        return service_error("VERSION_NOT_FOUND", "Version not found", 404)
    return jsonify({"version_id": version_id, **meta})


@diy_bp.route("/activate", methods=["POST"])
def activate():
    data = request.json or {}
    if "version_id" not in data:
        return missing_field("version_id")
    ok = activate_version("diy_mlp", data["version_id"])
    if not ok:
        return service_error("VERSION_NOT_FOUND", "Version not found", 404)
    return jsonify({"status": "activated", "version_id": data["version_id"]})


@diy_bp.route("/version/<version_id>", methods=["DELETE"])
def delete_model_version(version_id):
    ok = delete_version("diy_mlp", version_id)
    if not ok:
        return service_error("VERSION_NOT_FOUND", "Version not found", 404)
    return jsonify({"status": "deleted"})


@diy_bp.route("/clear", methods=["POST"])
def clear():
    vid = get_active_version("diy_mlp")
    if vid:
        delete_version("diy_mlp", vid)
    return jsonify({"status": "cleared"})
