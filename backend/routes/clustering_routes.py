"""Clustering training & prediction API routes."""
from flask import Blueprint, jsonify, request

from routes._helpers import coerce_columns_like, numeric_prediction_error, prediction_exception_message
from routes._responses import missing_field, service_error, session_expired
from routes._versioning import setup_version_routes
from services.clustering_service import elbow, predict_batch, predict_one, train
from session_store import get_session, get_session_meta

cluster_bp = Blueprint("clustering", __name__)
setup_version_routes(cluster_bp, "clustering")


@cluster_bp.route("/train", methods=["POST"])
def train_route():
    data = request.json or {}
    for field in ("session_id", "feature_cols", "algorithm"):
        if field not in data:
            return missing_field(field)

    df = get_session(data["session_id"])
    if df is None:
        return session_expired()

    algorithm = data["algorithm"]
    params = data.get("params") or {}
    if algorithm == "kmeans":
        params.setdefault("n_clusters", int(data.get("n_clusters", 3)))
    elif algorithm == "dbscan":
        params.setdefault("eps", float(data.get("eps", 0.5)))
        params.setdefault("min_samples", int(data.get("min_samples", 5)))
    else:
        return service_error("INVALID_ALGORITHM", "聚类算法只支持 kmeans 或 dbscan。", 400)

    try:
        result, err = train(
            df,
            coerce_columns_like(df, data["feature_cols"]),
            algorithm,
            params,
            dataset_name=(get_session_meta(data["session_id"]) or {}).get("source_name", ""),
            session_id=data["session_id"],
        )
    except Exception:
        return service_error("TRAINING_FAILED", "聚类训练失败，请检查字段类型和训练参数。", 400)
    if err:
        return service_error("TRAINING_FAILED", err, 400)
    return jsonify(result)


@cluster_bp.route("/elbow", methods=["POST"])
def elbow_route():
    data = request.json or {}
    for field in ("session_id", "feature_cols"):
        if field not in data:
            return missing_field(field)
    df = get_session(data["session_id"])
    if df is None:
        return session_expired()
    result = elbow(df, coerce_columns_like(df, data["feature_cols"]), int(data.get("max_k", 10)))
    return jsonify(result)


@cluster_bp.route("/predict", methods=["POST"])
def predict_route():
    data = request.json or {}
    if "features" not in data:
        return missing_field("features")
    if err := numeric_prediction_error(data["features"], "聚类预测输入"):
        return service_error("PREDICTION_FAILED", err, 400)
    try:
        result, err = predict_one(data["features"], data.get("version_id"))
    except Exception:
        return service_error("PREDICTION_FAILED", prediction_exception_message(), 400)
    if err:
        code = "MODEL_NOT_FOUND" if "没有找到已保存" in str(err) else "PREDICTION_FAILED"
        return service_error(code, err, 404 if code == "MODEL_NOT_FOUND" else 400)
    return jsonify(result)


@cluster_bp.route("/batch_predict", methods=["POST"])
def batch_predict_route():
    data = request.json or {}
    if "rows" not in data:
        return missing_field("rows")
    if err := numeric_prediction_error(data["rows"], "聚类批量预测输入", batch=True):
        return service_error("PREDICTION_FAILED", err, 400)
    try:
        result, err = predict_batch(data["rows"], data.get("version_id"))
    except Exception:
        return service_error("PREDICTION_FAILED", prediction_exception_message(), 400)
    if err:
        code = "MODEL_NOT_FOUND" if "没有找到已保存" in str(err) else "PREDICTION_FAILED"
        return service_error(code, err, 404 if code == "MODEL_NOT_FOUND" else 400)
    return jsonify(result)
