"""Classification training & prediction API routes."""
from flask import Blueprint, request, jsonify
from services.classification_service import train, predict_one, predict_batch
from app import get_session

cls_bp = Blueprint("classification", __name__)


def _require(data, *keys):
    for k in keys:
        if k not in data:
            return jsonify({"error": f"Missing required field: {k}"}), 400
    return None


@cls_bp.route("/train", methods=["POST"])
def train_model():
    data = request.json or {}
    if err := _require(data, "session_id", "target_col", "feature_cols", "learning_rate", "epochs", "batch_size"):
        return err
    sid = data["session_id"]
    df = get_session(sid)
    if df is None:
        return jsonify({"error": "Session not found or expired"}), 404

    target_col = data["target_col"]
    col_type = type(df.columns[0])
    feature_cols = []
    for c in data["feature_cols"]:
        try:
            feature_cols.append(col_type(c))
        except (ValueError, TypeError):
            feature_cols.append(c)

    n_samples = len(df)
    n_features = len(feature_cols)

    if n_samples < 500:
        h1, h2, dr = max(8, n_features), max(4, n_features // 2), 0.1
    elif n_samples < 5000:
        h1, h2, dr = n_features * 2, n_features, 0.2
    else:
        h1, h2, dr = n_features * 3, n_features * 2, 0.3

    result = train(df, target_col, feature_cols, h1, h2, dr,
                   data["learning_rate"], data["epochs"], data["batch_size"],
                   data.get("device", "cpu"))
    return jsonify(result)


@cls_bp.route("/predict", methods=["POST"])
def predict():
    data = request.json
    result, err = predict_one(data["features"], data.get("device", "cpu"))
    if err:
        return jsonify({"error": err}), 404
    return jsonify(result)


@cls_bp.route("/batch_predict", methods=["POST"])
def batch_predict():
    data = request.json
    result, err = predict_batch(data["rows"], data.get("device", "cpu"))
    if err:
        return jsonify({"error": err}), 404
    return jsonify(result)


@cls_bp.route("/clear", methods=["POST"])
def clear():
    import os
    for f in ["models/cls_best_model.pth", "models/cls_scaler.pkl", "models/cls_config.json"]:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f)
        if os.path.exists(path):
            os.remove(path)
    return jsonify({"status": "cleared"})
