"""Data upload & processing API routes."""
import numpy as np
from flask import Blueprint, request, jsonify
from routes._responses import api_error, session_expired
from services.data_service import parse_file, detect_outliers, build_data_summary, serialize_preview
from session_store import create_session, get_session, get_session_meta, update_session

data_bp = Blueprint("data", __name__)


@data_bp.route("/upload", methods=["POST"])
def upload():
    """Upload CSV/Excel, return session_id + data summary."""
    if 'file' not in request.files:
        return api_error("NO_FILE", "No file provided", 400, "Upload a CSV or Excel file in the `file` form field.")
    file = request.files['file']
    try:
        file_bytes = file.read()
        df = parse_file(file_bytes, file.filename)
        sid = create_session(df, source_name=file.filename)
        meta = get_session_meta(sid)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        outliers = detect_outliers(df)
        return jsonify({
            "session_id": sid,
            "session_meta": meta,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": [str(c) for c in df.columns],
            "numeric_cols": [str(c) for c in numeric_cols],
            "categorical_cols": [str(c) for c in cat_cols],
            "outliers": {str(k): v for k, v in outliers.items()},
            "preview": serialize_preview(df),
            "summary": build_data_summary(df)
        })
    except Exception as e:
        return api_error("UPLOAD_FAILED", "Failed to parse uploaded file", 500, str(e))


@data_bp.route("/<sid>/outliers", methods=["GET"])
def get_outliers(sid):
    """Re-run outlier detection on session data."""
    df = get_session(sid)
    if df is None:
        return session_expired()
    outliers = detect_outliers(df)
    return jsonify({"outliers": {str(k): v for k, v in outliers.items()}})


@data_bp.route("/<sid>/process", methods=["POST"])
def process_data(sid):
    """Apply processing operations to session data. Returns updated preview."""
    df = get_session(sid)
    if df is None:
        return session_expired()

    data = request.json or {}
    ops = data.get("operations", [])
    for op in ops:
        op_type = op.get("op")
        try:
            if op_type == "drop_na":
                df = df.dropna()
            elif op_type == "drop_duplicates":
                df = df.drop_duplicates()
            elif op_type == "drop_cols":
                df = df.drop(columns=op["cols"], errors="ignore")
            elif op_type == "drop_rows":
                indices = op.get("indices", [])
                df = df.drop(index=indices, errors="ignore")
            elif op_type == "rename_col":
                df = df.rename(columns={op["old"]: op["new"]})
            elif op_type == "astype":
                df[op["col"]] = df[op["col"]].astype(op["dtype"])
        except Exception as e:
            return api_error("PROCESSING_FAILED", f"Operation {op_type} failed", 400, str(e))

    update_session(sid, df)

    return jsonify({
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": [str(c) for c in df.columns],
        "preview": serialize_preview(df)
    })


@data_bp.route("/<sid>/summary", methods=["GET"])
def get_summary(sid):
    """Get data summary for LLM analysis."""
    df = get_session(sid)
    if df is None:
        return session_expired()
    return jsonify({"summary": build_data_summary(df), "session_meta": get_session_meta(sid)})


@data_bp.route("/<sid>/sync", methods=["POST"])
def sync_data(sid):
    """Sync current DataFrame to session. Used by ML pages before training."""
    if 'file' not in request.files:
        return api_error("NO_FILE", "No file provided", 400, "Upload a CSV representation in the `file` form field.")
    file = request.files['file']
    try:
        df = parse_file(file.read(), file.filename)
        if not update_session(sid, df, source_name=file.filename):
            return session_expired()
        return jsonify({
            "status": "synced",
            "session_meta": get_session_meta(sid),
            "n_rows": len(df),
            "n_cols": len(df.columns),
        })
    except Exception as e:
        return api_error("SYNC_FAILED", "Failed to sync current data", 500, str(e))
