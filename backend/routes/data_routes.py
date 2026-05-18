"""Data upload & processing API routes."""
from flask import Blueprint, request, jsonify
from services.data_service import parse_file, detect_outliers, build_data_summary, serialize_preview
from app import create_session, get_session

data_bp = Blueprint("data", __name__)


@data_bp.route("/upload", methods=["POST"])
def upload():
    """Upload CSV/Excel, return session_id + data summary."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    try:
        df = parse_file(file.read(), file.filename)
        sid = create_session(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        outliers = detect_outliers(df)
        return jsonify({
            "session_id": sid,
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
        return jsonify({"error": str(e)}), 500


@data_bp.route("/<sid>/outliers", methods=["GET"])
def get_outliers(sid):
    """Re-run outlier detection on session data."""
    df = get_session(sid)
    if df is None:
        return jsonify({"error": "Session not found or expired"}), 404
    outliers = detect_outliers(df)
    return jsonify({"outliers": {str(k): v for k, v in outliers.items()}})


@data_bp.route("/<sid>/process", methods=["POST"])
def process_data(sid):
    """Apply processing operations to session data. Returns updated preview."""
    df = get_session(sid)
    if df is None:
        return jsonify({"error": "Session not found or expired"}), 404

    ops = request.json.get("operations", [])
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
            return jsonify({"error": f"Operation {op_type} failed: {str(e)}"}), 400

    # Update session
    from app import _sessions, _sessions_lock
    import time
    with _sessions_lock:
        _sessions[sid] = {"df": df, "at": time.time()}

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
        return jsonify({"error": "Session not found or expired"}), 404
    return jsonify({"summary": build_data_summary(df)})
