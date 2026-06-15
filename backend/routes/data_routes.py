"""Data upload & processing API routes."""
import logging
from flask import Blueprint, request, jsonify
from routes._responses import api_error, session_expired
from services.data_service import (
    parse_file,
    detect_outliers,
    build_data_summary,
    serialize_preview,
    build_data_profile,
    apply_processing_operations,
)
from services.visualization_service import build_visualization_payload
from session_store import create_session, get_session, get_session_meta, update_session

data_bp = Blueprint("data", __name__)
_log = logging.getLogger(__name__)


MAX_FILE_SIZE = 256 * 1024 * 1024  # 256 MB

@data_bp.route("/upload", methods=["POST"])
def upload():
    """Upload CSV/Excel, return session_id + data summary."""
    if 'file' not in request.files:
        return api_error("NO_FILE", "没有收到上传文件", 400, "请在 file 表单字段中上传 CSV 或 Excel 文件。")
    file = request.files['file']
    # Check Content-Length before reading into memory
    cl = request.content_length
    if cl is not None and cl > MAX_FILE_SIZE:
        return api_error(
            "FILE_TOO_LARGE",
            f"上传文件过大（约 {cl / 1024 / 1024:.0f} MB），最大允许 256 MB。",
            413,
            detail="请先压缩、拆分文件，或减少数据量后再上传。",
        )
    try:
        file_bytes = file.read()
        if len(file_bytes) > MAX_FILE_SIZE:
            return api_error(
                "FILE_TOO_LARGE",
                f"上传文件过大（约 {len(file_bytes) / 1024 / 1024:.0f} MB），最大允许 256 MB。",
                413,
            )
        df = parse_file(file_bytes, file.filename)
        sid = create_session(df, source_name=file.filename)
        meta = get_session_meta(sid)
        return jsonify(build_data_profile(df, session_id=sid, session_meta=meta))
    except Exception:
        _log.exception("Failed to parse uploaded file")
        return api_error(
            "UPLOAD_FAILED",
            "上传文件解析失败",
            400,
            "请确认文件格式为 CSV 或 Excel，且内容没有损坏。",
        )


@data_bp.route("/<sid>/outliers", methods=["GET"])
def get_outliers(sid):
    """Re-run outlier detection on session data."""
    df = get_session(sid)
    if df is None:
        return session_expired()
    outliers = detect_outliers(df, coefficient=1.5)
    return jsonify({"outliers": {str(k): v for k, v in outliers.items()}})


@data_bp.route("/<sid>/process", methods=["POST"])
def process_data(sid):
    """Apply processing operations to session data. Returns updated preview."""
    df = get_session(sid)
    if df is None:
        return session_expired()

    data = request.json or {}
    ops = data.get("operations", [])
    try:
        df, messages = apply_processing_operations(df, ops)
    except Exception as exc:
        _log.exception("Data processing operation failed")
        return api_error(
            "PROCESSING_FAILED",
            "数据处理操作失败",
            400,
            str(exc) or "请检查选择的列、行号、目标类型或表达式是否有效。",
        )

    update_session(sid, df)

    profile = build_data_profile(df, session_id=sid, session_meta=get_session_meta(sid))
    profile["operations_applied"] = messages
    return jsonify(profile)


@data_bp.route("/<sid>/profile", methods=["GET"])
def get_profile(sid):
    """Return structured profile + preview for frontend recovery and data pages."""
    df = get_session(sid)
    if df is None:
        return session_expired()
    return jsonify(build_data_profile(df, session_id=sid, session_meta=get_session_meta(sid)))


@data_bp.route("/<sid>/summary", methods=["GET"])
def get_summary(sid):
    """Get data summary for LLM analysis."""
    df = get_session(sid)
    if df is None:
        return session_expired()
    return jsonify({"summary": build_data_summary(df), "session_meta": get_session_meta(sid)})


@data_bp.route("/<sid>/visualize", methods=["POST"])
def visualize_data(sid):
    """Return structured chart data for the React visualization workspace."""
    df = get_session(sid)
    if df is None:
        return session_expired()

    try:
        payload = build_visualization_payload(df, request.json or {})
        return jsonify(payload)
    except Exception as exc:
        _log.exception("Visualization generation failed")
        return api_error(
            "VISUALIZATION_FAILED",
            "图表数据生成失败",
            400,
            str(exc) or "请检查图表类型、字段选择和聚合配置。",
        )


@data_bp.route("/<sid>/sync", methods=["POST"])
def sync_data(sid):
    """Sync current DataFrame to session. Used by ML pages before training."""
    if 'file' not in request.files:
        return api_error("NO_FILE", "没有收到同步文件", 400, "请在 file 表单字段中上传当前数据的 CSV 内容。")
    file = request.files['file']
    cl = request.content_length
    if cl is not None and cl > MAX_FILE_SIZE:
        return api_error(
            "FILE_TOO_LARGE",
            f"同步文件过大（约 {cl / 1024 / 1024:.0f} MB），最大允许 256 MB。",
            413,
            detail="请减少数据量后再同步。",
        )
    try:
        file_bytes = file.read()
        if len(file_bytes) > MAX_FILE_SIZE:
            return api_error(
                "FILE_TOO_LARGE",
                f"同步文件过大（约 {len(file_bytes) / 1024 / 1024:.0f} MB），最大允许 256 MB。",
                413,
                detail="请减少数据量后再同步。",
            )
        df = parse_file(file_bytes, file.filename)
        if not update_session(sid, df, source_name=file.filename):
            return session_expired()
        return jsonify({
            "status": "synced",
            "session_meta": get_session_meta(sid),
            "n_rows": len(df),
            "n_cols": len(df.columns),
        })
    except Exception:
        _log.exception("Failed to sync current data")
        return api_error(
            "SYNC_FAILED",
            "当前数据同步失败",
            400,
            "请确认当前数据可以导出为 CSV，且列名和内容没有异常。",
        )
