"""Data upload & processing API routes."""
import logging
from flask import Blueprint, request, jsonify
from routes._auth import current_user_id
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
from session_store import (
    create_session,
    get_pipeline,
    get_session,
    get_session_meta,
    processing_history,
    redo_session,
    save_pipeline,
    undo_session,
    update_session,
)

data_bp = Blueprint("data", __name__)
_log = logging.getLogger(__name__)


MAX_FILE_SIZE = 256 * 1024 * 1024  # 256 MB


def _profile_response(df, sid, extra=None):
    user_id = current_user_id()
    profile = build_data_profile(df, session_id=sid, session_meta=get_session_meta(sid, user_id=user_id))
    profile["processing_history"] = processing_history(sid, user_id=user_id)
    if extra:
        profile.update(extra)
    return profile

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
        sid = create_session(df, source_name=file.filename, user_id=current_user_id())
        return jsonify(_profile_response(df, sid))
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
    df = get_session(sid, user_id=current_user_id())
    if df is None:
        return session_expired()
    try:
        coefficient = float(request.args.get("coefficient", 1.5))
    except (TypeError, ValueError):
        return api_error("INVALID_OUTLIER_COEFFICIENT", "异常值检测系数必须是数字。", 400)
    if coefficient <= 0 or coefficient > 10:
        return api_error("INVALID_OUTLIER_COEFFICIENT", "异常值检测系数需要在 0 到 10 之间。", 400)
    outliers = detect_outliers(df, coefficient=coefficient)
    return jsonify({"coefficient": coefficient, "outliers": {str(k): v for k, v in outliers.items()}})


@data_bp.route("/<sid>/process", methods=["POST"])
def process_data(sid):
    """Apply processing operations to session data. Returns updated preview."""
    df = get_session(sid, user_id=current_user_id())
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

    update_session(sid, df, user_id=current_user_id(), history_entry={
        "label": messages[0] if messages else "数据处理",
        "operations": ops,
        "messages": messages,
    })

    return jsonify(_profile_response(df, sid, {"operations_applied": messages}))


@data_bp.route("/<sid>/profile", methods=["GET"])
def get_profile(sid):
    """Return structured profile + preview for frontend recovery and data pages."""
    df = get_session(sid, user_id=current_user_id())
    if df is None:
        return session_expired()
    return jsonify(_profile_response(df, sid))


@data_bp.route("/<sid>/history", methods=["GET"])
def get_processing_history(sid):
    """Return processing history, undo/redo state, and saved pipelines."""
    if get_session(sid, user_id=current_user_id()) is None:
        return session_expired()
    return jsonify(processing_history(sid, user_id=current_user_id()))


@data_bp.route("/<sid>/undo", methods=["POST"])
def undo_processing(sid):
    """Restore the previous processing snapshot for this session."""
    user_id = current_user_id()
    df = undo_session(sid, user_id=user_id)
    if df is None:
        if get_session(sid, user_id=user_id) is None:
            return session_expired()
        return api_error("UNDO_NOT_AVAILABLE", "当前没有可撤销的数据处理步骤", 400)
    return jsonify(_profile_response(df, sid, {"operations_applied": ["已撤销上一步处理"]}))


@data_bp.route("/<sid>/redo", methods=["POST"])
def redo_processing(sid):
    """Restore the next processing snapshot for this session."""
    user_id = current_user_id()
    df = redo_session(sid, user_id=user_id)
    if df is None:
        if get_session(sid, user_id=user_id) is None:
            return session_expired()
        return api_error("REDO_NOT_AVAILABLE", "当前没有可重做的数据处理步骤", 400)
    return jsonify(_profile_response(df, sid, {"operations_applied": ["已重做下一步处理"]}))


@data_bp.route("/<sid>/pipelines", methods=["POST"])
def save_processing_pipeline(sid):
    """Save current processing history or supplied operations as a reusable pipeline."""
    user_id = current_user_id()
    if get_session(sid, user_id=user_id) is None:
        return session_expired()
    data = request.json or {}
    pipeline = save_pipeline(sid, name=data.get("name"), operations=data.get("operations"), user_id=user_id)
    if pipeline is None:
        return api_error(
            "PIPELINE_EMPTY",
            "当前没有可保存的数据处理流水线",
            400,
            "请先执行至少一步数据处理，或提交 operations 字段。",
        )
    return jsonify({"pipeline": pipeline, "processing_history": processing_history(sid, user_id=user_id)})


@data_bp.route("/<sid>/pipelines/<pipeline_id>/apply", methods=["POST"])
def apply_processing_pipeline(sid, pipeline_id):
    """Apply a saved processing pipeline to the current session data."""
    user_id = current_user_id()
    df = get_session(sid, user_id=user_id)
    if df is None:
        return session_expired()
    pipeline = get_pipeline(sid, pipeline_id, user_id=user_id)
    if pipeline is None:
        return api_error("PIPELINE_NOT_FOUND", "没有找到对应的数据处理流水线", 404)
    operations = pipeline.get("operations") or []
    try:
        df, messages = apply_processing_operations(df, operations)
    except Exception as exc:
        _log.exception("Failed to apply processing pipeline")
        return api_error(
            "PIPELINE_APPLY_FAILED",
            "数据处理流水线执行失败",
            400,
            str(exc) or "请检查流水线中的列名、表达式或数据类型是否仍然适用于当前数据。",
        )
    update_session(sid, df, user_id=user_id, history_entry={
        "label": f"应用流水线：{pipeline.get('name') or pipeline_id}",
        "operations": operations,
        "messages": messages,
    })
    return jsonify(_profile_response(df, sid, {"operations_applied": messages}))


@data_bp.route("/<sid>/summary", methods=["GET"])
def get_summary(sid):
    """Get data summary for LLM analysis."""
    df = get_session(sid, user_id=current_user_id())
    if df is None:
        return session_expired()
    return jsonify({"summary": build_data_summary(df), "session_meta": get_session_meta(sid, user_id=current_user_id())})


@data_bp.route("/<sid>/visualize", methods=["POST"])
def visualize_data(sid):
    """Return structured chart data for the React visualization workspace."""
    df = get_session(sid, user_id=current_user_id())
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
        if not update_session(sid, df, source_name=file.filename, user_id=current_user_id()):
            return session_expired()
        return jsonify({
            "status": "synced",
            "session_meta": get_session_meta(sid, user_id=current_user_id()),
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
