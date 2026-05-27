"""Shared version-management routes — one factory for all 5 ML modules.

Usage in e.g. regression_routes.py:
    from routes._versioning import setup_version_routes
    reg_bp = Blueprint("regression", __name__)
    setup_version_routes(reg_bp, "regression")
"""

from flask import jsonify, request
from routes._responses import missing_field, service_error
from models.registry import (
    list_versions, get_version, get_active_version,
    activate_version, delete_version,
)


def setup_version_routes(bp, model_type):
    """Register /status /versions /activate /clear and version CRUD on *bp*."""

    @bp.route("/status", methods=["GET"])
    def status():
        vid = get_active_version(model_type)
        if not vid:
            return jsonify({"has_model": False})
        meta = get_version(model_type, vid)
        if not meta:
            return jsonify({"has_model": False})
        return jsonify({
            "has_model": True, "version_id": vid,
            "features": meta.get("features", []),
            "categorical_features": meta.get("categorical_features", []),
            "target": meta.get("target", ""),
            "metrics": meta.get("metrics", {}),
            "params": meta.get("params", {}),
            "created_at": meta.get("created_at", ""),
            "dataset_name": meta.get("dataset_name", ""),
        })

    @bp.route("/versions", methods=["GET"])
    def versions():
        versions = list_versions(model_type)
        active = get_active_version(model_type)
        return jsonify({"versions": versions, "active": active})

    @bp.route("/version/<version_id>", methods=["GET"])
    def version_detail(version_id):
        meta = get_version(model_type, version_id)
        if not meta:
            return service_error("VERSION_NOT_FOUND", "Version not found", 404)
        return jsonify({"version_id": version_id, **meta})

    @bp.route("/activate", methods=["POST"])
    def activate():
        data = request.json or {}
        if "version_id" not in data:
            return missing_field("version_id")
        ok = activate_version(model_type, data["version_id"])
        if not ok:
            return service_error("VERSION_NOT_FOUND", "Version not found", 404)
        return jsonify({"status": "activated", "version_id": data["version_id"]})

    @bp.route("/version/<version_id>", methods=["DELETE"])
    def delete_version_route(version_id):
        ok = delete_version(model_type, version_id)
        if not ok:
            return service_error("VERSION_NOT_FOUND", "Version not found", 404)
        return jsonify({"status": "deleted"})

    @bp.route("/clear", methods=["POST"])
    def clear():
        vid = get_active_version(model_type)
        if vid:
            delete_version(model_type, vid)
        return jsonify({"status": "cleared"})
