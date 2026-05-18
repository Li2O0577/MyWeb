"""LLM chat API route with SSE streaming."""
from flask import Blueprint, request, Response, jsonify
import json
from services.llm_service import stream_chat

llm_bp = Blueprint("llm", __name__)


@llm_bp.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    for k in ("api_base", "api_key", "model", "messages"):
        if k not in data:
            return jsonify({"error": f"Missing required field: {k}"}), 400
    api_base = data["api_base"]
    api_key = data["api_key"]
    model = data["model"]
    messages = data["messages"]

    def generate():
        full_text = ""
        for chunk, err in stream_chat(api_base, api_key, model, messages):
            if err:
                yield f"data: {json.dumps({'error': err})}\n\n"
                return
            if chunk:
                full_text += chunk
                yield f"data: {json.dumps({'chunk': chunk, 'full': full_text})}\n\n"
        yield f"data: {json.dumps({'done': True, 'full': full_text})}\n\n"

    return Response(generate(), mimetype="text/event-stream")
