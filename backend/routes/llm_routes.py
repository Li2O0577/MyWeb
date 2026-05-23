"""LLM chat API route with SSE streaming."""
import os
from flask import Blueprint, request, Response
import json
from routes._responses import error_event, missing_field
from services.llm_service import stream_chat

llm_bp = Blueprint("llm", __name__)


@llm_bp.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    for k in ("api_base", "model", "messages"):
        if k not in data:
            return missing_field(k)
    api_base = data["api_base"]
    api_key = data.get("api_key", "") or os.environ.get("LLM_API_KEY", "")
    model = data["model"]
    messages = data["messages"]

    def generate():
        full_text = ""
        for chunk, err in stream_chat(api_base, api_key, model, messages):
            if err:
                yield f"data: {json.dumps(error_event('LLM_STREAM_FAILED', err, err))}\n\n"
                return
            if chunk:
                full_text += chunk
                yield f"data: {json.dumps({'chunk': chunk, 'full': full_text})}\n\n"
        yield f"data: {json.dumps({'done': True, 'full': full_text})}\n\n"

    return Response(generate(), mimetype="text/event-stream")
