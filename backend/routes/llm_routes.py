"""LLM direct and agent chat routes."""
import json

from flask import Blueprint, Response, request

from routes._responses import missing_field

llm_bp = Blueprint("llm", __name__)


def stream_chat(_api_base, _api_key, _model, _messages):
    yield "当前未配置真实 LLM 服务。", None


def agent_chat(_api_base, _api_key, _model, _messages, _session_id):
    yield {"chunk": "当前未配置真实 Agent 服务。"}
    yield {"done": True, "tools_used": 0}


def _sse(events):
    def generate():
        for event in events:
            yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
    return Response(generate(), mimetype="text/event-stream")


@llm_bp.route("/chat", methods=["POST"])
def chat_route():
    data = request.json or {}
    for field in ("api_base", "model", "messages"):
        if field not in data:
            return missing_field(field)

    def events():
        full = ""
        for chunk, err in stream_chat(data["api_base"], data.get("api_key", ""), data["model"], data["messages"]):
            if err:
                yield {"error": err}
                return
            full += chunk
            yield {"chunk": chunk, "full": full}
        yield {"done": True, "full": full}

    return _sse(events())


@llm_bp.route("/agent", methods=["POST"])
def agent_route():
    data = request.json or {}
    for field in ("session_id", "api_base", "model", "messages"):
        if field not in data:
            return missing_field(field)

    return _sse(agent_chat(
        data["api_base"],
        data.get("api_key", ""),
        data["model"],
        data["messages"],
        data["session_id"],
    ))
