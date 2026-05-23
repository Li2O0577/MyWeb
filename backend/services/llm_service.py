"""LLM chat proxy with SSE streaming support."""
import json
import os
import re
import requests

# 允许的 API Base 域名白名单 — 防止 SSRF 攻击
_ALLOWED_HOSTS = [
    # OpenAI + Azure
    "api.openai.com",
    r".*\.openai\.azure\.com",
    # 国内主流 LLM 提供商
    "api.deepseek.com",
    "dashscope.aliyuncs.com",         # 阿里通义千问
    "open.bigmodel.cn",               # 智谱 GLM
    "api.moonshot.cn",                # Moonshot/Kimi
    "api.baichuan-ai.com",            # 百川
    "api.minimax.chat",               # MiniMax
    "api.zhipuai.cn",                 # 智谱 AI
    # 本地开发
    "localhost",
    "127.0.0.1",
    r"192\.168\..*",
    r"10\..*",
    r"172\.(1[6-9]|2[0-9]|3[0-1])\..*",
]

_MAX_MESSAGE_LENGTH = 32000  # max chars per message to avoid abuse


def _validate_api_base(api_base):
    """Check api_base against the whitelist. Returns (ok, error_message)."""
    from urllib.parse import urlparse

    if not api_base or not isinstance(api_base, str):
        return False, "api_base 不能为空"
    if len(api_base) > 256:
        return False, "api_base 长度不能超过 256 字符"

    try:
        parsed = urlparse(api_base)
    except Exception:
        return False, f"无法解析 api_base: {api_base}"

    hostname = parsed.hostname or ""
    if not hostname:
        return False, f"api_base 缺少有效主机名: {api_base}"

    for pattern in _ALLOWED_HOSTS:
        if re.fullmatch(pattern, hostname):
            return True, None

    return False, f"不允许的 API 主机: {hostname}。请使用受支持的 LLM 提供商。"


def stream_chat(api_base, api_key, model, messages, temperature=0.7, timeout=180):
    """Generator that yields text chunks from an OpenAI-compatible chat API.

    If api_key is empty, falls back to LLM_API_KEY environment variable.
    """
    ok, err = _validate_api_base(api_base)
    if not ok:
        yield None, err
        return

    # 优先使用传入的 key，否则从环境变量读取
    effective_key = api_key or os.environ.get("LLM_API_KEY", "")
    if not effective_key:
        yield None, "未提供 API Key。请在设置中填写，或设置环境变量 LLM_API_KEY。"
        return

    # 验证 messages 结构
    if not isinstance(messages, list) or len(messages) == 0:
        yield None, "messages 不能为空"
        return
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            yield None, f"messages[{i}] 格式无效"
            return
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > _MAX_MESSAGE_LENGTH:
            yield None, f"messages[{i}] 内容过长，上限 {_MAX_MESSAGE_LENGTH} 字符"
            return

    try:
        resp = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {effective_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": True
            },
            timeout=timeout,
            stream=True
        )
    except Exception as e:
        yield None, f"无法连接到 LLM API: {e}"
        return

    if resp.status_code != 200:
        yield None, f"API 错误 [{resp.status_code}]: {resp.text[:500]}"
        return

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break
        try:
            delta = json.loads(data_str)["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                yield delta["content"], None
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
