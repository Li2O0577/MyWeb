"""LLM chat proxy with SSE streaming support."""
import json
import requests


def stream_chat(api_base, api_key, model, messages, temperature=0.7, timeout=180):
    """Generator that yields text chunks from an OpenAI-compatible chat API."""
    resp = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
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

    if resp.status_code != 200:
        yield None, f"API 错误 [{resp.status_code}]: {resp.text}"
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
