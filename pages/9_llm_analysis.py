"""LLM analysis page — Flask backend proxies chat completions with SSE streaming."""
import streamlit as st
import pandas as pd
import numpy as np
import json
import base64
import requests
from pages._prepare import render_sidebar, data_uploader
from pages._api import chat_llm, agent_chat_llm, get_summary, backend_status_badge, render_backend_sync_panel
from pages._ui_common import render_notice, render_page_header, render_section_header, render_status_strip

MAX_RENDER_IMAGE_BYTES = 8 * 1024 * 1024
MAX_RENDER_IMAGE_B64_CHARS = (MAX_RENDER_IMAGE_BYTES * 4) // 3 + 8
MAX_DIRECT_DATA_ROWS = 1000
MAX_DIRECT_DATA_BYTES = 512 * 1024

LLM_ERROR_MESSAGES = {
    "NO_API_KEY": "未提供 API Key。请填写 API Key，或在后端环境变量中设置 LLM_API_KEY。",
    "INVALID_API_BASE": "API Base 地址不可用。请使用受支持的 LLM 服务地址；本地地址需要开启 LLM_ALLOW_LOCAL_API_BASE=1。",
    "INVALID_MODEL": "模型名称不符合要求，请检查 Model 输入。",
    "INVALID_MESSAGES": "对话内容格式异常，请清空聊天记录后重试。",
    "SESSION_EXPIRED": "当前数据已过期。请点击上方“重新同步当前数据到后端”后再试。",
    "AGENT_TOOL_LIMIT": "本次 Agent 分析调用工具次数过多，请缩小问题范围后继续。",
    "LLM_STREAM_FAILED": "大模型流式响应失败，请检查 API 配置、模型名称或网络状态。",
    "LLM_REQUEST_FAILED": "大模型请求失败，请检查 API 配置或稍后重试。",
    "ERROR": "请求没有完成，请稍后重试。",
}

st.set_page_config(page_title="大模型分析", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
[data-testid="stickyNav"] {display: none;}
.stChatMessage [data-testid="stExpander"] { margin-bottom: 0.2rem !important; }
.stChatMessage [data-testid="stExpander"] + [data-testid="stExpander"] { margin-top: 0 !important; }
.stChatMessage [data-testid="stCaptionContainer"] { margin-bottom: 0.3rem !important; }
[data-testid="stExpander"] .stExpander { gap: 0.2rem; }
.stImage { display: block !important; }
.stImage img { display: block !important; max-width: min(100%, 680px) !important; height: auto !important; }
</style>""", unsafe_allow_html=True)
render_sidebar("pages/9_llm_analysis.py")
render_page_header("大模型分析", "使用 Smart、Direct 或 Agent 模式完成数据问答、图表生成和自动分析。")

backend_status_badge()

df = data_uploader(upload_to_backend=False)

if df is None:
    st.info("👋 请先在「📊 数据加载」页面上传数据，然后再使用大模型分析功能。")
    st.stop()

backend_synced = render_backend_sync_panel(df)


def _local_data_summary(df):
    """Build a comprehensive data summary locally when no backend session is available."""
    import io
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    buf = io.StringIO()
    buf.write(f"## Data Summary\n\n**Rows:** {len(df)}  |  **Columns:** {len(df.columns)}\n\n")

    if len(numeric_cols) > 0:
        buf.write(f"### Numeric Columns ({len(numeric_cols)})\n")
        for c in numeric_cols[:20]:
            s = df[c]
            buf.write(f"- **{c}**: mean={s.mean():.2f}, std={s.std():.2f}, min={s.min():.2f}, max={s.max():.2f}, missing={s.isna().sum()}\n")
        if len(numeric_cols) > 20:
            buf.write(f"... and {len(numeric_cols) - 20} more numeric columns\n")

    if len(cat_cols) > 0:
        buf.write(f"\n### Categorical Columns ({len(cat_cols)})\n")
        for c in cat_cols[:10]:
            s = df[c]
            buf.write(f"- **{c}**: unique={s.nunique()}, missing={s.isna().sum()}\n")
        if len(cat_cols) > 10:
            buf.write(f"... and {len(cat_cols) - 10} more categorical columns\n")

    return buf.getvalue()


def _render_image(img_event):
    """Render a base64-encoded image from an SSE image event."""
    import io as _io
    try:
        encoded = img_event.get("base64", "")
        if not isinstance(encoded, str) or not encoded:
            st.toast("图片数据缺失，已跳过。", icon="⚠️")
            return
        if len(encoded) > MAX_RENDER_IMAGE_B64_CHARS:
            st.toast("图片数据过大，已跳过渲染。", icon="⚠️")
            return
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) < 64:
            st.toast("图片数据异常（过短），已跳过。", icon="⚠️")
            return
        if len(raw) > MAX_RENDER_IMAGE_BYTES:
            st.toast("图片数据过大，已跳过渲染。", icon="⚠️")
            return
        buf = _io.BytesIO(raw)
        buf.seek(0)
        st.image(buf, caption=img_event.get("title", ""), width=640)
    except Exception:
        st.toast("图片解码失败，已跳过。", icon="⚠️")


def _message_for_llm(msg):
    """Keep chat payloads compatible with OpenAI-style message schemas."""
    return {"role": msg.get("role"), "content": str(msg.get("content", ""))}


def _direct_data_text(df):
    """Return bounded CSV text for Direct mode, or a user-facing error."""
    if len(df) > MAX_DIRECT_DATA_ROWS:
        return None, (
            f"直接模式最多发送 {MAX_DIRECT_DATA_ROWS} 行原始数据，当前数据有 {len(df)} 行。"
            "请改用智能模式或 Agent 模式，或先在数据处理页筛选/抽样后再试。"
        )
    data_text = df.to_csv(index=False)
    data_bytes = len(data_text.encode("utf-8"))
    if data_bytes > MAX_DIRECT_DATA_BYTES:
        return None, (
            f"直接模式最多发送约 {MAX_DIRECT_DATA_BYTES // 1024} KB 原始数据，"
            f"当前约 {data_bytes // 1024} KB。请改用智能模式或 Agent 模式，或先减少列/行。"
        )
    return data_text, None


def _format_llm_error(err):
    if isinstance(err, dict):
        code = err.get("code", "ERROR")
        message = err.get("message") or LLM_ERROR_MESSAGES.get(code) or "请求没有完成。"
        friendly = LLM_ERROR_MESSAGES.get(code, message)
        if message and message != friendly and code not in {"INVALID_API_BASE", "AGENT_TOOL_LIMIT"}:
            return f"{friendly}（{message}）"
        return friendly
    if err:
        return str(err)
    return "请求没有完成，请稍后重试。"


def _format_response_error(response, fallback):
    """Return a friendly message from an error HTTP response."""
    try:
        payload = response.json()
    except Exception:
        return fallback
    if isinstance(payload, dict) and "error" in payload:
        return _format_llm_error(payload["error"])
    return fallback


def _format_request_exception(exc, context="大模型请求"):
    """Map request exceptions to user-facing Chinese messages."""
    if isinstance(exc, requests.exceptions.ConnectionError):
        return f"{context}连接失败，请检查 Flask 后端、API Base 或网络状态。"
    if isinstance(exc, requests.exceptions.Timeout):
        return f"{context}超时，请缩小问题范围、减少上下文，或稍后重试。"
    if isinstance(exc, requests.exceptions.RequestException):
        return f"{context}没有完成，请检查 API 配置或网络状态。"
    return f"{context}出现异常，请稍后重试。"


def _iter_sse_events(response):
    """Yield parsed JSON events from an SSE streaming response."""
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        try:
            yield json.loads(line[6:])
        except json.JSONDecodeError:
            continue


def _build_segments(full_reply, images):
    """Build response segments without breaking markdown blocks.

    Returns a list of dicts: {"type": "text", "content": ...} or
    {"type": "image", "base64": ..., "title": ..., "alt": ...}.
    """
    segments = []
    if full_reply:
        segments.append({"type": "text", "content": full_reply})
    if not images:
        return segments
    for img in images:
        segments.append({
            "type": "image",
            "base64": img.get("base64", ""),
            "title": img.get("title", ""),
            "alt": img.get("alt", "")
        })

    return segments


def _render_segments(segments):
    """Render a list of content segments (text + images) in order."""
    for seg in segments:
        if seg["type"] == "text":
            st.markdown(seg["content"])
        elif seg["type"] == "image":
            _render_image(seg)


TOOL_NAME_CN = {
    "get_data_overview": "数据概览",
    "get_column_details": "列详情",
    "run_regression": "回归分析",
    "run_classification": "分类分析",
    "run_clustering": "聚类分析",
    "run_correlation_analysis": "相关性分析",
    "generate_chart": "图表生成",
    "run_code_interpreter": "代码执行",
}


def _render_compact_tools(tools):
    """Render tool calls as a compact grouped block."""
    if not tools:
        return
    with st.container():
        tool_names = []
        for t in tools:
            name = t.get("name") or t.get("tool_name", "Tool")
            tool_names.append(TOOL_NAME_CN.get(name, name))
        render_notice("工具调用", "已使用工具：" + " → ".join(tool_names), level="info")
        for t in tools:
            name = t.get("name") or t.get("tool_name", "Tool")
            display_name = TOOL_NAME_CN.get(name, name)
            content = t.get("content", "")
            if not content:
                args = t.get("args", {})
                content = f"参数: {json.dumps(args, ensure_ascii=False)}"
            with st.expander(f"🔧 {display_name}", expanded=False):
                st.code(content, language=None)


def _tool_event_content(tool_event):
    args = json.dumps(tool_event.get("args", {}), ensure_ascii=False)
    parts = [f"参数: {args}"]
    result = str(tool_event.get("result", "")).strip()
    if result:
        parts.append(f"结果:\n{result}")
    image_titles = [img.get("title") or img.get("alt") or "未命名图片" for img in tool_event.get("images", [])]
    if image_titles:
        parts.append("图片: " + "、".join(image_titles))
    return "\n\n".join(parts)


def _finalize_agent_response(full_reply, images, tool_events):
    """Render and save agent response, then trigger page rerun."""
    segments = _build_segments(full_reply, images)
    _render_segments(segments)
    if tool_events:
        _render_compact_tools(tool_events)
    msg_record = {"role": "assistant", "content": full_reply, "segments": segments}
    st.session_state.chat_messages.append(msg_record)
    for te in tool_events:
        st.session_state.chat_messages.append({
            "role": "tool",
            "tool_name": te["name"],
            "content": _tool_event_content(te)
        })
    st.rerun()


def _rollback_user_message(data_was_attached=False):
    """Remove the last user message on failure and revert data-sent flag."""
    if st.session_state.chat_messages:
        st.session_state.chat_messages.pop()
    if data_was_attached:
        st.session_state.chat_data_sent = False


SKILL_INFO = """
## Available Analysis Skills in This Platform

You are part of a data analysis platform. The following built-in modules are available on other pages. Do NOT try to execute them yourself — your job is to recommend which ones the user should use and what to look for.

| # | Page | What It Does |
|---|------|---------------|
| 1 | Data Load | Upload CSV/Excel, basic preview |
| 2 | Data Visualization | Scatter, line, bar, histogram, box plot, density, correlation heatmap, pie, pair plot |
| 3 | Data Processing | Drop columns/rows, rename, type conversion, statistics, StandardScaler, MinMaxScaler, Gaussian noise, Label/OneHot encoding, custom formula columns, PCA |
| 4 | Regression | Train PyTorch MLP for regression (Flask backend) |
| 5 | Classification | Train PyTorch MLP for binary/multi-class classification (Flask backend) |
| 6 | DIY MLP | Customize MLP architecture (layers, neurons, activation, learning rate) (Flask backend) |
| 7 | Decision Tree | Train decision tree classifier + visualize tree + export rules (Flask backend) |
| 8 | Clustering | K-means + DBSCAN clustering, elbow method, silhouette score, PCA visualization (Flask backend) |
| 9 | LLM Analysis | THIS page — you are providing analysis recommendations now |

**Your Task:**
1. Review the data summary below.
2. Give a concise data overview.
3. Recommend 3-5 specific analysis actions, mapping each to the appropriate page above.
4. For each recommendation, explain WHY it's useful.

Respond in clear, well-structured Markdown.
"""

# ── Mode selection ──
render_section_header("分析模式", "选择发送摘要、发送受限原始数据，或让 Agent 自主调用平台工具。")
mode = st.radio("选择与 LLM 的交互方式：", [
    "📊 智能模式 — 发送数据摘要，让 AI 推荐平台分析工具",
    "📄 直接模式 — 发送原始数据给 AI（消耗更多 Token）",
    "🤖 Agent 模式 — AI 自主调用平台工具执行完整分析（需要后端 session）"
], index=0, help="智能：低 Token 成本。直接：发送完整数据。Agent：AI 自主训练模型、聚类、相关性分析、画图、执行代码。")
is_smart = mode.startswith("📊")
is_direct = mode.startswith("📄")
is_agent = mode.startswith("🤖")

# Initialize or reset chat state when mode changes
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
    st.session_state.chat_data_sent = False
    st.session_state.chat_mode = mode
elif st.session_state.get("chat_mode") != mode:
    st.session_state.chat_messages = []
    st.session_state.chat_data_sent = False
    st.session_state.chat_mode = mode
    st.rerun()

# Agent mode: require backend session
if is_agent:
    sid = st.session_state.get("session_id")
    if not sid or not backend_synced:
        st.warning("⚠️ Agent 模式需要有效的后端 session。请先同步当前数据到后端，或确认 Flask 后端已启动。")
        st.stop()

# ── API config ──
render_section_header("API 配置", "默认可使用后端 LLM_API_KEY；本地/私网 API Base 需要后端开启 LLM_ALLOW_LOCAL_API_BASE=1。")
with st.expander("连接设置", expanded=False):
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        api_base = st.text_input("API Base URL", value="https://api.openai.com/v1", placeholder="https://api.openai.com/v1")
    with col2:
        api_key = st.text_input("API Key", type="password", placeholder="sk-...")
    with col3:
        model = st.text_input("Model", value="gpt-4o", placeholder="gpt-4o / deepseek-chat / qwen-plus")

meta = st.session_state.get("session_meta", {}) or {}
dataset_name = meta.get("source_name", "当前数据")
session_id = st.session_state.get("session_id", "")
render_status_strip([
    ("当前数据集", f"{dataset_name} · {len(df)} 行 · {len(df.columns)} 列"),
    ("后端同步状态", "已同步，可以使用 Agent" if backend_synced else "未同步或后端不可用"),
    ("当前 LLM 配置", f"{mode.split('—')[0].strip()} · {model or '未设置模型'} · session {session_id[:12] or '无'}"),
])

if is_agent:
    with st.expander("ℹ️ Agent 模式说明", expanded=False):
        st.markdown("""
        **Agent 模式下 AI 可以自主执行以下操作：**
        - 📊 数据概览 — 获取统计摘要和相关性
        - 🔍 列详情 — 深入分析特定列的分布
        - 📈 回归分析 — 训练 MLP 模型预测连续值目标
        - 🏷️ 分类分析 — 训练 MLP 模型预测分类标签
        - 🔗 聚类分析 — K-means / DBSCAN 发现数据分组
        - 🔗 相关性分析 — 发现变量间线性关系
        - 🎨 图表生成 — AI 自主绘制散点图、直方图、热力图等并内嵌显示
        - 💻 代码执行 — AI 编写 Python 代码进行自定义分析，图表自动捕获

        **使用方式：** 直接用自然语言告诉 AI 你想分析什么，例如：
        - "帮我完整分析这份数据"
        - "画个散点图看看价格和面积的关系"
        - "写代码分析数据分布，画出直方图"
        - "预测房价并评估模型效果"

        AI 会自动选择工具、执行分析、生成图表，所有结果直接展示在对话中。
        """)
else:
    with st.expander("📝 系统提示词（可编辑）", expanded=False):
        default_system = SKILL_INFO if is_smart else "You are a professional data analyst. Please analyze the given data and provide insightful findings."
        system_prompt = st.text_area("系统提示词", value=default_system, height=200 if is_smart else 100)

# ── Chat display ──
col_title, col_clear = st.columns([4, 1])
with col_title:
    render_section_header("对话", "聊天记录、工具调用结果和图表会保留在当前页面。")
with col_clear:
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.chat_data_sent = False
        st.rerun()

# Render chat history
_tool_batch = []
for _msg in st.session_state.chat_messages:
    _role = _msg["role"]
    if _role == "tool":
        _tool_batch.append(_msg)
        continue
    # Flush any pending tool batch before a non-tool message
    if _tool_batch:
        _render_compact_tools(_tool_batch)
        _tool_batch = []
    if _role == "user":
        with st.chat_message("user"):
            st.markdown(_msg["content"])
    elif _role == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            if "segments" in _msg:
                _render_segments(_msg["segments"])
            else:
                st.markdown(_msg["content"])
                for img in _msg.get("images", []):
                    _render_image(img)
# Flush remaining tool batch at end
if _tool_batch:
    _render_compact_tools(_tool_batch)


# ── Chat input ──
user_input = st.chat_input(
    placeholder="输入你的问题或分析需求..." if not is_agent else "告诉 AI 你想做什么分析，例如：「帮我分析这份数据，看看哪些因素影响价格」"
)

if user_input:
    if not model:
        st.toast("请输入模型名称。", icon="❌")
    elif not user_input.strip():
        st.toast("请输入消息。", icon="❌")
    else:
        text = user_input.strip()
        st.session_state.chat_messages.append({"role": "user", "content": text})

        with st.chat_message("user"):
            st.markdown(text)

        if is_agent:
            # ── Agent Mode (buffered: render all at once after completion) ──
            agent_messages = []
            for msg in st.session_state.chat_messages:
                if msg["role"] != "tool":
                    agent_messages.append(_message_for_llm(msg))

            sid = st.session_state.get("session_id", "")

            with st.chat_message("assistant", avatar="🤖"):
                status_placeholder = st.empty()
                tool_placeholder = st.empty()

                tool_events = []
                full_reply = ""
                images = []
                error_occurred = False

                try:
                    status_placeholder.markdown("🤔 正在分析数据...")
                    resp = agent_chat_llm(sid, api_base, api_key, model, agent_messages)
                    if resp.status_code != 200:
                        _rollback_user_message()
                        status_placeholder.empty()
                        tool_placeholder.empty()
                        st.toast(
                            _format_response_error(
                                resp,
                                f"大模型分析请求失败（HTTP {resp.status_code}），请检查 API 配置或稍后重试。",
                            ),
                            icon="❌",
                        )
                        st.stop()

                    for event in _iter_sse_events(resp):
                        if "error" in event:
                            status_placeholder.empty()
                            tool_placeholder.empty()
                            _rollback_user_message()
                            err = event["error"]
                            st.toast(_format_llm_error(err), icon="❌")
                            error_occurred = True
                            break

                        if event.get("status") == "thinking":
                            status_placeholder.markdown(f"🤔 {event.get('message', '正在分析...')}")

                        elif event.get("status") == "image":
                            images.append(event)
                            for te in reversed(tool_events):
                                if te["status"] == "running":
                                    te.setdefault("images", []).append(event)
                                    break

                        elif event.get("status") == "tool_call":
                            tool_name = event["tool"]
                            tool_args = event.get("args", {})
                            tool_events.append({"name": tool_name, "args": tool_args, "status": "running"})
                            lines = [f"⏳ **{te['name']}** `{json.dumps(te.get('args', {}), ensure_ascii=False)}`" if te["status"] == "running" else f"✅ **{te['name']}**" for te in tool_events]
                            tool_placeholder.markdown("  \n".join(lines))
                            status_placeholder.markdown(f"🔧 正在执行: {TOOL_NAME_CN.get(tool_name, tool_name)}...")

                        elif event.get("status") == "tool_result":
                            for te in tool_events:
                                if te["name"] == event["tool"] and te["status"] == "running":
                                    te["status"] = "done"
                                    te["result"] = event.get("result", "")
                                    break
                            lines = [f"⏳ **{te['name']}** `{json.dumps(te.get('args', {}), ensure_ascii=False)}`" if te["status"] == "running" else f"✅ **{te['name']}**" for te in tool_events]
                            tool_placeholder.markdown("  \n".join(lines))

                        elif "chunk" in event:
                            full_reply += event["chunk"]
                            status_placeholder.markdown("📝 正在生成分析报告...")

                        elif event.get("done"):
                            break

                    if not error_occurred and (full_reply or images or tool_events):
                        status_placeholder.empty()
                        tool_placeholder.empty()
                        _finalize_agent_response(full_reply, images, tool_events)
                    elif not error_occurred:
                        _rollback_user_message()
                        st.toast("Agent 返回了空响应，请重试。", icon="❌")

                except Exception as e:
                    if full_reply or images or tool_events:
                        status_placeholder.empty()
                        tool_placeholder.empty()
                        st.toast(_format_request_exception(e, "Agent 连接") + " 已保存部分结果。", icon="⚠️")
                        _finalize_agent_response(full_reply, images, tool_events)
                    else:
                        _rollback_user_message()
                        st.toast(_format_request_exception(e, "Agent 请求"), icon="❌")

        else:
            # ── Smart / Direct Mode ──
            api_messages = [{"role": "system", "content": system_prompt}]
            for msg in st.session_state.chat_messages:
                if msg["role"] == "tool":
                    continue
                api_messages.append(_message_for_llm(msg))

            data_was_attached = False
            if not st.session_state.chat_data_sent and df is not None:
                if is_smart:
                    if "session_id" in st.session_state:
                        summary_resp = get_summary(st.session_state.session_id)
                        if summary_resp:
                            data_summary = summary_resp.get("summary", "")
                        else:
                            data_summary = _local_data_summary(df)
                    else:
                        data_summary = _local_data_summary(df)
                    api_messages[-1]["content"] = f"{api_messages[-1]['content']}\n\n---\n\n{data_summary}"
                else:
                    data_text, direct_err = _direct_data_text(df)
                    if direct_err:
                        _rollback_user_message()
                        st.toast(direct_err, icon="❌")
                        st.stop()
                    api_messages[-1]["content"] = f"{api_messages[-1]['content']}\n\n--- Data (CSV) ---\n{data_text}"
                st.session_state.chat_data_sent = True
                data_was_attached = True

            with st.chat_message("assistant", avatar="🤖"):
                text_placeholder = st.empty()
                full_reply = ""
                cursor = "▌"

                try:
                    resp = chat_llm(api_base, api_key, model, api_messages)
                    if resp.status_code != 200:
                        _rollback_user_message(data_was_attached)
                        st.toast(
                            _format_response_error(
                                resp,
                                f"大模型请求失败（HTTP {resp.status_code}），请检查 API 配置或稍后重试。",
                            ),
                            icon="❌",
                        )
                        st.stop()

                    for event in _iter_sse_events(resp):
                        if "error" in event:
                            _rollback_user_message(data_was_attached)
                            st.toast(_format_llm_error(event["error"]), icon="❌")
                            st.stop()
                        if event.get("done"):
                            break
                        if "chunk" in event:
                            full_reply += event["chunk"]
                            text_placeholder.markdown(full_reply + cursor)

                    if full_reply:
                        st.session_state.chat_messages.append({"role": "assistant", "content": full_reply, "images": []})
                        text_placeholder.markdown(full_reply)
                        st.rerun()
                    else:
                        _rollback_user_message(data_was_attached)
                        st.toast("LLM 返回了空响应，请重试。", icon="❌")

                except Exception as e:
                    if full_reply:
                        text_placeholder.empty()
                        st.session_state.chat_messages.append({"role": "assistant", "content": full_reply, "images": []})
                        st.toast(_format_request_exception(e, "大模型连接") + " 已保存部分结果。", icon="⚠️")
                        st.rerun()
                    else:
                        _rollback_user_message(data_was_attached)
                        st.toast(_format_request_exception(e, "大模型请求"), icon="❌")

st.divider()
with st.expander("📖 使用说明", expanded=False):
    st.markdown("""
    **智能模式（推荐）：**
    1. 上传数据 → 填写 API 配置 → 开始对话
    2. 首条消息自动附带数据摘要，不发送原始数据
    3. AI 推荐使用平台的哪些分析工具及关注点

    **直接模式：**
    1. 上传数据 → 填写 API 配置 → 开始对话
    2. 首条消息附带完整原始数据作为上下文

    **Agent 模式（推荐）：**
    1. 确保数据已同步到后端
    2. 填写 API 配置（需使用支持 function calling 的模型）
    3. 直接用自然语言告诉 AI 想分析什么
    4. AI 会自动调用平台工具（回归/分类/聚类/画图/代码执行）执行分析
    5. 图表和代码输出实时内嵌显示在对话中
    6. 实时查看工具执行进度，最终获得完整的分析报告

    **多轮对话：** 对话历史自动保存，每次发送都会带上完整上下文。切换分析模式会清空对话。
    """)
