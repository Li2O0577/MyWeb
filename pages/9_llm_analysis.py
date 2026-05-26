"""LLM analysis page — Flask backend proxies chat completions with SSE streaming."""
import streamlit as st
import pandas as pd
import numpy as np
import json
from pages._prepare import render_sidebar, data_uploader
from pages._api import chat_llm, agent_chat_llm, get_summary, backend_status_badge, render_backend_sync_panel

st.set_page_config(page_title="大模型分析", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>[data-testid="stickyNav"] {display: none;}</style>""", unsafe_allow_html=True)
render_sidebar("pages/9_llm_analysis.py")
st.title("🤖 大模型分析")

backend_status_badge()

df = data_uploader(upload_to_backend=False)

if df is None:
    st.info("👋 请先在「📊 数据加载」页面上传数据，然后再使用大模型分析功能。")
    st.stop()

render_backend_sync_panel(df)


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

# Analysis mode
st.divider()
st.subheader("🔀 分析模式")
mode = st.radio("选择与 LLM 的交互方式：", [
    "📊 智能模式 — 发送数据摘要，让 AI 推荐平台分析工具",
    "📄 直接模式 — 发送原始数据给 AI（消耗更多 Token）",
    "🤖 Agent 模式 — AI 自主调用平台工具执行完整分析（需要后端 session）"
], index=0, help="智能：低 Token 成本。直接：发送完整数据。Agent：AI 自主训练模型、聚类、相关性分析。")
is_smart = mode.startswith("📊")
is_direct = mode.startswith("📄")
is_agent = mode.startswith("🤖")

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
    if not sid:
        st.warning("⚠️ Agent 模式需要后端 session。请先在「📊 数据加载」页面上传数据到后端，或点击上方的「重新同步当前数据到后端」按钮。")
        st.stop()

# API config
st.divider()
st.subheader("⚙️ API 配置")
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    api_base = st.text_input("API Base URL", value="https://api.openai.com/v1", placeholder="https://api.openai.com/v1")
with col2:
    api_key = st.text_input("API Key", type="password", placeholder="sk-...")
with col3:
    model = st.text_input("Model", value="gpt-4o", placeholder="gpt-4o / deepseek-chat / qwen-plus", help="Agent 模式建议使用支持 function calling 的模型（gpt-4o, deepseek-chat 等）")
st.caption("支持所有兼容 OpenAI API 格式的服务（DeepSeek、通义千问、智谱等）")

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

        **使用方式：** 直接用自然语言告诉 AI 你想分析什么，例如：
        - "帮我完整分析这份数据"
        - "预测房价并评估模型效果"
        - "看看数据里有哪些自然的群体"

        AI 会自动选择工具、执行分析、给出结论。你可以在过程中看到每个工具的执行进度。
        """)
else:
    with st.expander("📝 系统提示词（可编辑）", expanded=False):
        default_system = SKILL_INFO if is_smart else "You are a professional data analyst. Please analyze the given data and provide insightful findings."
        system_prompt = st.text_area("系统提示词", value=default_system, height=200 if is_smart else 100)

# Chat display
st.divider()
col_title, col_clear = st.columns([4, 1])
with col_title:
    st.subheader("💬 对话")
with col_clear:
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.chat_data_sent = False
        st.rerun()

st.markdown("""
<style>
.chat-box { height: 70vh; min-height: 500px; overflow-y: auto; border: 1px solid #444; border-radius: 12px; padding: 16px; background-color: #0e1117; margin-bottom: 8px; }
.chat-box .empty-hint { color: #555; text-align: center; padding-top: 180px; font-size: 0.95em; }
.chat-msg { margin-bottom: 14px; }
.chat-msg .role { font-weight: 600; font-size: 0.85em; margin-bottom: 4px; }
.chat-msg.user .role { color: #58a6ff; }
.chat-msg.user .body { background-color: #0d2137; border-left: 3px solid #58a6ff; padding: 10px 14px; border-radius: 0 10px 10px 0; color: #c9d1d9; line-height: 1.6; }
.chat-msg.assistant .role { color: #7ee787; }
.chat-msg.assistant .body { background-color: #0d1a14; border-left: 3px solid #7ee787; padding: 10px 14px; border-radius: 0 10px 10px 0; color: #c9d1d9; line-height: 1.6; }
.chat-msg.tool .role { color: #d2a8ff; }
.chat-msg.tool .body { background-color: #1a0d2e; border-left: 3px solid #d2a8ff; padding: 8px 14px; border-radius: 0 10px 10px 0; color: #c9d1d9; font-size: 0.9em; font-family: monospace; white-space: pre-wrap; max-height: 200px; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)

msg_count = len(st.session_state.chat_messages)
chat_html = '<div class="chat-box">'
if not st.session_state.chat_messages:
    if is_agent:
        chat_html += '<div class="empty-hint">🤖 Agent 模式：直接告诉 AI 你想分析什么，它会自动调用平台工具执行。例如：「帮我完整分析这份数据」</div>'
    else:
        chat_html += '<div class="empty-hint">👋 上传数据后开始对话，我会根据数据为你推荐分析方向</div>'
else:
    for msg in st.session_state.chat_messages:
        role = msg["role"]
        if role == "user":
            role_class = "user"
            label = "🧑 You"
        elif role == "tool":
            role_class = "tool"
            label = f"🔧 {msg.get('tool_name', 'Tool')}"
        else:
            role_class = "assistant"
            label = "🤖 Assistant"
        body = msg["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        chat_html += f'<div class="chat-msg {role_class}"><div class="role">{label}</div><div class="body">{body}</div></div>'
chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

# Input
col_input, col_send = st.columns([6, 1])
with col_input:
    placeholder = "输入你的问题或分析需求..." if not is_agent else "告诉 AI 你想做什么分析，例如：「帮我分析这份数据，看看哪些因素影响价格」"
    user_input = st.text_area("输入消息", key=f"chat_input_{msg_count}", label_visibility="collapsed", placeholder=placeholder, height=68)
with col_send:
    st.write(""); st.write("")
    send_btn = st.button("🚀 发送", use_container_width=True, type="primary")

if send_btn:
    if not api_key:
        st.toast("请输入 API Key。", icon="❌")
    elif not model:
        st.toast("请输入模型名称。", icon="❌")
    elif not user_input.strip():
        st.toast("请输入消息。", icon="❌")
    else:
        st.session_state.chat_messages.append({"role": "user", "content": user_input.strip()})

        if is_agent:
            # ── Agent Mode ──
            agent_messages = []
            for msg in st.session_state.chat_messages:
                if msg["role"] != "tool":
                    agent_messages.append({"role": msg["role"], "content": msg["content"]})

            sid = st.session_state.get("session_id", "")

            # Real-time display area
            progress_container = st.container()
            reply_container = st.container()

            tool_events = []
            full_reply = ""
            cursor = "▌"
            error_occurred = False

            with progress_container:
                progress_placeholder = st.empty()

            with reply_container:
                st.markdown("**🤖 Assistant**")
                reply_placeholder = st.empty()

            try:
                resp = agent_chat_llm(sid, api_base, api_key, model, agent_messages)
                if resp.status_code != 200:
                    st.session_state.chat_messages.pop()
                    st.toast(f"API 错误 [{resp.status_code}]", icon="❌")
                    st.stop()

                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if "error" in event:
                        st.session_state.chat_messages.pop()
                        err = event["error"]
                        code = err.get("code", "ERROR")
                        message = err.get("message", "请求失败")
                        st.toast(f"Agent 错误 [{code}]: {message}", icon="❌")
                        error_occurred = True
                        break

                    if event.get("status") == "thinking":
                        pass  # skip

                    elif event.get("status") == "tool_call":
                        tool_name = event["tool"]
                        tool_args = event.get("args", {})
                        tool_events.append({"name": tool_name, "args": tool_args, "status": "running"})
                        # Render tool progress
                        lines_html = ""
                        for te in tool_events:
                            icon = "⏳" if te["status"] == "running" else "✅"
                            args_str = json.dumps(te.get("args", {}), ensure_ascii=False)
                            lines_html += f'<div style="color:#d2a8ff;font-size:0.85em;margin:2px 0;">{icon} <b>{te["name"]}</b> <span style="color:#8b949e;">{args_str}</span></div>'
                        progress_placeholder.markdown(lines_html, unsafe_allow_html=True)

                    elif event.get("status") == "tool_result":
                        for te in tool_events:
                            if te["name"] == event["tool"] and te["status"] == "running":
                                te["status"] = "done"
                                te["result"] = event.get("result", "")
                                break
                        # Re-render
                        lines_html = ""
                        for te in tool_events:
                            icon = "⏳" if te["status"] == "running" else "✅"
                            args_str = json.dumps(te.get("args", {}), ensure_ascii=False)
                            lines_html += f'<div style="color:#d2a8ff;font-size:0.85em;margin:2px 0;">{icon} <b>{te["name"]}</b> <span style="color:#8b949e;">{args_str}</span></div>'
                        progress_placeholder.markdown(lines_html, unsafe_allow_html=True)

                    elif "chunk" in event:
                        full_reply += event["chunk"]
                        reply_placeholder.markdown(full_reply + cursor)

                    elif event.get("done"):
                        break

                if full_reply:
                    st.session_state.chat_messages.append({"role": "assistant", "content": full_reply})
                    # Add tool events as chat messages for history context
                    for te in tool_events:
                        short_result = te.get("result", "")
                        if len(short_result) > 500:
                            short_result = short_result[:500] + "..."
                        st.session_state.chat_messages.append({
                            "role": "tool",
                            "tool_name": te["name"],
                            "content": f"参数: {json.dumps(te.get('args', {}), ensure_ascii=False)}\n结果: {short_result}"
                        })
                    progress_placeholder.empty()
                    reply_placeholder.empty()
                    st.rerun()
                elif not error_occurred:
                    st.session_state.chat_messages.pop()
                    st.toast("Agent 返回了空响应，请重试。", icon="❌")

            except Exception as e:
                st.session_state.chat_messages.pop()
                st.toast(f"Agent 错误: {e}", icon="❌")

        else:
            # ── Smart / Direct Mode (existing logic) ──
            api_messages = [{"role": "system", "content": system_prompt}]
            for msg in st.session_state.chat_messages:
                api_messages.append(dict(msg))

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
                    data_text = df.to_csv(index=False)
                    api_messages[-1]["content"] = f"{api_messages[-1]['content']}\n\n--- Data (CSV) ---\n{data_text}"
                st.session_state.chat_data_sent = True
                data_was_attached = True

            # Stream via Flask backend SSE
            stream_container = st.container()
            with stream_container:
                st.markdown("**🤖 Assistant**")
                stream_placeholder = st.empty()
                full_reply = ""
                cursor = "▌"

                try:
                    resp = chat_llm(api_base, api_key, model, api_messages)
                    if resp.status_code != 200:
                        st.session_state.chat_messages.pop()
                        if data_was_attached:
                            st.session_state.chat_data_sent = False
                        st.toast(f"API 错误 [{resp.status_code}]", icon="❌")
                        st.stop()

                    for line in resp.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        try:
                            event = json.loads(data_str)
                            if "error" in event:
                                st.session_state.chat_messages.pop()
                                if data_was_attached:
                                    st.session_state.chat_data_sent = False
                                err = event["error"]
                                if isinstance(err, dict):
                                    code = err.get("code", "ERROR")
                                    message = err.get("message", "请求失败")
                                    detail = err.get("detail", "")
                                    suffix = f" ({detail})" if detail and detail != message else ""
                                    st.toast(f"API 错误: {code}: {message}{suffix}", icon="❌")
                                else:
                                    st.toast(f"API 错误: {err}", icon="❌")
                                st.stop()
                            if event.get("done"):
                                break
                            if "chunk" in event:
                                full_reply += event["chunk"]
                                stream_placeholder.markdown(full_reply + cursor)
                        except json.JSONDecodeError:
                            continue

                    if full_reply:
                        st.session_state.chat_messages.append({"role": "assistant", "content": full_reply})
                        stream_placeholder.empty()
                        st.rerun()
                    else:
                        st.session_state.chat_messages.pop()
                        if data_was_attached:
                            st.session_state.chat_data_sent = False
                        st.toast("LLM 返回了空响应，请重试。", icon="❌")

                except Exception as e:
                    st.session_state.chat_messages.pop()
                    if data_was_attached:
                        st.session_state.chat_data_sent = False
                    st.toast(f"未知错误：{e}", icon="❌")

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

    **Agent 模式：**
    1. 确保数据已同步到后端
    2. 填写 API 配置（需使用支持 function calling 的模型）
    3. 直接用自然语言告诉 AI 想分析什么
    4. AI 会自动调用平台工具（回归/分类/聚类/相关性分析）执行分析
    5. 实时查看工具执行进度，最终获得完整的分析报告

    **多轮对话：** 对话历史自动保存，每次发送都会带上完整上下文。切换分析模式会清空对话。
    """)
