"""LLM analysis page — Flask backend proxies chat completions with SSE streaming."""
import streamlit as st
import pandas as pd
import numpy as np
import json
from pages._prepare import render_sidebar, data_uploader
from pages._api import chat_llm, get_summary, backend_status_badge, render_backend_sync_panel

st.set_page_config(page_title="大模型分析", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>[data-testid="stSidebarNav"] {display: none;}</style>""", unsafe_allow_html=True)
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
    "📄 直接模式 — 发送原始数据给 AI（消耗更多 Token）"
], index=0, help="智能：低 Token 成本。直接：发送完整数据。")
is_smart = mode.startswith("📊")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
    st.session_state.chat_data_sent = False
    st.session_state.chat_mode = is_smart
elif st.session_state.chat_mode != is_smart:
    st.session_state.chat_messages = []
    st.session_state.chat_data_sent = False
    st.session_state.chat_mode = is_smart
    st.rerun()

# API config
st.divider()
st.subheader("⚙️ API 配置")
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    api_base = st.text_input("API Base URL", value="https://api.openai.com/v1", placeholder="https://api.openai.com/v1")
with col2:
    api_key = st.text_input("API Key", type="password", placeholder="sk-...")
with col3:
    model = st.text_input("Model", value="gpt-4o", placeholder="gpt-4o / deepseek-chat / qwen-plus")
st.caption("支持所有兼容 OpenAI API 格式的服务（DeepSeek、通义千问、智谱等）")

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
.chat-box { height: 420px; overflow-y: auto; border: 1px solid #444; border-radius: 12px; padding: 16px; background-color: #0e1117; margin-bottom: 8px; }
.chat-box .empty-hint { color: #555; text-align: center; padding-top: 180px; font-size: 0.95em; }
.chat-msg { margin-bottom: 14px; }
.chat-msg .role { font-weight: 600; font-size: 0.85em; margin-bottom: 4px; }
.chat-msg.user .role { color: #58a6ff; }
.chat-msg.user .body { background-color: #0d2137; border-left: 3px solid #58a6ff; padding: 10px 14px; border-radius: 0 10px 10px 0; color: #c9d1d9; line-height: 1.6; }
.chat-msg.assistant .role { color: #7ee787; }
.chat-msg.assistant .body { background-color: #0d1a14; border-left: 3px solid #7ee787; padding: 10px 14px; border-radius: 0 10px 10px 0; color: #c9d1d9; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

msg_count = len(st.session_state.chat_messages)
chat_html = '<div class="chat-box">'
if not st.session_state.chat_messages:
    chat_html += '<div class="empty-hint">👋 上传数据后开始对话，我会根据数据为你推荐分析方向</div>'
else:
    for msg in st.session_state.chat_messages:
        role_class = "user" if msg["role"] == "user" else "assistant"
        label = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
        body = msg["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        chat_html += f'<div class="chat-msg {role_class}"><div class="role">{label}</div><div class="body">{body}</div></div>'
chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

# Input
col_input, col_send = st.columns([6, 1])
with col_input:
    user_input = st.text_area("输入消息", key=f"chat_input_{msg_count}", label_visibility="collapsed", placeholder="输入你的问题或分析需求...", height=68)
with col_send:
    st.write(""); st.write("")
    send_btn = st.button("🚀 发送", use_container_width=True, type="primary")

if send_btn:
    if not api_key:
        st.error("请输入 API Key。")
    elif not model:
        st.error("请输入模型名称。")
    elif not user_input.strip():
        st.error("请输入消息。")
    else:
        st.session_state.chat_messages.append({"role": "user", "content": user_input.strip()})

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
                    st.error(f"API 错误 [{resp.status_code}]")
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
                                st.error(f"API 错误: {code}: {message}{suffix}")
                            else:
                                st.error(f"API 错误: {err}")
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
                    st.error("LLM 返回了空响应，请重试。")

            except Exception as e:
                st.session_state.chat_messages.pop()
                if data_was_attached:
                    st.session_state.chat_data_sent = False
                st.error(f"未知错误：{e}")

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

    **多轮对话：** 对话历史自动保存，每次发送都会带上完整上下文。切换分析模式会清空对话。
    """)
