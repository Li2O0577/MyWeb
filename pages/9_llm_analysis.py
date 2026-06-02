"""LLM analysis page — Flask backend proxies chat completions with SSE streaming."""
import streamlit as st
import pandas as pd
import numpy as np
import json
import base64
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


def _render_image(img_event):
    """Render a base64-encoded image from an SSE image event."""
    try:
        st.image(base64.b64decode(img_event["base64"]),
                 caption=img_event.get("title", ""),
                 use_container_width=True)
    except Exception:
        st.toast("图片解码失败，已跳过。", icon="⚠️")


def _iter_sse_events(response):
    """Yield parsed JSON events from an SSE streaming response."""
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        try:
            yield json.loads(line[6:])
        except json.JSONDecodeError:
            continue


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
st.divider()
st.subheader("🔀 分析模式")
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
    if not sid:
        st.warning("⚠️ Agent 模式需要后端 session。请先在「📊 数据加载」页面上传数据到后端，或点击上方的「重新同步当前数据到后端」按钮。")
        st.stop()

# ── API config ──
st.divider()
st.subheader("⚙️ API 配置")
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    api_base = st.text_input("API Base URL", value="https://api.openai.com/v1", placeholder="https://api.openai.com/v1")
with col2:
    api_key = st.text_input("API Key", type="password", placeholder="sk-...")
with col3:
    model = st.text_input("Model", value="gpt-4o", placeholder="gpt-4o / deepseek-chat / qwen-plus")

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
st.divider()
col_title, col_clear = st.columns([4, 1])
with col_title:
    st.subheader("💬 对话")
with col_clear:
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.chat_data_sent = False
        st.rerun()

# Render chat history
for msg in st.session_state.chat_messages:
    role = msg["role"]
    if role == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif role == "tool":
        with st.chat_message("assistant", avatar="🔧"):
            tool_name = msg.get("tool_name", "Tool")
            with st.expander(f"🔧 {tool_name}", expanded=False):
                st.code(msg["content"], language=None)
    elif role == "assistant":
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["content"])
            for img in msg.get("images", []):
                _render_image(img)


# ── Chat input ──
user_input = st.chat_input(
    placeholder="输入你的问题或分析需求..." if not is_agent else "告诉 AI 你想做什么分析，例如：「帮我分析这份数据，看看哪些因素影响价格」"
)

if user_input:
    if not api_key:
        st.toast("请输入 API Key。", icon="❌")
    elif not model:
        st.toast("请输入模型名称。", icon="❌")
    elif not user_input.strip():
        st.toast("请输入消息。", icon="❌")
    else:
        text = user_input.strip()
        st.session_state.chat_messages.append({"role": "user", "content": text})

        with st.chat_message("user"):
            st.markdown(text)

        if is_agent:
            # ── Agent Mode ──
            agent_messages = []
            for msg in st.session_state.chat_messages:
                if msg["role"] != "tool":
                    agent_messages.append({"role": msg["role"], "content": msg["content"]})

            sid = st.session_state.get("session_id", "")

            with st.chat_message("assistant", avatar="🤖"):
                text_placeholder = st.empty()
                tool_placeholder = st.empty()
                img_status_placeholder = st.empty()

                tool_events = []
                full_reply = ""
                images = []
                cursor = "▌"
                error_occurred = False

                try:
                    resp = agent_chat_llm(sid, api_base, api_key, model, agent_messages)
                    if resp.status_code != 200:
                        _rollback_user_message()
                        tool_placeholder.empty()
                        img_status_placeholder.empty()
                        st.toast(f"大模型分析请求失败（HTTP {resp.status_code}），请检查 API 配置或稍后重试。", icon="❌")
                        st.stop()

                    for event in _iter_sse_events(resp):
                        if "error" in event:
                            _rollback_user_message()
                            err = event["error"]
                            code = err.get("code", "ERROR")
                            message = err.get("message", "请求失败")
                            st.toast(f"Agent 错误 [{code}]: {message}", icon="❌")
                            error_occurred = True
                            break

                        if event.get("status") == "thinking":
                            pass

                        elif event.get("status") == "image":
                            images.append(event)
                            _render_image(event)
                            img_status_placeholder.caption(f"📊 已生成 {len(images)} 张图表")

                        elif event.get("status") == "tool_call":
                            tool_name = event["tool"]
                            tool_args = event.get("args", {})
                            tool_events.append({"name": tool_name, "args": tool_args, "status": "running"})
                            lines = [f"⏳ **{te['name']}** `{json.dumps(te.get('args', {}), ensure_ascii=False)}`" if te["status"] == "running" else f"✅ **{te['name']}**" for te in tool_events]
                            tool_placeholder.markdown("  \n".join(lines))

                        elif event.get("status") == "tool_result":
                            for te in tool_events:
                                if te["name"] == event["tool"] and te["status"] == "running":
                                    te["status"] = "done"
                                    break
                            lines = [f"⏳ **{te['name']}** `{json.dumps(te.get('args', {}), ensure_ascii=False)}`" if te["status"] == "running" else f"✅ **{te['name']}**" for te in tool_events]
                            tool_placeholder.markdown("  \n".join(lines))

                        elif "chunk" in event:
                            full_reply += event["chunk"]
                            text_placeholder.markdown(full_reply + cursor)

                        elif event.get("done"):
                            break

                    if full_reply or images or tool_events:
                        text_placeholder.markdown(full_reply)
                        tool_placeholder.empty()
                        img_status_placeholder.empty()

                        msg_record = {"role": "assistant", "content": full_reply, "images": images}
                        st.session_state.chat_messages.append(msg_record)
                        for te in tool_events:
                            short_info = f"参数: {json.dumps(te.get('args', {}), ensure_ascii=False)}"
                            st.session_state.chat_messages.append({
                                "role": "tool",
                                "tool_name": te["name"],
                                "content": short_info
                            })
                        st.rerun()
                    elif not error_occurred:
                        _rollback_user_message()
                        st.toast("Agent 返回了空响应，请重试。", icon="❌")

                except Exception as e:
                    if full_reply or images or tool_events:
                        text_placeholder.empty()
                        tool_placeholder.empty()
                        img_status_placeholder.empty()
                        msg_record = {"role": "assistant", "content": full_reply, "images": images}
                        st.session_state.chat_messages.append(msg_record)
                        for te in tool_events:
                            st.session_state.chat_messages.append({
                                "role": "tool",
                                "tool_name": te["name"],
                                "content": f"参数: {json.dumps(te.get('args', {}), ensure_ascii=False)}"
                            })
                        st.toast(f"Agent 连接中断（已保存部分结果）: {e}", icon="⚠️")
                        st.rerun()
                    else:
                        _rollback_user_message()
                        st.toast(f"Agent 错误: {e}", icon="❌")

        else:
            # ── Smart / Direct Mode ──
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

            with st.chat_message("assistant", avatar="🤖"):
                text_placeholder = st.empty()
                full_reply = ""
                cursor = "▌"

                try:
                    resp = chat_llm(api_base, api_key, model, api_messages)
                    if resp.status_code != 200:
                        _rollback_user_message(data_was_attached)
                        st.toast(f"大模型请求失败（HTTP {resp.status_code}），请检查 API 配置或稍后重试。", icon="❌")
                        st.stop()

                    for event in _iter_sse_events(resp):
                        if "error" in event:
                            _rollback_user_message(data_was_attached)
                            err = event["error"]
                            if isinstance(err, dict):
                                code = err.get("code", "ERROR")
                                message = err.get("message", "请求失败")
                                detail = err.get("detail", "")
                                suffix = f" ({detail})" if detail and detail != message else ""
                                st.toast(f"请求没有完成：{message}{suffix}", icon="❌")
                            else:
                                st.toast(f"请求没有完成：{err}", icon="❌")
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
                        st.toast(f"连接中断（已保存部分结果）: {e}", icon="⚠️")
                        st.rerun()
                    else:
                        _rollback_user_message(data_was_attached)
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

    **Agent 模式（推荐）：**
    1. 确保数据已同步到后端
    2. 填写 API 配置（需使用支持 function calling 的模型）
    3. 直接用自然语言告诉 AI 想分析什么
    4. AI 会自动调用平台工具（回归/分类/聚类/画图/代码执行）执行分析
    5. 图表和代码输出实时内嵌显示在对话中
    6. 实时查看工具执行进度，最终获得完整的分析报告

    **多轮对话：** 对话历史自动保存，每次发送都会带上完整上下文。切换分析模式会清空对话。
    """)
