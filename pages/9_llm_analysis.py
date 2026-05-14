import streamlit as st
import pandas as pd
import numpy as np
import requests
from pages._prepare import render_sidebar, data_uploader

# 1. 配置页面
st.set_page_config(page_title="大模型分析", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)
render_sidebar("pages/9_llm_analysis.py")
st.title("🤖 大模型分析")

# 2. 数据上传
df = data_uploader()

# ── Skill 信息 ──
SKILL_INFO = """
## Available Analysis Skills in This Platform

You are part of a data analysis platform. The following built-in modules are available on other pages. Do NOT try to execute them yourself — your job is to recommend which ones the user should use and what to look for.

| # | Page | What It Does |
|---|------|---------------|
| 1 | Data Load | Upload CSV/Excel, basic preview |
| 2 | Data Visualization | Scatter, line, bar, histogram, box plot, density, correlation heatmap, pie, pair plot |
| 3 | Data Processing | Drop columns/rows, rename, type conversion, statistics (mean/std/min/max), StandardScaler, MinMaxScaler, Gaussian noise, Label/OneHot encoding, custom formula columns, PCA |
| 4 | Regression | Train PyTorch MLP for regression |
| 5 | Classification | Train PyTorch MLP for binary/multi-class classification |
| 6 | DIY MLP | Customize MLP architecture (layers, neurons, activation, learning rate) |
| 7 | Decision Tree | Train decision tree classifier + visualize tree + export rules |
| 8 | K-means | K-means clustering + elbow method + PCA visualization |
| 9 | LLM Analysis | THIS page — you are providing analysis recommendations now |

**Your Task:**
1. Review the data summary below.
2. Give a concise data overview.
3. Recommend 3-5 specific analysis actions, mapping each to the appropriate page above.
4. For each recommendation, explain WHY it's useful.

Respond in clear, well-structured Markdown.
"""

# ── 辅助函数 ──

def build_data_summary(df):
    lines = []
    lines.append("## Data Summary")
    lines.append(f"- Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    lines.append(f"- Column names: {', '.join(str(c) for c in df.columns)}")
    lines.append("")

    lines.append("### Column Types")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        lines.append(f"- {dtype}: {count} columns")
    lines.append("")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        lines.append("### Numeric Columns Statistics")
        lines.append(df[numeric_cols].describe().to_string())
        lines.append("")

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            corr_upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            corr_pairs = corr_upper.stack().reset_index()
            corr_pairs.columns = ["Col A", "Col B", "Correlation"]
            corr_pairs["AbsCorr"] = corr_pairs["Correlation"].abs()
            corr_top = corr_pairs.sort_values("AbsCorr", ascending=False).head(10)
            if len(corr_top) > 0:
                lines.append("### Top Correlations")
                for _, row in corr_top.iterrows():
                    lines.append(f"- {row['Col A']} vs {row['Col B']}: {row['Correlation']:.3f}")
                lines.append("")

    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        lines.append("### Categorical Columns (unique values / total)")
        for col in cat_cols:
            n_unique = df[col].nunique()
            n_total = df[col].notna().sum()
            lines.append(f"- **{col}**: {n_unique} unique / {n_total} non-null")
        lines.append("")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        lines.append("### Missing Values")
        for col, cnt in missing.items():
            lines.append(f"- {col}: {cnt} ({cnt/len(df)*100:.1f}%)")
        lines.append("")
    else:
        lines.append("### Missing Values: None")
        lines.append("")

    lines.append("### First 5 Rows (Preview)")
    lines.append(df.head(5).to_string())

    return "\n".join(lines)


# 3. 分析模式选择
st.divider()
st.subheader("🔀 分析模式")

mode = st.radio(
    "选择与 LLM 的交互方式：",
    [
        "📊 智能模式 — 发送数据摘要，让 AI 推荐平台分析工具",
        "📄 直接模式 — 发送原始数据给 AI（消耗更多 Token）"
    ],
    index=0,
    help="智能：低 Token 成本，AI 推荐使用平台上的分析工具。直接：发送完整数据，AI 可做自定义分析。"
)
is_smart = mode.startswith("📊")

# 4. 初始化 session_state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
    st.session_state.chat_data_sent = False
    st.session_state.chat_mode = is_smart
elif st.session_state.chat_mode != is_smart:
    st.session_state.chat_messages = []
    st.session_state.chat_data_sent = False
    st.session_state.chat_mode = is_smart
    st.rerun()

# 5. API 配置
st.divider()
st.subheader("⚙️ API 配置")

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    api_base = st.text_input(
        "API Base URL",
        value="https://api.openai.com/v1",
        placeholder="https://api.openai.com/v1",
        help="支持 OpenAI 兼容接口（DeepSeek、Qwen 等均可）"
    )
with col2:
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-...",
        help="密钥仅保存在当前会话中"
    )
with col3:
    model = st.text_input(
        "Model",
        value="gpt-4o",
        placeholder="gpt-4o / deepseek-chat / qwen-plus"
    )

st.caption("支持所有兼容 OpenAI API 格式的服务（DeepSeek、通义千问、智谱等）")

# 系统提示词（折叠）
with st.expander("📝 系统提示词（可编辑）", expanded=False):
    default_system = SKILL_INFO if is_smart else "You are a professional data analyst. Please analyze the given data and provide insightful findings."
    system_prompt = st.text_area(
        "系统提示词",
        value=default_system,
        height=200 if is_smart else 100
    )
    if df is not None and is_smart:
        st.caption("💡 智能模式下，首条消息会自动附带数据摘要（统计量、相关性、缺失值等）")

# 6. 对话区域
st.divider()
col_title, col_clear = st.columns([4, 1])
with col_title:
    st.subheader("💬 对话")
with col_clear:
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.chat_data_sent = False
        st.rerun()

# 固定高度可滑动对话框
st.markdown("""
<style>
.chat-box {
    height: 420px;
    overflow-y: auto;
    border: 1px solid #444;
    border-radius: 12px;
    padding: 16px;
    background-color: #0e1117;
    margin-bottom: 8px;
}
.chat-box .empty-hint {
    color: #555;
    text-align: center;
    padding-top: 180px;
    font-size: 0.95em;
}
.chat-msg {
    margin-bottom: 14px;
}
.chat-msg .role {
    font-weight: 600;
    font-size: 0.85em;
    margin-bottom: 4px;
}
.chat-msg.user .role { color: #58a6ff; }
.chat-msg.user .body {
    background-color: #0d2137;
    border-left: 3px solid #58a6ff;
    padding: 10px 14px;
    border-radius: 0 10px 10px 0;
    color: #c9d1d9;
    line-height: 1.6;
}
.chat-msg.assistant .role { color: #7ee787; }
.chat-msg.assistant .body {
    background-color: #0d1a14;
    border-left: 3px solid #7ee787;
    padding: 10px 14px;
    border-radius: 0 10px 10px 0;
    color: #c9d1d9;
    line-height: 1.6;
}
.chat-msg .body p { margin: 0 0 6px 0; }
.chat-msg .body p:last-child { margin-bottom: 0; }
.chat-msg .body ul, .chat-msg .body ol { margin: 4px 0; padding-left: 20px; }
.chat-msg .body li { margin-bottom: 2px; }
.chat-msg .body code {
    background-color: #1a1a2e;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9em;
}
.chat-msg .body pre {
    background-color: #1a1a2e;
    padding: 10px;
    border-radius: 6px;
    overflow-x: auto;
}
.chat-msg .body table {
    border-collapse: collapse;
    width: 100%;
    margin: 8px 0;
}
.chat-msg .body th, .chat-msg .body td {
    border: 1px solid #30363d;
    padding: 6px 10px;
    text-align: left;
}
.chat-msg .body th { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)

msg_count = len(st.session_state.chat_messages)

chat_html = '<div class="chat-box">'
if not st.session_state.chat_messages:
    hint = "👋 上传数据后开始对话，我会根据数据为你推荐分析方向"
    chat_html += f'<div class="empty-hint">{hint}</div>'
else:
    for msg in st.session_state.chat_messages:
        role_class = "user" if msg["role"] == "user" else "assistant"
        label = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
        body = msg["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        chat_html += f'<div class="chat-msg {role_class}"><div class="role">{label}</div><div class="body">{body}</div></div>'
chat_html += '</div>'

st.markdown(chat_html, unsafe_allow_html=True)

# 7. 输入区域
col_input, col_send = st.columns([6, 1])
with col_input:
    user_input = st.text_area(
        "输入消息",
        key=f"chat_input_{msg_count}",
        label_visibility="collapsed",
        placeholder="输入你的问题或分析需求...  (Shift+Enter 换行)",
        height=68
    )
with col_send:
    st.write("")
    st.write("")
    send_btn = st.button("🚀 发送", use_container_width=True, type="primary")

# 8. 发送逻辑
if send_btn:
    if not api_key:
        st.error("请输入 API Key。")
    elif not model:
        st.error("请输入模型名称。")
    elif not user_input.strip():
        st.error("请输入消息。")
    else:
        st.session_state.chat_messages.append({"role": "user", "content": user_input.strip()})

        with st.spinner("等待 LLM 回复中..."):
            try:
                api_messages = [{"role": "system", "content": system_prompt}]

                for msg in st.session_state.chat_messages:
                    api_messages.append(dict(msg))

                if not st.session_state.chat_data_sent and df is not None:
                    if is_smart:
                        data_summary = build_data_summary(df)
                        api_messages[-1]["content"] = f"{api_messages[-1]['content']}\n\n---\n\n{data_summary}"
                    else:
                        data_text = df.to_csv(index=False)
                        api_messages[-1]["content"] = f"{api_messages[-1]['content']}\n\n--- Data (CSV) ---\n{data_text}"
                    st.session_state.chat_data_sent = True

                resp = requests.post(
                    f"{api_base.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": api_messages,
                        "temperature": 0.7
                    },
                    timeout=120
                )

                if resp.status_code == 200:
                    data = resp.json()
                    reply = data["choices"][0]["message"]["content"]
                    st.session_state.chat_messages.append({"role": "assistant", "content": reply})
                    st.rerun()
                else:
                    st.session_state.chat_messages.pop()
                    st.error(f"API 错误 [{resp.status_code}]: {resp.text}")

            except requests.exceptions.Timeout:
                st.session_state.chat_messages.pop()
                st.error("请求超时，请缩短提示词后重试。")
            except requests.exceptions.ConnectionError:
                st.session_state.chat_messages.pop()
                st.error(f"连接失败，请检查 API 地址：{api_base}")
            except Exception as e:
                st.session_state.chat_messages.pop()
                st.error(f"未知错误：{e}")

# 9. 使用说明
st.divider()
with st.expander("📖 使用说明", expanded=False):
    st.markdown("""
    **智能模式（推荐）：**
    1. 上传数据 → 填写 API 配置 → 开始对话
    2. 首条消息自动附带数据摘要（统计量、相关性、列信息），不发送原始数据
    3. AI 推荐使用平台的哪些分析工具及关注点
    4. 后续对话正常进行，不再重复发送数据摘要

    **直接模式：**
    1. 上传数据 → 填写 API 配置 → 开始对话
    2. 首条消息附带完整原始数据作为上下文
    3. 适合需要 AI 读取实际数值的自定义分析

    **多轮对话：**
    - 对话历史自动保存，每次发送都会带上完整上下文
    - 切换分析模式会清空对话
    - 点击"清空对话"按钮重置

    **支持的平台：**
    - OpenAI (GPT-4o, GPT-4 等)
    - DeepSeek
    - 通义千问 (Qwen)
    - 智谱 GLM (Zhipu)
    - 其他兼容 OpenAI API 的服务
    """)
