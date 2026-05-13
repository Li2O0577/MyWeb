import streamlit as st
import pandas as pd
import numpy as np
import requests
from pages._prepare import render_sidebar, data_uploader

# 1. 配置页面
st.set_page_config(page_title="LLM Analysis", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)
render_sidebar("pages/9_llm_analysis.py")
st.title("🤖 LLM Analysis")

# 2. 数据上传
df = data_uploader()

# ── Skill 信息（所有页面能力描述） ──
SKILL_INFO = """
## Available Analysis Skills in This Platform

You are part of a data analysis platform. The following built-in modules are available on other pages. Do NOT try to execute them yourself — your job is to recommend which ones the user should use and what to look for.

| # | Page | What It Does |
|---|------|---------------|
| 1 | Data Load | Upload CSV/Excel, basic preview |
| 2 | Data Visualization | Scatter, line, bar, histogram, box plot, density, correlation heatmap, pie, pair plot |
| 3 | Data Processing | Drop columns/rows, rename, type conversion, statistics (mean/std/min/max), StandardScaler, MinMaxScaler, Gaussian noise, Label/OneHot encoding, custom formula columns, PCA dimensionality reduction |
| 4 | Regression | Train PyTorch MLP for regression (predict continuous values) |
| 5 | Classification | Train PyTorch MLP for binary/multi-class classification |
| 6 | DIY MLP | Customize MLP architecture (layers, neurons, activation, learning rate) |
| 7 | Decision Tree | Train decision tree classifier + visualize tree + export rules |
| 8 | K-means | K-means clustering + elbow method + PCA visualization |
| 9 | LLM Analysis | THIS page — you are providing analysis recommendations now |

**Your Task:**
1. Review the data summary below.
2. Give a concise data overview: what the dataset seems to be about, key columns, data quality observations.
3. Recommend 3-5 specific analysis actions the user should take, mapping each to the appropriate page above.
4. For each recommendation, explain WHY it's useful for this dataset.

Respond in a clear, well-structured Markdown format. Use headings, bullet points, and tables where appropriate.
"""

# ── 辅助函数 ──

def build_data_summary(df):
    """生成数据摘要，只发送元信息不发送原始数据"""
    lines = []
    lines.append(f"## Data Summary")
    lines.append(f"- Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    lines.append(f"- Column names: {', '.join(str(c) for c in df.columns)}")
    lines.append("")

    # 数据类型分布
    lines.append("### Column Types")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        lines.append(f"- {dtype}: {count} columns")
    lines.append("")

    # 数值列统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        lines.append("### Numeric Columns Statistics")
        desc = df[numeric_cols].describe()
        lines.append(desc.to_string())
        lines.append("")

        # 相关性 Top-N（限制输出大小）
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            # 取上三角，按绝对值排序取 Top-10
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

    # 非数值列统计
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        lines.append("### Categorical Columns (unique values / total)")
        for col in cat_cols:
            n_unique = df[col].nunique()
            n_total = df[col].notna().sum()
            lines.append(f"- **{col}**: {n_unique} unique / {n_total} non-null")
        lines.append("")

    # 缺失值
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

    # 前 5 行预览
    lines.append("### First 5 Rows (Preview)")
    lines.append(df.head(5).to_string())

    return "\n".join(lines)


# 3. 分析模式选择
st.divider()
st.subheader("🔀 Analysis Mode")

mode = st.radio(
    "Choose how to interact with the LLM:",
    [
        "📊 Smart Mode — Send data summary, let AI recommend analyses using platform skills",
        "📄 Direct Mode — Send raw data directly to AI (higher token cost)"
    ],
    index=0,
    help="Smart: low token cost, AI recommends which platform tools to use. Direct: full data sent, AI can do custom analysis."
)

is_smart = mode.startswith("📊")

# 4. API 配置
st.divider()
st.subheader("⚙️ API Configuration")

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

# 5. Prompt 输入
st.divider()
st.subheader("💬 Prompt")

if is_smart:
    default_system = SKILL_INFO
    default_user = "Please analyze the data based on the summary above and recommend specific analysis steps using the platform's tools."
    user_help = "描述你对数据的具体问题，系统会自动附带数据摘要"
else:
    default_system = "You are a professional data analyst. Please analyze the given data and provide insightful findings."
    default_user = "Please analyze the following data and give me a summary:"
    user_help = "如果已上传数据，数据会自动追加到 prompt 后面"

col1, col2 = st.columns(2)
with col1:
    system_prompt = st.text_area(
        "System Prompt",
        value=default_system,
        height=200 if is_smart else 120
    )
with col2:
    user_prompt = st.text_area(
        "User Prompt",
        value=default_user,
        height=200 if is_smart else 120,
        help=user_help
    )

# Direct 模式专属：数据包含选项
if not is_smart:
    if df is not None:
        with st.expander("📋 Data Preview & Context Settings", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Data shape: {df.shape[0]} rows × {df.shape[1]} columns")

            col1, col2 = st.columns(2)
            with col1:
                max_rows = st.slider(
                    "包含数据行数（过多可能超出 token 限制）",
                    0, min(100, len(df)), min(20, len(df)),
                    help="0 表示不包含数据，仅发送 prompt"
                )
            with col2:
                col_subset = st.multiselect(
                    "选择包含的列（留空 = 全部）",
                    df.columns.tolist(),
                    default=[]
                )
    else:
        max_rows = 0
        col_subset = []

# 6. 发送请求
st.divider()
col_btn, col_info = st.columns([1, 4])
with col_btn:
    send_label = "🚀 Analyze (Smart)" if is_smart else "🚀 Send"
    send_btn = st.button(send_label, type="primary", use_container_width=True)
with col_info:
    if is_smart and df is not None:
        st.caption(f"💡 Smart mode: only data summary (~{len(df.columns)} cols × stats) will be sent, saving tokens")

if send_btn:
    if not api_key:
        st.error("Please enter your API Key.")
    elif not model:
        st.error("Please enter a model name.")
    else:
        messages = [{"role": "system", "content": system_prompt}]

        if is_smart and df is not None:
            # Smart 模式：发送数据摘要
            data_summary = build_data_summary(df)
            full_content = f"{user_prompt}\n\n---\n\n{data_summary}"
        elif not is_smart:
            # Direct 模式（原有逻辑）
            content_parts = [user_prompt]
            if df is not None and max_rows > 0:
                subset = df[col_subset] if col_subset else df
                subset = subset.head(max_rows)
                data_text = subset.to_csv(index=False)
                content_parts.append(f"\n\n--- Data (CSV) ---\n{data_text}")
            full_content = "\n".join(content_parts)
        else:
            full_content = user_prompt

        messages.append({"role": "user", "content": full_content})

        with st.spinner("Waiting for LLM response..."):
            try:
                resp = requests.post(
                    f"{api_base.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": 0.7
                    },
                    timeout=120
                )

                if resp.status_code == 200:
                    data = resp.json()
                    reply = data["choices"][0]["message"]["content"]

                    st.divider()
                    st.subheader("📝 Response")
                    st.markdown(reply)

                    usage = data.get("usage", {})
                    if usage:
                        with st.expander("📊 Token Usage", expanded=False):
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Prompt Tokens", usage.get("prompt_tokens", "N/A"))
                            c2.metric("Completion Tokens", usage.get("completion_tokens", "N/A"))
                            c3.metric("Total Tokens", usage.get("total_tokens", "N/A"))
                else:
                    st.error(f"API Error [{resp.status_code}]: {resp.text}")

            except requests.exceptions.Timeout:
                st.error("Request timeout. Please try with a shorter prompt.")
            except requests.exceptions.ConnectionError:
                st.error(f"Connection failed. Please check your API Base URL: {api_base}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# 7. 使用说明
st.divider()
with st.expander("📖 Instructions", expanded=False):
    st.markdown("""
    **Smart Mode (Recommended):**
    1. Upload your data
    2. Fill in your API configuration
    3. The AI receives only data summary (stats, correlations, column info) — no raw rows
    4. AI recommends which platform tools to use and what to look for
    5. Much lower token cost, results focused on actionable next steps

    **Direct Mode:**
    1. Upload your data
    2. Fill in your API configuration
    3. Write your prompt — raw data is included as context
    4. Good for custom analysis that requires the AI to read actual values

    **Supported platforms:**
    - OpenAI (GPT-4o, GPT-4, etc.)
    - DeepSeek
    - Qwen (通义千问)
    - Zhipu (智谱 GLM)
    - Any other OpenAI-compatible API
    """)
