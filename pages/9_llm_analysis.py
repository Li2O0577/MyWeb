import streamlit as st
import pandas as pd
import json
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

# 3. API 配置
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

# 4. Prompt 输入
st.divider()
st.subheader("💬 Prompt")

col1, col2 = st.columns(2)
with col1:
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a professional data analyst. Please analyze the given data and provide insightful findings.",
        height=120
    )
with col2:
    user_prompt = st.text_area(
        "User Prompt",
        value="Please analyze the following data and give me a summary:",
        height=120,
        help="如果已上传数据，数据会自动追加到 prompt 后面"
    )

# 数据包含选项
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

# 5. 发送请求
st.divider()
col_btn, _ = st.columns([1, 4])
with col_btn:
    send_btn = st.button("🚀 Send", type="primary", use_container_width=True)

if send_btn:
    if not api_key:
        st.error("Please enter your API Key.")
    elif not model:
        st.error("Please enter a model name.")
    else:
        # 构建消息
        messages = [{"role": "system", "content": system_prompt}]

        content_parts = [user_prompt]

        if max_rows > 0 and df is not None:
            subset = df[col_subset] if col_subset else df
            subset = subset.head(max_rows)
            data_text = subset.to_csv(index=False)
            content_parts.append(f"\n\n--- Data (CSV) ---\n{data_text}")

        full_content = "\n".join(content_parts)
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

                    # 使用信息
                    usage = data.get("usage", {})
                    if usage:
                        with st.expander("📊 Token Usage", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Prompt Tokens", usage.get("prompt_tokens", "N/A"))
                            col2.metric("Completion Tokens", usage.get("completion_tokens", "N/A"))
                            col3.metric("Total Tokens", usage.get("total_tokens", "N/A"))
                else:
                    st.error(f"API Error [{resp.status_code}]: {resp.text}")

            except requests.exceptions.Timeout:
                st.error("Request timeout. Please try with fewer data rows or a shorter prompt.")
            except requests.exceptions.ConnectionError:
                st.error(f"Connection failed. Please check your API Base URL: {api_base}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# 6. 使用说明
st.divider()
with st.expander("📖 Instructions", expanded=False):
    st.markdown("""
    **How to use:**
    1. Upload your data in the Data Input section (or use data uploaded on other pages)
    2. Fill in your API configuration (Base URL, API Key, Model)
    3. Write your prompt - data will be automatically included as context
    4. Click Send and wait for the LLM's analysis

    **Supported platforms:**
    - OpenAI (GPT-4o, GPT-4, etc.)
    - DeepSeek
    - Qwen (通义千问)
    - Zhipu (智谱 GLM)
    - Any other OpenAI-compatible API

    **Tips:**
    - For large datasets, limit the rows included to avoid token limits
    - You can select specific columns to focus the analysis
    - The System Prompt helps set the role and behavior of the LLM
    """)
