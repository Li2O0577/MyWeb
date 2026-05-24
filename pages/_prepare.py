"""Shared Streamlit utilities — adapted for Flask backend."""
import hashlib
import io
import streamlit as st
import pandas as pd
import numpy as np
from pages._api import upload_data, try_upload_backend
from backend.services.data_service import detect_outliers


def hide_native_sidebar():
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {display: none;}
        </style>
    """, unsafe_allow_html=True)


def data_uploader(warn_outliers=True, force_cached=False, upload_to_backend=True):
    """Data upload component. If upload_to_backend=True, also sends file to Flask."""
    if "original_df" not in st.session_state:
        st.session_state.original_df = None

    st.subheader("📂 数据导入")
    uploaded_file = st.file_uploader("上传 CSV/Excel", type=["csv", "xlsx"], key="global_uploader")

    if uploaded_file is not None:
        current_name = uploaded_file.name
        raw_bytes = uploaded_file.getvalue()
        current_hash = hashlib.sha256(raw_bytes).hexdigest()

        if current_hash != st.session_state.get("_source_file_hash", ""):
            st.session_state._source_file = current_name
            st.session_state._source_file_hash = current_hash
            st.session_state._data_cleaned = False
            st.session_state._data_loaded = False
            st.session_state.pop("session_id", None)
            st.session_state.pop("session_meta", None)
            # Save raw bytes for efficient re-upload
            st.session_state._raw_file_bytes = raw_bytes
            st.session_state._raw_file_name = uploaded_file.name

        if not force_cached and not st.session_state.get("_data_loaded", False):
            try:
                # 1. Read file locally — always fast, never blocks
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(io.BytesIO(raw_bytes))
                else:
                    df = pd.read_excel(io.BytesIO(raw_bytes))

                with st.expander("🛠️ 快速操作"):
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.checkbox("删除空格所在行"):
                            df = df.dropna()
                    with col2:
                        if st.checkbox("删除重复行"):
                            df = df.drop_duplicates()

                    to_drop = st.multiselect("删除列", df.columns)
                    if to_drop:
                        df = df.drop(columns=to_drop)

                # 2. Store df locally FIRST — data is immediately available to all pages
                outlier_info = detect_outliers(df)
                st.session_state.outliers = outlier_info
                st.session_state['main_df'] = df
                st.session_state._data_loaded = True
                st.toast(f"成功加载: {uploaded_file.name}", icon="✅")

                # 3. Try Flask backend upload — non-blocking, fails fast (2s connect timeout)
                if upload_to_backend and "session_id" not in st.session_state:
                    try_upload_backend(raw_bytes, uploaded_file.name)

                if warn_outliers and outlier_info:
                    total = sum(v["count"] for v in outlier_info.values())
                    cols = len(outlier_info)
                    st.warning(
                        f"⚠️ 检测到 **{total}** 个异常值，分布在 **{cols}** 个列中。"
                        f"建议前往 **📊 数据加载** 页面查看详情并进行处理。"
                    )

                return df

            except Exception as e:
                st.toast(f"错误：{e}", icon="❌")
                return None

        if 'main_df' in st.session_state:
            df = st.session_state['main_df']
            # Re-detect outliers after data was cleaned (indices/bounds may have changed)
            if force_cached or 'outliers' not in st.session_state:
                st.session_state.outliers = detect_outliers(df)
            outlier_info = st.session_state.outliers
            if warn_outliers and outlier_info:
                total = sum(v["count"] for v in outlier_info.values())
                cols = len(outlier_info)
                st.warning(
                    f"⚠️ 检测到 **{total}** 个异常值，分布在 **{cols}** 个列中。"
                    f"建议前往 **📊 数据加载** 页面查看详情并进行处理。"
                )
            return df

    if 'main_df' in st.session_state:
        df = st.session_state['main_df']
        # Always re-detect — main_df may have changed columns (e.g. PCA in processing page)
        st.session_state.outliers = detect_outliers(df)
        outlier_info = st.session_state.outliers
        if warn_outliers and outlier_info:
            total = sum(v["count"] for v in outlier_info.values())
            cols = len(outlier_info)
            st.warning(
                f"检测到 **{total}** 个异常值，分布在 **{cols}** 个列中。"
                f"建议前往 **数据加载** 页面查看详情并进行处理。"
            )
        return df

    return None


def render_sidebar(current_path):
    """Sidebar navigation — identical to original."""
    hide_native_sidebar()
    with st.sidebar:
        st.title("🧭 导航")
        st.divider()

        pages = [
            ("📊 数据加载", "pages/1_data_load.py"),
            ("📈 数据可视化", "pages/2_data_visualization.py"),
            ("🧹 数据处理", "pages/3_data_processing.py"),
            ("🧠 回归预测", "pages/4_regression.py"),
            ("🔮 分类决策", "pages/5_classification.py"),
            ("🛠️ 自定义 MLP", "pages/6_diy_mlp.py"),
            ("🌳 决策树", "pages/7_decision_tree.py"),
            ("🧪 聚类分析", "pages/8_clustering.py"),
            ("🤖 大模型分析", "pages/9_llm_analysis.py"),
        ]

        for label, path in pages:
            if path == current_path:
                st.button(label, use_container_width=True, type="primary")
            else:
                if st.button(label, use_container_width=True):
                    st.switch_page(path)

        st.divider()
        if st.button("🏠 返回首页", use_container_width=True):
            st.switch_page("main.py")
        st.divider()
        st.caption("Indeterminate | 数据分析平台")
