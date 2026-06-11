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


def _outlier_warning(outlier_info):
    """Build a unified warning message from outlier detection results."""
    total_outliers = sum(v.get("count", 0) for v in outlier_info.values())
    total_nans = sum(v.get("nan_count", 0) for v in outlier_info.values())
    n_cols_outlier = sum(1 for v in outlier_info.values() if v.get("count", 0) > 0)
    n_cols_nan = sum(1 for v in outlier_info.values() if v.get("nan_count", 0) > 0)

    parts = []
    if total_outliers > 0:
        parts.append(f"**{total_outliers}** 个异常值（{n_cols_outlier} 列）")
    if total_nans > 0:
        parts.append(f"**{total_nans}** 个缺失值（{n_cols_nan} 列）")
    if not parts:
        return

    msg = "⚠️ 检测到 " + "，".join(parts)
    msg += "。建议前往 **📊 数据加载** 页面查看详情并进行处理。"
    st.warning(msg)


def _quick_ops_key(name):
    source_hash = st.session_state.get("_source_file_hash", "no_file")[:12]
    return f"quick_ops_{name}_{source_hash}"


def _render_quick_operations(df):
    """Render persistent quick cleanup controls for the current uploaded dataset."""
    if df is None:
        return df

    with st.expander("🛠️ 快速操作", expanded=False):
        st.caption("用于导入后做轻量清理。选择操作后点击“应用快速处理”，结果会保存为当前数据集。")
        col1, col2 = st.columns(2)
        with col1:
            drop_na = st.checkbox("删除包含空值的行", key=_quick_ops_key("drop_na"))
        with col2:
            drop_dup = st.checkbox("删除重复行", key=_quick_ops_key("drop_dup"))

        to_drop = st.multiselect("删除列", df.columns.tolist(), key=_quick_ops_key("drop_cols"))
        apply_ops = st.button("应用快速处理", key=_quick_ops_key("apply"), use_container_width=True)

        if apply_ops:
            new_df = df.copy()
            if drop_na:
                new_df = new_df.dropna()
            if drop_dup:
                new_df = new_df.drop_duplicates()
            if to_drop:
                new_df = new_df.drop(columns=to_drop, errors="ignore")
            new_df = new_df.reset_index(drop=True)

            st.session_state["main_df"] = new_df
            st.session_state._data_loaded = True
            st.session_state._data_cleaned = True
            st.session_state.outliers = detect_outliers(
                new_df,
                coefficient=st.session_state.get("outlier_coefficient", 3.0),
            )
            st.toast("快速处理已应用。", icon="✅")
            st.rerun()

    return df


def data_uploader(warn_outliers=True, force_cached=False, upload_to_backend=True):
    """Data upload component. If upload_to_backend=True, also sends file to Flask."""
    if "original_df" not in st.session_state:
        st.session_state.original_df = None
    if "outlier_coefficient" not in st.session_state:
        st.session_state.outlier_coefficient = 3.0

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

                # 2. Store df locally FIRST — data is immediately available to all pages
                outlier_info = detect_outliers(df, coefficient=st.session_state.get("outlier_coefficient", 3.0))
                st.session_state.outliers = outlier_info
                st.session_state['main_df'] = df
                st.session_state._data_loaded = True
                st.toast(f"成功加载: {uploaded_file.name}", icon="✅")

                # 3. Try Flask backend upload — non-blocking, fails fast (2s connect timeout)
                if upload_to_backend and "session_id" not in st.session_state:
                    try_upload_backend(raw_bytes, uploaded_file.name)

                if warn_outliers and outlier_info:
                    _outlier_warning(outlier_info)

                _render_quick_operations(df)
                return df

            except Exception as e:
                st.toast(f"错误：{e}", icon="❌")
                return None

        if 'main_df' in st.session_state:
            df = st.session_state['main_df']
            if force_cached or 'outliers' not in st.session_state:
                st.session_state.outliers = detect_outliers(df, coefficient=st.session_state.get("outlier_coefficient", 3.0))
            if warn_outliers:
                _outlier_warning(st.session_state.outliers)
            if uploaded_file is not None:
                _render_quick_operations(df)
            return df

    if 'main_df' in st.session_state:
        df = st.session_state['main_df']
        if 'outliers' not in st.session_state:
            st.session_state.outliers = detect_outliers(df, coefficient=st.session_state.get("outlier_coefficient", 3.0))
        if warn_outliers:
            _outlier_warning(st.session_state.outliers)
        if uploaded_file is not None:
            _render_quick_operations(df)
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
