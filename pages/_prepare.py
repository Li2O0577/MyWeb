"""Shared Streamlit utilities — adapted for Flask backend."""
import streamlit as st
import pandas as pd
import numpy as np
from pages._api import upload_data, get_outliers


def hide_native_sidebar():
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {display: none;}
        </style>
    """, unsafe_allow_html=True)


def detect_outliers(df):
    """IQR-based outlier detection (local helper for UI display)."""
    outliers = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        if mask.any():
            outliers[col] = {
                "count": int(mask.sum()),
                "indices": df.index[mask].tolist(),
                "values": df.loc[mask, col].tolist(),
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
            }
    return outliers


def data_uploader(warn_outliers=True, force_cached=False, upload_to_backend=True):
    """Data upload component. If upload_to_backend=True, also sends file to Flask."""
    if "original_df" not in st.session_state:
        st.session_state.original_df = None

    st.subheader("📂 数据导入")
    uploaded_file = st.file_uploader("上传 CSV/Excel", type=["csv", "xlsx"], key="global_uploader")

    if uploaded_file is not None:
        current_name = uploaded_file.name

        if current_name != st.session_state.get("_source_file", ""):
            st.session_state._source_file = current_name
            st.session_state._data_cleaned = False
            st.session_state.pop("session_id", None)

        if not force_cached and not st.session_state.get("_data_cleaned", False):
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

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

                # Upload to Flask backend
                if upload_to_backend and "session_id" not in st.session_state:
                    uploaded_file.seek(0)
                    result = upload_data(uploaded_file.read(), uploaded_file.name)
                    if result:
                        st.session_state.session_id = result["session_id"]
                        st.session_state.backend_data = result

                outlier_info = detect_outliers(df)
                st.session_state.outliers = outlier_info
                st.session_state['main_df'] = df
                st.success(f"成功加载: {uploaded_file.name}")

                if warn_outliers and outlier_info:
                    total = sum(v["count"] for v in outlier_info.values())
                    cols = len(outlier_info)
                    st.warning(
                        f"⚠️ 检测到 **{total}** 个异常值，分布在 **{cols}** 个列中。"
                        f"建议前往 **📊 数据加载** 页面查看详情并进行处理。"
                    )

                return df

            except Exception as e:
                st.error(f"错误：{e}")
                return None

        if 'main_df' in st.session_state:
            df = st.session_state['main_df']
            if 'outliers' not in st.session_state:
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
        if 'outliers' not in st.session_state:
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
