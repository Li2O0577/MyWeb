import streamlit as st
import pandas as pd
import numpy as np

# current_path: 当前页面的文件路径

# 隐藏原生侧边栏导航
def hide_native_sidebar():
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {display: none;}
        </style>
    """, unsafe_allow_html=True)


def detect_outliers(df):
    """
    使用 IQR 方法检测数值列的异常值。
    返回 dict: {col_name: {count, indices, values, lower_bound, upper_bound}}
    """
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


def data_uploader(warn_outliers=True, force_cached=False):
    """
    通用数据上传与预处理组件
    warn_outliers: 是否显示异常值紧凑警告（Page 1 应设为 False，自行处理详情）
    force_cached: 跳过文件读取，强制使用 session_state 缓存数据（Page 1 修复异常值后使用）
    返回: 处理后的 DataFrame 或 None
    """
    if "original_df" not in st.session_state:
        st.session_state.original_df = None

    st.subheader("📂 Data Input")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], key="global_uploader")

    if uploaded_file is not None:
        current_name = uploaded_file.name

        # 检测是否为新文件上传（文件名变了 → 清除修复标记）
        if current_name != st.session_state.get("_source_file", ""):
            st.session_state._source_file = current_name
            st.session_state._data_cleaned = False

        if not force_cached and not st.session_state.get("_data_cleaned", False):
            try:
                # 1. 读取数据
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                # 2. 基础预处理工具栏
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

                # 3. 异常值检测
                outlier_info = detect_outliers(df)
                st.session_state.outliers = outlier_info

                # 4. 存入 Session State
                st.session_state['main_df'] = df
                st.success(f"成功加载: {uploaded_file.name}")

                # 5. 紧凑警告（非 Page 1 的页面）
                if warn_outliers and outlier_info:
                    total = sum(v["count"] for v in outlier_info.values())
                    cols = len(outlier_info)
                    st.warning(
                        f"⚠️ 检测到 **{total}** 个异常值，分布在 **{cols}** 个列中。"
                        f"建议前往 **📊 Data Load** 页面查看详情并进行处理。"
                    )

                return df

            except Exception as e:
                st.error(f"Error: {e}")
                return None

        # force_cached 或 _data_cleaned：使用缓存数据
        if 'main_df' in st.session_state:
            df = st.session_state['main_df']
            outlier_info = detect_outliers(df)
            st.session_state.outliers = outlier_info
            if warn_outliers and outlier_info:
                total = sum(v["count"] for v in outlier_info.values())
                cols = len(outlier_info)
                st.warning(
                    f"⚠️ 检测到 **{total}** 个异常值，分布在 **{cols}** 个列中。"
                    f"建议前往 **📊 Data Load** 页面查看详情并进行处理。"
                )
            return df

    # 如果之前已经上传过数据（且 file_uploader 已被清除），返回缓存
    if 'main_df' in st.session_state:
        df = st.session_state['main_df']
        outlier_info = detect_outliers(df)
        st.session_state.outliers = outlier_info
        if warn_outliers and outlier_info:
            total = sum(v["count"] for v in outlier_info.values())
            cols = len(outlier_info)
            st.warning(
                f"⚠️ 检测到 **{total}** 个异常值，分布在 **{cols}** 个列中。"
                f"建议前往 **📊 Data Load** 页面查看详情并进行处理。"
            )
        return df

    return None

# 通用侧边栏函数（全页面复用）
def render_sidebar(current_path):
    hide_native_sidebar()
    with st.sidebar:
        st.title("🧭 Navigation")
        st.divider()

        pages = [
            ("📊 Data Load", "pages/1_data_load.py"),
            ("📈 Data Visualization", "pages/2_data_visualization.py"),
            ("🧹 Data Processing", "pages/3_data_processing.py"),
            ("🧠 Regression", "pages/4_regression.py"),
            ("🔮 Classification", "pages/5_classification.py"),
            ("🛠️ DIY MLP", "pages/6_diy_mlp.py"),
            ("🌳 Decision Tree", "pages/7_decision_tree.py"),
            ("🧪 K-means", "pages/8_k_means.py"),
            ("🤖 LLM Analysis", "pages/9_llm_analysis.py"),
        ]

        # 循环渲染按钮（自动判断高亮+禁用点击）
        for label, path in pages:
            # 如果是当前页面 → 蓝色按钮 + 点击无反应
            if path == current_path:
                st.button(
                    label,
                    use_container_width=True,
                    type="primary"  # 蓝色高亮
                )

            else:
                if st.button(label, use_container_width=True):
                    st.switch_page(path)

        st.divider()

        # 返回主页按钮
        if st.button("🏠 Back to Home", use_container_width=True):
            st.switch_page("main.py")
        st.divider()
        st.caption("Indeterminate | Data Analysis Platform")
