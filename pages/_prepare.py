import streamlit as st
import pandas as pd

# current_path: 当前页面的文件路径

# 隐藏原生侧边栏导航
def hide_native_sidebar():
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {display: none;}
        </style>
    """, unsafe_allow_html=True)

def data_uploader():
    """
    通用数据上传与预处理组件
    返回: 处理后的 DataFrame 或 None
    """
    if "original_df" not in st.session_state:
        st.session_state.original_df = None

    st.subheader("📂 Data Input")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], key="global_uploader")

    if uploaded_file is not None:
        try:
            # 1. 读取数据
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # 2. 基础预处理工具栏（放在收纳盒里，不占空间）
            with st.expander("🛠️ 快速操作"):
                col1, col2 = st.columns(2)
                with col1:
                    if st.checkbox("删除空格所在行"):
                        df = df.dropna()
                with col2:
                    if st.checkbox("删除重复行"):
                        df = df.drop_duplicates()

                # 手动删列
                to_drop = st.multiselect("删除列", df.columns)
                if to_drop:
                    df = df.drop(columns=to_drop)

            # 3. 存入 Session State 确保跨页面可用
            st.session_state['main_df'] = df
            st.success(f"成功加载: {uploaded_file.name}")
            return df

        except Exception as e:
            st.error(f"Error: {e}")
            return None

    # 如果之前已经上传过数据，直接返回缓存的数据
    if 'main_df' in st.session_state:
        return st.session_state['main_df']

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
