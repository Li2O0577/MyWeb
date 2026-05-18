import streamlit as st

st.set_page_config(page_title="首页", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

st.title("Indeterminate")
st.write("数据分析和处理的 Web 平台 — Flask + Streamlit 前后端分离版")
st.divider()

st.subheader("基础功能")
st.write("")

left_space, col1, col2, col3, right_space = st.columns([1, 2, 2, 2, 1], gap="large")
with col1:
    if st.button("📊 数据加载", use_container_width=True, type="secondary"):
        st.switch_page("pages/1_data_load.py")
with col2:
    if st.button("📈 数据可视化", use_container_width=True, type="secondary"):
        st.switch_page("pages/2_data_visualization.py")
with col3:
    if st.button("🧹 数据处理", use_container_width=True, type="secondary"):
        st.switch_page("pages/3_data_processing.py")

st.divider()

st.subheader("高级功能")
st.caption("可能需要较好的电脑性能！")
st.write("")

left_space, col4, col5, col6, right_space = st.columns([1, 2, 2, 2, 1], gap="large")
with col4:
    if st.button("🧠 回归预测", use_container_width=True, type="secondary"):
        st.switch_page("pages/4_regression.py")
with col5:
    if st.button("🔮 分类决策", use_container_width=True, type="secondary"):
        st.switch_page("pages/5_classification.py")
with col6:
    if st.button("🛠️ 自定义 MLP", use_container_width=True, type="secondary"):
        st.switch_page("pages/6_diy_mlp.py")

left_space, col7, col8, col9, right_space = st.columns([1, 2, 2, 2, 1], gap="large")
with col7:
    if st.button("🌳 决策树", use_container_width=True, type="secondary"):
        st.switch_page("pages/7_decision_tree.py")
with col8:
    if st.button("🧪 聚类分析", use_container_width=True, type="secondary"):
        st.switch_page("pages/8_clustering.py")
with col9:
    if st.button("🤖 大模型分析", use_container_width=True, type="secondary"):
        st.switch_page("pages/9_llm_analysis.py")

st.divider()
st.write("开发者：Jiayang Li")

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
        if st.button(label, use_container_width=True):
            st.switch_page(path)
    st.divider()
    if st.button("🏠 返回首页", use_container_width=True, type="primary"):
        st.switch_page("main.py")
    st.divider()
    st.caption("Indeterminate | Flask + Streamlit")
