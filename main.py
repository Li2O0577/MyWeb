import streamlit as st


# 1. 配置页面，隐藏侧边栏导航 (initial_sidebar_state="collapsed")
st.set_page_config(page_title="Home Page", layout="wide", initial_sidebar_state="collapsed")
# 隐藏侧边栏默认导航栏的黑科技
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)

st.title("Indeterminate")
st.write("This is a web page for data processing and analysis")
st.divider()


# 2. 创建按钮布局
st.subheader("basic functions")
st.write("")

left_space, col1, col2, col3, right_space = st.columns([1, 2, 2, 2, 1], gap="large")
with col1:
    if st.button("📊 Data Load", use_container_width=True, type="secondary"):
        st.switch_page("pages/1_data_load.py")      #（数据查看）
with col2:
    if st.button("📈 Data Visualization", use_container_width=True, type="secondary"):
        st.switch_page("pages/2_data_visualization.py")        #（数据可视化）
with col3:
    if st.button("🧹 Data processing", use_container_width=True, type="secondary"):
        st.switch_page("pages/3_data_processing.py")       #（数据简单处理）

st.divider()

st.subheader("advanced functions")
st.caption("maybe need PC performance!")
st.write("")

left_space, col4, col5, col6, right_space = st.columns([1, 2, 2, 2, 1], gap="large")
with col4:
    if st.button("🧠 Regression", use_container_width=True,type="secondary"):
        st.switch_page("pages/4_regression.py")      #（回归）
with col5:
    if st.button("🔮 Classification", use_container_width=True,type="secondary"):
        st.switch_page("pages/5_classification.py")      #（分类）
with col6:
    if st.button("🛠️ DIY MLP", use_container_width=True,type="secondary"):
        st.switch_page("pages/6_diy_mlp.py")      #（自定义mlp）

left_space, col7, col8, col9, right_space = st.columns([1, 2, 2, 2, 1], gap="large")
with col7:
    if st.button("🌳 Decision Tree", use_container_width=True,type="secondary"):
        st.switch_page("pages/7_decision_tree.py")      #（决策树）
with col8:
    if st.button("🧪 K-means", use_container_width=True,type="secondary"):
        st.switch_page("pages/8_k_means.py")      #（K-means聚类）
with col9:
    if st.button("🤖 LLM Analysis", use_container_width=True, type="secondary"):
        st.switch_page("pages/9_llm_analysis.py")      #（大模型分析）

st.divider()

st.write("Develpoped by : Jiayang Li")


# 3. 创建侧边栏布局
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
    for label, path in pages:
        if st.button(label, use_container_width=True):
            st.switch_page(path)
    st.divider()

    st.button("🏠 Back to Home", use_container_width=True, type="primary")
    st.divider()
    st.caption("Indeterminate | Data Analysis Platform")
