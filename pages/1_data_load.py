import streamlit as st
from pages._prepare import data_uploader, render_sidebar

# 1. 配置页面，隐藏侧边栏导航 
st.set_page_config(page_title="Data Load", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
"""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)
render_sidebar("pages/1_data_load.py")
st.title("📊 Data Load")

# 2. 数据上传与预处理组件
df = data_uploader()
if df is not None:
    st.dataframe(df)

# 3. 解释以及后续步骤提示
st.divider()
if df is None:
    st.info("你应当先上传数据，才能进行后续的可视化、处理和建模等操作（支持 CSV 和 Excel）。")
else :
    st.success("数据加载成功！你现在可以导航到其他页面进行可视化、处理或建模。")