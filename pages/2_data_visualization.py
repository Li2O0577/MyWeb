import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pages._prepare import render_sidebar, data_uploader

# 1. 配置页面
st.set_page_config(page_title="Data Load", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
"""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)
render_sidebar("pages/2_data_visualization.py")
st.title("📈 Data Visualization")

# 2. 数据上传与画图
df = data_uploader()

if df is not None:
    st.success("Data loaded successfully! You can now create visualizations.")
    st.dataframe(df.head(5), use_container_width=True)

    # 强制所有列名转为字符串（避免数字列名导致的画图问题）
    df.columns = df.columns.astype(str)
    
    # 1. 选择图表类型
    chart_type = st.selectbox("choose chart type", [
        "Line Chart (折线图) ",
        "Bar Chart (柱状图)",
        "Scatter Plot (散点图)",
        "Histogram (直方图)",
        "Box Plot (箱线图)",
        "Density Plot (密度图)",
        "Correlation Heatmap (热力图)",
        "Pie Chart (饼图)",
        "Pair Plot (成对关系图)"
    ])

    # 获取数值列（只允许数字列画图）
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    all_cols = df.columns.tolist()

    # 2. 根据类型画图
    if chart_type == "Line Chart (折线图) ":

        x_options = ["默认索引（默认）"] + numeric_cols
        x = st.selectbox("X轴", x_options)
        x = None if x == "默认索引（默认）" else x
        
        y = st.multiselect("Y轴（可多选）", numeric_cols, default=numeric_cols[:1])
        if y:
            st.line_chart(df, x=x, y=y, use_container_width=True)

    elif chart_type == "Bar Chart (柱状图)":
        x = st.selectbox("X轴", all_cols)
        y = st.selectbox("Y轴", numeric_cols)
        fig = px.bar(df, x=x, y=y, title=f"{y} 按 {x} 分布")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Plot (散点图)":
        x = st.selectbox("X轴", numeric_cols)
        y = st.selectbox("Y轴", numeric_cols)
        color_col = st.selectbox("颜色分类（可选）", [None] + all_cols)
        fig = px.scatter(df, x=x, y=y, color=color_col, title=f"{x} vs {y}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Histogram (直方图)":
        col = st.selectbox("选择列", numeric_cols)
        bins = st.slider("柱子数量", 10, 100, 30)
        fig, ax = plt.subplots()
        sns.histplot(df[col], bins=bins, kde=True, ax=ax)
        ax.set_title(f"Histogram: {col}")
        st.pyplot(fig, use_container_width=True)

    elif chart_type == "Box Plot (箱线图)":
        col = st.selectbox("选择列", numeric_cols)
        fig, ax = plt.subplots()
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(f"Box Plot: {col}")
        st.pyplot(fig, use_container_width=True)

    elif chart_type == "Density Plot (密度图)":
        col = st.selectbox("选择列", numeric_cols)
        fig, ax = plt.subplots()
        sns.kdeplot(df[col], fill=True, ax=ax)
        ax.set_title(f"Density Plot: {col}")
        st.pyplot(fig, use_container_width=True)

    elif chart_type == "Correlation Heatmap (热力图)":
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig, use_container_width=True)

    elif chart_type == "Pie Chart (饼图)":
        col = st.selectbox("分类列", all_cols)
        fig = px.pie(df, names=col, title=f"Pie Chart: {col}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pair Plot (成对关系图)":
        cols = st.multiselect("选择列（最多4个）", numeric_cols, default=numeric_cols[:2])
        if len(cols) > 1:
            pair_fig = sns.pairplot(df[cols])
            st.pyplot(pair_fig, use_container_width=True)

else:
    st.warning("Please upload a dataset to visualize. Supported formats: CSV, Excel.")