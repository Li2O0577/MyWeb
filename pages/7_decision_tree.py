"""Decision tree page — Flask backend for training, Streamlit for UI."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import classification_report
from pages._prepare import render_sidebar, data_uploader
from pages._api import train_decision_tree, predict_decision_tree, clear_decision_tree, ensure_session, decision_tree_status, backend_status_badge, render_backend_sync_panel

st.set_page_config(page_title="决策树", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>[data-testid="stSidebarNav"] {display: none;}</style>""", unsafe_allow_html=True)
render_sidebar("pages/7_decision_tree.py")
st.title("🌳 决策树模型")

backend_status_badge()

with st.expander("📢 功能介绍", expanded=True):
    st.markdown("""
    ### 决策树模型（分类 & 回归 · 带可解释性）
    1. **双任务支持**：分类任务 + 回归任务，一键切换
    2. **智能预处理**：自动识别分类特征 + 独热编码，数值特征标准化
    3. **可解释性**：展示每一层决策规则、划分特征、阈值、不纯度
    4. **前后端分离**：Flask 后端训练，Streamlit 前端展示
    """)

df = data_uploader()
if df is None:
    st.warning("⚠️ 请先上传数据！")
    st.stop()

df_before = len(df)
df_clean = df.dropna()
dropped = df_before - len(df_clean)
if dropped > 0:
    st.warning(f"⚠️ 已自动丢弃 {dropped} 行含缺失值的数据（剩余 {len(df_clean)} 行 / {df_before} 行）。可在「数据处理」页面手动处理缺失值。")
if len(df_clean) < 10 or df_clean.shape[1] < 2:
    st.error("❌ 数据无效！需要至少10行有效数据")
    st.stop()

render_backend_sync_panel(df_clean)

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()

if "dt_task_type" not in st.session_state:
    st.session_state.dt_task_type = "classification"

st.subheader("⚙️ 模型参数配置")
target_col = st.selectbox("选择目标列（y）", df_clean.columns, index=len(df_clean.columns)-1)

task_col, depth_col, crit_col = st.columns(3)
with task_col:
    task_type = st.selectbox("任务类型", ["分类 (Classification)", "回归 (Regression)"],
                             index=0 if st.session_state.dt_task_type == "classification" else 1)
    st.session_state.dt_task_type = "classification" if "分类" in task_type else "regression"
with depth_col:
    max_depth = st.number_input("决策树最大深度", min_value=2, max_value=15, value=3, step=1)
with crit_col:
    if st.session_state.dt_task_type == "classification":
        criterion = st.selectbox("划分标准", ["entropy", "gini"], index=0)
    else:
        criterion = st.selectbox("划分标准", ["squared_error", "friedman_mse", "absolute_error", "poisson"], index=0)
    st.info(f"✅ 数值特征：{len(numeric_cols)} 个 | 分类特征：{len(categorical_cols)} 个")

feature_cols = [col for col in df_clean.columns if col != target_col]
n_features = len(feature_cols)
n_samples = len(df_clean)

if n_features == 0:
    st.warning("⚠️ 无可用特征列！")
    st.stop()

if st.session_state.dt_task_type == "classification":
    n_classes = len(df_clean[target_col].unique())
    if n_classes < 2:
        st.warning("⚠️ 目标列至少需要2个类别！")
        st.stop()
    if n_classes > n_samples * 0.5:
        st.warning(f"⚠️ 类别数 ({n_classes}) 接近样本数 ({n_samples})，可能过拟合")
else:
    if target_col not in numeric_cols:
        st.warning("⚠️ 回归任务请选择数值型目标列！")
        st.stop()

is_cls = (st.session_state.dt_task_type == "classification")

# Auto-detect saved model
if "dt_result" not in st.session_state:
    status = decision_tree_status()
    if status and status.get("has_model"):
        st.session_state.dt_result = {}
        st.session_state.dt_features = status.get("features", [])
        st.session_state.dt_target = status.get("target", "")
        st.session_state.dt_task_saved = status.get("task_type", "classification")
        st.success(f"✅ 已自动加载上次保存的决策树模型（目标列：{st.session_state.dt_target}）")

st.subheader("🚀 模型训练")
train_col, clear_col = st.columns(2)

with train_col:
    if st.button("开始训练" if "dt_result" not in st.session_state else "重新训练", type="primary", use_container_width=True):
        with st.spinner("训练中（后端 Flask 计算）..."):
            sid = ensure_session(df_clean)
            if not sid:
                st.error("无法连接到 Flask 后端 (http://localhost:5001)。请确保后端已启动。")
                st.stop()
            task_str = "classification" if is_cls else "regression"
            result = train_decision_tree(
                sid,
                str(target_col),
                [str(c) for c in feature_cols],
                task_str, criterion, max_depth
            )
            if result:
                st.session_state.dt_result = result
                st.session_state.dt_features = feature_cols
                st.session_state.dt_target = target_col
                st.session_state.dt_task_saved = task_str

                if is_cls:
                    st.success(f"✅ 训练完成！测试集准确率 = {result['acc']:.4f}")
                else:
                    st.success(f"✅ 训练完成！R² = {result['r2']:.4f} | MAE = {result['mae']:.4f} | RMSE = {result['rmse']:.4f}")

                if is_cls and "cm" in result:
                    cm = result["cm"]
                    label_names = result["label_names"]
                    col_cm, col_report = st.columns([1, 1])
                    with col_cm:
                        st.caption("混淆矩阵")
                        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=label_names, y=label_names, text=cm, texttemplate="%{text}", textfont=dict(size=14), colorscale="Blues", showscale=False))
                        fig_cm.update_layout(xaxis_title="预测值", yaxis_title="实际值", height=300, margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(fig_cm, use_container_width=True)
                if not is_cls:
                    st.metric("R² Score", f"{result['r2']:.4f}")
                    st.metric("MAE", f"{result['mae']:.4f}")
                    st.metric("RMSE", f"{result['rmse']:.4f}")

                if "tree_rules" in result:
                    st.markdown("### 🌿 决策树层级规则")
                    st.code(result["tree_rules"], language="text")
                if "tree_nodes" in result:
                    st.markdown("### 📊 每层节点详细信息")
                    st.dataframe(pd.DataFrame(result["tree_nodes"]), use_container_width=True)
            else:
                st.error(f"训练失败：{result}")

with clear_col:
    if st.button("清除已保存决策树模型", use_container_width=True):
        clear_decision_tree()
        for k in ["dt_result", "dt_features", "dt_target", "dt_task_saved"]:
            if k in st.session_state: del st.session_state[k]
        st.warning("已清除决策树模型！")

# Prediction
st.subheader("🎯 新数据预测")
if "dt_result" not in st.session_state:
    st.warning("请先训练模型！")
else:
    features = st.session_state.dt_features
    predict_is_cls = (st.session_state.get("dt_task_saved", "classification") == "classification")
    st.info(f"✅ 模型已训练 | 任务类型：{'分类' if predict_is_cls else '回归'}")

    st.write("请输入特征值进行预测：")
    input_data = {}
    cols = st.columns(min(len(features), 5))
    for i, col in enumerate(cols):
        feat = features[i]
        if feat in categorical_cols:
            options = df_clean[feat].unique().tolist()
            val = col.selectbox(f"{feat}", options, key=f"dt_{i}")
        else:
            val = col.number_input(f"{feat}", value=0.0, step=0.1, key=f"dt_{i}")
        input_data[feat] = [val]
    if len(features) > 5:
        for row_start in range(5, len(features), 5):
            cols = st.columns(5)
            for j, col in enumerate(cols):
                idx = row_start + j
                if idx < len(features):
                    feat = features[idx]
                    if feat in categorical_cols:
                        options = df_clean[feat].unique().tolist()
                        val = col.selectbox(f"{feat}", options, key=f"dt_{idx}")
                    else:
                        val = col.number_input(f"{feat}", value=0.0, step=0.1, key=f"dt_{idx}")
                    input_data[feat] = [val]

    btn_label = "执行分类预测" if predict_is_cls else "执行回归预测"
    if st.button(btn_label, use_container_width=True):
        task_str = "classification" if predict_is_cls else "regression"
        result = predict_decision_tree(input_data, task_str)
        if result:
            if predict_is_cls:
                st.success(f"🎯 预测类别：{result['pred_class']}")
            else:
                st.success(f"🎯 预测结果：{result['pred_value']:.4f}")
