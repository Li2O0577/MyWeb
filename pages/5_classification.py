"""Classification page — Flask backend for training, Streamlit for UI."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import classification_report
from pages._prepare import render_sidebar, data_uploader
from pages._mlp_common import render_device_selector, check_constant_features, plot_loss_curve, validate_input_array
from pages._api import train_classification, predict_classification, batch_predict_classification, clear_classification

st.set_page_config(page_title="分类决策", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>[data-testid="stSidebarNav"] {display: none;}</style>""", unsafe_allow_html=True)
render_sidebar("pages/5_classification.py")
st.title("🔮 分类决策")

device = render_device_selector()

with st.expander("📢 功能介绍", expanded=True):
    st.markdown("""
    ### 智能神经网络分类决策模型
    1. **任务说明**：分类预测（二分类/多分类，如标签判断、类别决策、结果分类）
    2. **自动适配**：自动识别二分类(Sigmoid) / 多分类(Softmax)
    3. **防过拟合**：自适应网络结构 + Dropout + 早停 + L2正则
    4. **前后端分离**：Flask 后端训练，Streamlit 前端展示
    """)

df = data_uploader()
if df is None:
    st.warning("⚠️ 请先上传数据！")
    st.stop()

numeric_df = df.select_dtypes(include=[np.number]).dropna()
if len(numeric_df) < 10 or len(numeric_df.columns) < 2:
    st.error("❌ 数据无效！需要至少2列数值数据 + 10行数据")
    st.stop()

col_options = list(numeric_df.columns)

st.subheader("📊 数据自动分析与模型配置")
col1, col2 = st.columns(2)

with col1:
    target_col = st.selectbox("选择分类目标列（y）", col_options, index=len(col_options)-1)
    feature_cols = [col for col in col_options if col != target_col]
    n_features = len(feature_cols)
    n_samples = len(numeric_df)
    n_classes = len(numeric_df[target_col].unique())

    if n_features == 0:
        st.warning("⚠️ 警告：选择的目标列覆盖了所有列，无可用特征列！请重新选择目标列")
        st.stop()
    if n_classes < 2:
        st.warning(f"⚠️ 警告：选择的目标列「{target_col}」仅包含 {n_classes} 个类别，无法进行分类任务！")
        st.stop()
    if n_classes > n_samples * 0.5:
        st.warning(f"⚠️ 警告：类别数接近样本数，可能过拟合！建议更换目标列")
    st.info(f"✅ 数据：{n_samples} 行 | {n_features} 个特征 | {n_classes} 个类别")
    check_constant_features(numeric_df, feature_cols)

st.subheader("⚡ 训练参数")
hp_col1, hp_col2, hp_col3 = st.columns(3)
with hp_col1:
    learning_rate = st.selectbox("学习率 (LR)", [0.01, 0.005, 0.001, 0.0005, 0.0001], index=2)
with hp_col2:
    epochs = st.slider("最大训练轮数", 20, 500, 100, 20)
with hp_col3:
    batch_size = st.selectbox("批次大小", [4, 8, 16, 32, 64, 128], index=1)

st.subheader("🚀 模型训练")
train_col, clear_col = st.columns(2)

with train_col:
    if st.button("开始训练 / 重新训练模型", type="primary", use_container_width=True):
        if "session_id" not in st.session_state:
            st.error("⚠️ 请先在「数据加载」页面重新上传数据以初始化后端会话。")
        else:
            class_counts = numeric_df[target_col].value_counts()
            if class_counts.min() < 2:
                st.error("❌ 训练失败：每个类别至少需要2个样本！")
                st.stop()

            with st.spinner("分类模型训练中（后端 Flask 计算）..."):
                device_str = "cuda" if "CUDA" in str(device) else "cpu"
                result = train_classification(
                    st.session_state.session_id,
                    str(target_col),
                    [str(c) for c in feature_cols],
                    learning_rate, epochs, batch_size, device_str
                )
                if result and "acc" in result:
                    st.session_state.cls_result = result
                    st.session_state.cls_features = feature_cols
                    st.session_state.cls_target = target_col
                    st.session_state.cls_n_classes = result["n_classes"]
                    st.session_state.cls_reverse_label_map = result["reverse_label_map"]

                    st.success(f"训练完成！准确率 = {result['acc']:.4f}")

                    # Confusion matrix
                    cm = result["cm"]
                    label_names = result["label_names"]
                    col_cm, col_report = st.columns([1, 1])
                    with col_cm:
                        st.caption("混淆矩阵")
                        fig_cm = go.Figure(data=go.Heatmap(
                            z=cm, x=label_names, y=label_names,
                            text=cm, texttemplate="%{text}", textfont=dict(size=14),
                            colorscale="Blues", showscale=False
                        ))
                        fig_cm.update_layout(xaxis_title="预测值", yaxis_title="实际值",
                                             height=300, margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(fig_cm, use_container_width=True)

                    plot_loss_curve(result["train_losses"], result["val_losses"])
                else:
                    st.error(f"训练失败：{result}")

with clear_col:
    if st.button("清除已保存分类模型", use_container_width=True):
        clear_classification()
        for k in ["cls_result", "cls_features", "cls_target", "cls_n_classes", "cls_reverse_label_map"]:
            if k in st.session_state:
                del st.session_state[k]
        st.warning("已清除分类模型！")

st.subheader("🎯 决策预测")
if "cls_result" not in st.session_state:
    st.warning("请先训练模型！")
else:
    features = st.session_state.cls_features
    target = st.session_state.cls_target
    n_classes = st.session_state.cls_n_classes
    reverse_label_map = st.session_state.cls_reverse_label_map

    st.info(f"✅ 模型已训练 | 分类目标：{target} | 类别数：{n_classes}")
    st.write("请输入特征值进行决策预测：")
    input_data = []
    cols = st.columns(len(features))
    for i, col in enumerate(cols):
        val = col.number_input(f"特征 {features[i]}", value=0.0, step=0.1)
        input_data.append(val)

    if st.button("执行决策预测", use_container_width=True):
        device_str = "cuda" if "CUDA" in str(device) else "cpu"
        err = validate_input_array(np.array([input_data]), "单条预测")
        if err:
            st.error(err)
        else:
            result = predict_classification(input_data, device_str)
            if result and "pred_idx" in result:
                pred_idx = result["pred_idx"]
                pred_class = reverse_label_map.get(str(pred_idx), pred_idx)
                st.success(f"🎯 预测类别：{pred_class} | 置信度：{result['prob']:.4f}")

    st.divider()
    st.subheader("📦 批量预测")
    batch_file = st.file_uploader("上传包含特征列的 CSV 文件", type=["csv"], key="cls_batch")
    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        missing_cols = set(features) - set(batch_df.columns)
        if missing_cols:
            st.error(f"缺少特征列：{missing_cols}")
        else:
            batch_X = batch_df[features].values
            err = validate_input_array(batch_X, "批量预测")
            if err:
                st.error(err)
            else:
                result = batch_predict_classification(batch_X.tolist(), device_str)
                if result and "pred_indices" in result:
                    result_df = batch_df.copy()
                    result_df["预测类别"] = [reverse_label_map.get(str(i), i) for i in result["pred_indices"]]
                    result_df["置信度"] = result["confidences"]
                    st.dataframe(result_df, use_container_width=True)
                    csv = result_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 下载预测结果", csv, "predictions.csv", "text/csv", use_container_width=True)
