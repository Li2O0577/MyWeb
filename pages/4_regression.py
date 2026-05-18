"""Regression page — Flask backend for training, Streamlit for UI."""
import streamlit as st
import pandas as pd
import numpy as np
from pages._prepare import render_sidebar, data_uploader
from pages._mlp_common import render_device_selector, check_constant_features, plot_loss_curve, validate_input_array
from pages._api import train_regression, predict_regression, batch_predict_regression, clear_regression

st.set_page_config(page_title="回归预测", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>[data-testid="stSidebarNav"] {display: none;}</style>""", unsafe_allow_html=True)
render_sidebar("pages/4_regression.py")
st.title("🧠 回归预测")

device = render_device_selector()

with st.expander("📢 功能介绍", expanded=True):
    st.markdown("""
    ### 智能神经网络回归预测
    1. **任务说明**：预测**连续数值**（价格、销量、温度、浓度等）
    2. **智能建模**：自动分析数据规模，生成**防过拟合/欠拟合**的最优网络结构
    3. **硬件自适应**：自动检测GPU(CUDA)/CPU，自动选择最优训练设备
    4. **模型持久化**：训练后自动保存模型，重启页面**自动加载**，无需重复训练
    5. **前后端分离**：Flask 后端训练，Streamlit 前端展示
    """)

df = data_uploader()
if df is None:
    st.warning("⚠️ 请先上传数据！")
    st.stop()

numeric_df = df.select_dtypes(include=[np.number]).dropna()
if len(numeric_df) < 10 or len(numeric_df.columns) < 2:
    st.error("❌ 数据无效！需要至少2列数值数据 + 10行数据")
    st.stop()

st.subheader("📊 数据自动分析与模型配置")
col1, col2 = st.columns(2)

with col1:
    target_col = st.selectbox("选择预测目标列（y）", numeric_df.columns, index=len(numeric_df.columns)-1)
    feature_cols = [col for col in numeric_df.columns if col != target_col]
    n_features = len(feature_cols)
    n_samples = len(numeric_df)
    if n_features == 0:
        st.warning("⚠️ 警告：选择的目标列覆盖了所有列，无可用特征列！请重新选择目标列")
        st.stop()
    st.info(f"✅ 数据：{n_samples} 行 | {n_features} 个特征")
    check_constant_features(numeric_df, feature_cols)

with col2:
    if n_samples < 500:
        st.caption(f"小数据集 → 轻量网络 (h1={max(8, n_features)}, h2={max(4, n_features//2)}, dropout=0.1)")
    elif n_samples < 5000:
        st.caption(f"中数据集 → 标准网络 (h1={n_features*2}, h2={n_features}, dropout=0.2)")
    else:
        st.caption(f"大数据集 → 深层网络 (h1={n_features*3}, h2={n_features*2}, dropout=0.3)")

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
            with st.spinner("训练中（后端 Flask 计算中）..."):
                device_str = "cuda" if "CUDA" in str(device) else "cpu"
                result = train_regression(
                    st.session_state.session_id,
                    str(target_col),
                    [str(c) for c in feature_cols],
                    learning_rate, epochs, batch_size, device_str
                )
                if result and "r2" in result:
                    st.session_state.reg_result = result
                    st.session_state.reg_features = feature_cols
                    st.session_state.reg_target = target_col

                    st.success(f"训练完成！R² = {result['r2']:.4f} | MAE = {result['mae']:.4f} | RMSE = {result['rmse']:.4f}")
                    plot_loss_curve(result["train_losses"], result["val_losses"], y_label="损失值 (MSE)")
                else:
                    st.error(f"训练失败：{result}")

with clear_col:
    if st.button("清除已保存模型", use_container_width=True):
        clear_regression()
        for k in ["reg_result", "reg_features", "reg_target"]:
            if k in st.session_state:
                del st.session_state[k]
        st.warning("已清除所有保存的模型！")

st.subheader("🎯 数据预测")
if "reg_result" not in st.session_state:
    st.warning("请先训练模型！")
else:
    features = st.session_state.reg_features
    target = st.session_state.reg_target
    st.info(f"✅ 模型已训练 | 预测目标：{target}")
    st.write("请输入特征值进行预测：")
    input_data = []
    cols = st.columns(len(features))
    for i, col in enumerate(cols):
        val = col.number_input(f"{features[i]}", value=0.0, step=0.1)
        input_data.append(val)

    if st.button("执行预测", use_container_width=True):
        device_str = "cuda" if "CUDA" in str(device) else "cpu"
        err = validate_input_array(np.array([input_data]), "单条预测")
        if err:
            st.error(err)
        else:
            result = predict_regression(input_data, device_str)
            if result and "result" in result:
                st.success(f"预测结果：{result['result']:.4f}")

    st.divider()
    st.subheader("📦 批量预测")
    batch_file = st.file_uploader("上传包含特征列的 CSV 文件", type=["csv"], key="reg_batch")
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
                result = batch_predict_regression(batch_X.tolist(), device_str)
                if result and "predictions" in result:
                    result_df = batch_df.copy()
                    result_df[f"预测_{target}"] = result["predictions"]
                    st.dataframe(result_df, use_container_width=True)
                    csv = result_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 下载预测结果", csv, "predictions.csv", "text/csv", use_container_width=True)
