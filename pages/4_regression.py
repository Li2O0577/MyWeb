"""Regression page — Flask backend for training, Streamlit for UI."""
import streamlit as st
import pandas as pd
import numpy as np
from pages._prepare import render_sidebar, data_uploader
from pages._mlp_common import render_device_selector, check_constant_features, plot_loss_curve, validate_input_array, render_version_selector, render_ml_status_bar, render_ml_workbench_overview, render_stable_prediction_panel, render_small_dataset_warning
from pages._api import train_regression, predict_regression, batch_predict_regression, clear_regression, ensure_session, regression_status, backend_status_badge, render_backend_sync_panel, list_regression_versions, activate_regression_version, delete_regression_version
from pages._ui_common import render_page_header, render_section_header

st.set_page_config(page_title="回归预测", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>[data-testid="stSidebarNav"] {display: none;}</style>""", unsafe_allow_html=True)
render_sidebar("pages/4_regression.py")
render_page_header("回归预测", "训练连续值预测模型，并通过当前激活版本完成单条或批量预测。")

backend_status_badge()
device = render_device_selector()

with st.expander("功能介绍", expanded=False):
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

backend_synced = render_backend_sync_panel(numeric_df, compact=True)

# Version selector
active_vid = render_version_selector("回归", list_regression_versions, activate_regression_version, delete_regression_version)

# Auto-detect saved model — reloads when active version changes
status = regression_status()
render_ml_status_bar(numeric_df, backend_synced, status, "回归")
current_vid = status.get("version_id", "") if status else ""
if "reg_result" not in st.session_state or st.session_state.get("reg_version_id") != current_vid:
    if status and status.get("has_model"):
        st.session_state.reg_result = {
            "r2": status.get("metrics", {}).get("r2", 0),
            "mae": status.get("metrics", {}).get("mae", 0),
            "rmse": status.get("metrics", {}).get("rmse", 0),
            "train_losses": [], "val_losses": [], "restored": True
        }
        st.session_state.reg_features = status.get("features", [])
        st.session_state.reg_target = status.get("target", "")
        st.session_state.reg_version_id = current_vid
        ds = status.get("dataset_name", "")
        created = status.get("created_at", "")[:16].replace("T", " ")
        st.success(f"已加载模型版本（{ds} | {created} | 目标列：{st.session_state.reg_target}）")

render_section_header("训练配置", "选择目标列和特征列，系统会提示样本量和常量特征风险。")
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
    render_small_dataset_warning(n_samples)
    check_constant_features(numeric_df, feature_cols)

with col2:
    if n_samples < 500:
        st.caption(f"小数据集 → 轻量网络 (h1={max(8, n_features)}, h2={max(4, n_features//2)}, dropout=0.1)")
    elif n_samples < 5000:
        st.caption(f"中数据集 → 标准网络 (h1={n_features*2}, h2={n_features}, dropout=0.2)")
    else:
        st.caption(f"大数据集 → 深层网络 (h1={n_features*3}, h2={n_features*2}, dropout=0.3)")

render_ml_workbench_overview(
    "回归",
    numeric_df,
    backend_synced=backend_synced,
    model_status=status,
    task_text="回归",
    target_text=str(target_col),
    feature_count=n_features,
    sample_count=n_samples,
    extra_items=[("设备", str(device)), ("Batch", "训练参数中设置")],
)

render_section_header("训练参数", "设置学习率、训练轮数和 batch size。")
hp_col1, hp_col2, hp_col3 = st.columns(3)
with hp_col1:
    learning_rate = st.selectbox("学习率 (LR)", [0.01, 0.005, 0.001, 0.0005, 0.0001], index=2)
with hp_col2:
    epochs = st.slider("最大训练轮数", 20, 500, 100, 20)
with hp_col3:
    batch_size = st.selectbox("批次大小", [4, 8, 16, 32, 64, 128], index=1)

render_section_header("模型训练", "启动训练或清除当前回归模型。")
train_col, clear_col = st.columns(2)

with train_col:
    if st.button("开始训练 / 重新训练模型", type="primary", use_container_width=True):
        with st.spinner("训练中（后端 Flask 计算中）..."):
            sid = ensure_session(numeric_df)
            if not sid:
                st.toast("无法连接到 Flask 后端 (http://localhost:5001)。请确保后端已启动。", icon="❌")
                st.stop()
            device_str = "cuda" if "CUDA" in str(device) else "cpu"
            result = train_regression(
                sid,
                str(target_col),
                [str(c) for c in feature_cols],
                learning_rate, epochs, batch_size, device_str
            )
            if result and "r2" in result:
                st.session_state.reg_result = result
                st.session_state.reg_features = feature_cols
                st.session_state.reg_target = target_col
                st.session_state.reg_version_id = result.get("version_id", "")

                st.toast(f"训练完成！R² = {result['r2']:.4f} | MAE = {result['mae']:.4f} | RMSE = {result['rmse']:.4f} | 版本: {result.get('version_id', '?')[:20]}...", icon="✅")
                plot_loss_curve(result["train_losses"], result["val_losses"], y_label="损失值 (MSE)")
            elif result is not None:
                st.toast("训练没有完成：后端返回的结果不完整，请查看后端终端日志。", icon="❌")

with clear_col:
    if st.button("清除已保存模型", use_container_width=True):
        clear_regression()
        for k in ["reg_result", "reg_features", "reg_target", "reg_version_id", "reg_last_prediction"]:
            if k in st.session_state:
                del st.session_state[k]
        st.warning("已清除所有保存的模型！")

render_section_header("数据预测", "使用当前激活版本进行单条预测。")
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
        version_id = st.session_state.get("reg_version_id")
        err = validate_input_array(np.array([input_data]), "单条预测", expected_features=len(features))
        if err:
            st.toast(err)
        else:
            result = predict_regression(input_data, device_str, version_id=version_id)
            if result and "result" in result:
                st.session_state.reg_last_prediction = {
                    "main": f"{result['result']:.4f}",
                    "details": [
                        ("预测目标", target),
                        ("模型版本", (version_id or "当前激活版本")[:24]),
                    ],
                }
                st.toast(f"预测结果：{result['result']:.4f}", icon="✅")

    if st.session_state.get("reg_last_prediction"):
        pred = st.session_state.reg_last_prediction
        if isinstance(pred, dict):
            render_stable_prediction_panel(
                "预测结果",
                pred["main"],
                details=pred["details"],
                model_status=status,
                fallback_result=st.session_state.get("reg_result"),
            )
        else:
            st.session_state.pop("reg_last_prediction", None)

    st.divider()
    render_section_header("批量预测", "上传包含相同特征列的 CSV 文件并批量生成预测结果。")
    batch_file = st.file_uploader("上传包含特征列的 CSV 文件", type=["csv"], key="reg_batch")
    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        missing_cols = set(features) - set(batch_df.columns)
        if missing_cols:
            st.toast(f"缺少特征列：{missing_cols}", icon="❌")
        else:
            batch_X = batch_df[features].values
            device_str = "cuda" if "CUDA" in str(device) else "cpu"
            version_id = st.session_state.get("reg_version_id")
            err = validate_input_array(batch_X, "批量预测", expected_features=len(features))
            if err:
                st.toast(err, icon="❌")
            else:
                result = batch_predict_regression(batch_X.tolist(), device_str, version_id=version_id)
                if result and "predictions" in result:
                    result_df = batch_df.copy()
                    result_df[f"预测_{target}"] = result["predictions"]
                    st.dataframe(result_df, use_container_width=True)
                    csv = result_df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 下载预测结果", csv, "predictions.csv", "text/csv", use_container_width=True)
