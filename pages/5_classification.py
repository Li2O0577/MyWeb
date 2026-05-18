import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import json
import copy
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader
from pages._prepare import render_sidebar, data_uploader
from pages._mlp_common import (render_device_selector, check_constant_features,
                                plot_loss_curve, save_model_files, clear_model_files,
                                validate_input_array)

# 页面配置
st.set_page_config(page_title="分类决策", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
"""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)
render_sidebar("pages/5_classification.py")
st.title("🧠 分类决策")

# 模型保存路径
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "cls_best_model.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "cls_scaler.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "cls_config.json")

device = render_device_selector()

# 功能介绍
with st.expander("📢 功能介绍", expanded=True):
    st.markdown("""
    ### 智能神经网络分类决策模型
    1. **任务说明**：分类预测（二分类/多分类，如标签判断、类别决策、结果分类）
    2. **自动适配**：自动识别二分类(Sigmoid) / 多分类(Softmax)
    3. **防过拟合**：自适应网络结构 + Dropout + 早停 + L2正则
    4. **硬件自适应**：GPU/CPU自动切换
    5. **模型持久化**：自动保存/加载，无需重复训练
    """)

# 加载数据
df = data_uploader()
if df is None:
    st.warning("⚠️ 请先上传数据！")
    st.stop()

# 数据预处理
numeric_df = df.select_dtypes(include=[np.number]).dropna()
if len(numeric_df) < 10 or len(numeric_df.columns) < 2:
    st.error("❌ 数据无效！需要至少2列数值数据 + 10行数据")
    st.stop()

# 统一列名类型，兼容int/str列名
col_dtype = type(numeric_df.columns[0])
col_options = [col for col in numeric_df.columns]

# 自动数据分析 + 智能网络结构
st.subheader("📊 数据自动分析与模型配置")
col1, col2 = st.columns(2)

with col1:
    target_col = st.selectbox("选择分类目标列（y）", col_options, index=len(col_options)-1)
    feature_cols = [col for col in col_options if col != target_col]
    n_features = len(feature_cols)
    n_samples = len(numeric_df)
    
    n_classes = len(numeric_df[target_col].unique())
    
    # 列选择合法性校验
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

with col2:
    # 自适应网络结构
    if n_samples < 500:
        hidden1, hidden2 = max(8, n_features), max(4, n_features//2)
        dropout_rate = 0.1
    elif n_samples < 5000:
        hidden1, hidden2 = n_features*2, n_features
        dropout_rate = 0.2
    else:
        hidden1, hidden2 = n_features*3, n_features*2
        dropout_rate = 0.3

# 训练参数
st.subheader("⚡ 训练参数")
hp_col1, hp_col2, hp_col3 = st.columns(3)
with hp_col1:
    learning_rate = st.selectbox("学习率 (LR)", [0.01, 0.005, 0.001, 0.0005, 0.0001], index=2)
with hp_col2:
    epochs = st.slider("最大训练轮数", 20, 500, 100, 20)
with hp_col3:
    batch_size = st.selectbox("批次大小", [4, 8, 16, 32, 64, 128], index=1)

test_size = 0.2
patience = 10

# 分类模型（MLP，输出 logits，损失函数自带 Sigmoid/Softmax）
class ClassificationNet(nn.Module):
    def __init__(self, input_dim, h1, h2, dropout_rate, num_classes):
        super().__init__()
        output_dim = 1 if num_classes == 2 else num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h2, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# 自动加载模型
def load_saved_model():
    if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, CONFIG_PATH]):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            saved_features = [col_dtype(col) for col in config['features']]
            saved_target = col_dtype(config['target'])
            saved_n_classes = config['n_classes']
            reverse_label_map = config.get('reverse_label_map', {})

            if set(saved_features + [saved_target]) != set(col_options):
                st.warning("⚠️ 保存的模型列名与当前数据不匹配，无法加载！")
                return False

            model = ClassificationNet(n_features, hidden1, hidden2, dropout_rate, saved_n_classes).to(device)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            
            st.session_state.cls_model = model
            st.session_state.cls_scaler = scaler
            st.session_state.cls_features = saved_features
            st.session_state.cls_target = saved_target
            st.session_state.cls_classes = saved_n_classes
            st.session_state.reverse_label_map = reverse_label_map
            return True
        except Exception as e:
            st.error(f"加载模型失败：{str(e)}")
            return False
    return False

# 页面加载自动加载模型
if "cls_model" not in st.session_state:
    load_success = load_saved_model()
    if load_success:
        st.toast("✅ 自动加载已保存的分类模型！", icon="🎉")

# 模型训练（完整修复版）
st.subheader("🚀 模型训练")
train_col, clear_col = st.columns(2)
with train_col:
    if st.button("开始训练 / 重新训练模型", type="primary", use_container_width=True):
        with st.spinner("分类模型训练中..."):
            # 校验：每个类别至少2个样本
            class_counts = numeric_df[target_col].value_counts()
            if class_counts.min() < 2:
                st.error("❌ 训练失败：每个类别至少需要2个样本！")
                st.stop()

            # 标签映射为0开始的连续整数（解决CUDA越界）
            y_raw = numeric_df[target_col].values
            unique_labels = np.unique(y_raw)
            label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
            reverse_label_map = {i: lbl for lbl, i in label_map.items()}
            y = np.array([label_map[lbl] for lbl in y_raw])

            # 数据处理 — 三层拆分（分层抽样）：训练集 / 验证集(早停) / 测试集(最终评估)
            X = numeric_df[feature_cols].values

            # 先留出 20% 作为最终测试集
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            # 再从剩余数据中分 20% 作为验证集（stratify 保证类别分布一致）
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            # 张量转换
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

            st.info(f"📊 数据拆分：训练集 {len(X_train)} | 验证集 {len(X_val)} | 测试集 {len(X_test)}")

            # 自适应 batch_size
            actual_batch = min(batch_size, len(X_train))
            if actual_batch != batch_size:
                st.info(f"⚠️ 训练集较小，batch_size 自动调整为 {actual_batch}")
            train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                                      batch_size=actual_batch, shuffle=True)

            # 模型初始化
            model = ClassificationNet(n_features, hidden1, hidden2, dropout_rate, n_classes).to(device)
            criterion = nn.CrossEntropyLoss() if n_classes > 2 else nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

            # 早停机制 + 损失记录
            best_loss = float('inf')
            early_stop_count = 0
            best_state = copy.deepcopy(model.state_dict())
            train_losses, val_losses = [], []

            progress_bar = st.progress(0)
            status_text = st.empty()

            # 训练循环
            model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    pred = model(batch_x)
                    if n_classes == 2:
                        pred = pred.squeeze(1)
                        batch_y = batch_y.float()
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_train_loss = epoch_loss / len(train_loader)

                # 验证（使用验证集，不参与最终评估）
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val_tensor)
                    if n_classes == 2:
                        val_pred = val_pred.squeeze(1)
                        val_loss = criterion(val_pred, y_val_tensor.float()).item()
                    else:
                        val_loss = criterion(val_pred, y_val_tensor).item()
                model.train()

                train_losses.append(avg_train_loss)
                val_losses.append(val_loss)

                progress_bar.progress((epoch + 1) / epochs)
                status_text.text(f"轮次 {epoch+1}/{epochs} | 训练损失: {avg_train_loss:.4f} | 验证损失: {val_loss:.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    early_stop_count = 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    early_stop_count += 1
                    if early_stop_count >= patience:
                        st.info(f"⏹️ 早停触发！已训练 {epoch+1} 轮")
                        break

            progress_bar.empty()
            status_text.empty()

            model.load_state_dict(best_state)
            model.eval()

            # 评估
            with torch.no_grad():
                y_pred = model(X_test_tensor)
                if n_classes == 2:
                    y_pred = (torch.sigmoid(y_pred).view(-1) > 0.5).cpu().numpy()
                else:
                    y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()

            acc = accuracy_score(y_test, y_pred)

            # 混淆矩阵 + 分类报告
            cm = confusion_matrix(y_test, y_pred)
            unique_test_labels = sorted(set(y_test) | set(y_pred))
            label_names = [reverse_label_map.get(l, reverse_label_map.get(str(l), l)) for l in unique_test_labels]

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

            with col_report:
                st.caption("分类报告")
                try:
                    report = classification_report(y_test, y_pred, target_names=[str(n) for n in label_names],
                                                   output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).transpose()
                    # 只显示 precision/recall/f1-score/support
                    display_df = report_df[['precision', 'recall', 'f1-score', 'support']]
                    st.dataframe(display_df.style.format("{:.3f}", subset=['precision', 'recall', 'f1-score']),
                                 use_container_width=True)
                except Exception:
                    st.text(classification_report(y_test, y_pred, zero_division=0))

            # 训练/验证损失曲线
            plot_loss_curve(train_losses, val_losses)

            # 保存模型
            save_model_files(model, scaler, {
                "features": [str(col) for col in feature_cols],
                "target": str(target_col),
                "n_classes": n_classes,
                "label_map": {str(k): v for k, v in label_map.items()},
                "reverse_label_map": {str(k): str(v) for k, v in reverse_label_map.items()}
            }, MODEL_PATH, SCALER_PATH, CONFIG_PATH)

            # 保存到会话（键统一为字符串，兼容预测查找）
            st.session_state.cls_model = model
            st.session_state.cls_scaler = scaler
            st.session_state.cls_features = feature_cols
            st.session_state.cls_target = target_col
            st.session_state.cls_classes = n_classes
            st.session_state.reverse_label_map = {str(k): v for k, v in reverse_label_map.items()}

            st.success(f"训练完成！准确率 = {acc:.4f}")

# 清除模型
with clear_col:
    if st.button("清除已保存分类模型", use_container_width=True):
        clear_model_files(
            [MODEL_PATH, SCALER_PATH, CONFIG_PATH],
            ["cls_model", "cls_scaler", "cls_features", "cls_target", "cls_classes", "reverse_label_map"]
        )
        st.warning("已清除分类模型！")

# 预测功能（修复标签映射）
st.subheader("🎯 决策预测")
if "cls_model" not in st.session_state:
    st.warning("请先训练模型！")
else:
    model = st.session_state.cls_model
    scaler = st.session_state.cls_scaler
    features = st.session_state.cls_features
    target = st.session_state.cls_target
    n_classes = st.session_state.cls_classes
    reverse_label_map = st.session_state.reverse_label_map

    st.info(f"✅ 模型加载成功 | 分类目标：{target} | 类别数：{n_classes}")
    st.write("请输入特征值进行决策预测：")
    input_data = []
    cols = st.columns(len(features))

    for i, col in enumerate(cols):
        val = col.number_input(f"特征 {features[i]}", value=0.0, step=0.1)
        input_data.append(val)

    if st.button("执行决策预测", use_container_width=True):
        model.to(device)
        model.eval()
        with torch.no_grad():
            input_arr = np.array([input_data])
            err = validate_input_array(input_arr, "单条预测")
            if err:
                st.error(err)
                st.stop()
            input_scaled = scaler.transform(input_arr)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
            output = model(input_tensor)

            if n_classes == 2:
                prob = torch.sigmoid(output).squeeze().item()
                pred_idx = 1 if prob > 0.5 else 0
            else:
                prob = torch.softmax(output, dim=1).max().item()
                pred_idx = torch.argmax(output, dim=1).item()

            # 还原原始标签
            pred_class = reverse_label_map.get(str(pred_idx), pred_idx)
            st.success(f"🎯 预测类别：{pred_class} | 置信度：{prob:.4f}")

    # 批量预测
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
                model.to(device)
                model.eval()
                with torch.no_grad():
                    batch_scaled = scaler.transform(batch_X)
                    batch_tensor = torch.tensor(batch_scaled, dtype=torch.float32).to(device)
                    output = model(batch_tensor)

                    if n_classes == 2:
                        probs = torch.sigmoid(output).cpu().numpy().flatten()
                        pred_indices = (probs > 0.5).astype(int)
                        confidences = np.where(pred_indices == 1, probs, 1 - probs)
                    else:
                        probs_all = torch.softmax(output, dim=1).cpu().numpy()
                        pred_indices = np.argmax(probs_all, axis=1)
                        confidences = probs_all[np.arange(len(pred_indices)), pred_indices]

                result_df = batch_df.copy()
                result_df["预测类别"] = [reverse_label_map.get(str(i), i) for i in pred_indices]
                result_df["置信度"] = confidences
                st.dataframe(result_df, use_container_width=True)
                csv = result_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 下载预测结果", csv, "predictions.csv", "text/csv", use_container_width=True)
