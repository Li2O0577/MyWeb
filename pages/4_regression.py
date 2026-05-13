import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import json
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from pages._prepare import render_sidebar, data_uploader

# 好多用的包。。

# 1. 配置页面，隐藏侧边栏导航 
st.set_page_config(page_title="Regression", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
"""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)
render_sidebar("pages/4_regression.py")
st.title("🧠 Regression (Prediction)")

# 2. 数据上传与预处理组件
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "reg_best_model.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "reg_scaler.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "reg_config.json")

st.subheader("⚙️ 计算设备选择")
device_col1, device_col2 = st.columns(2)
with device_col1:
    device_choice = st.selectbox("选择运行设备", ["CPU", "CUDA (GPU)"], index=0)
# 设备校验逻辑（完全匹配分类页面，不符合立即报错）
if device_choice == "CUDA (GPU)":
    if not torch.cuda.is_available():
        st.error("❌ 设备选择错误：当前环境未安装CUDA/GPU，无法使用CUDA！请切换为CPU")
        st.stop()
    device = torch.device("cuda")
    st.success("✅ 已启用：CUDA GPU")
else:
    device = torch.device("cpu")
    st.info("⚙️ 已启用：CPU")
#具体功能

#  1. 功能介绍 
with st.expander("📢 功能介绍", expanded=True):
    st.markdown("""
    ### 智能神经网络回归预测
    1. **任务说明**：预测**连续数值**（价格、销量、温度、浓度等）
    2. **智能建模**：自动分析数据规模，生成**防过拟合/欠拟合**的最优网络结构
    3. **硬件自适应**：自动检测GPU(CUDA)/CPU，自动选择最优训练设备
    4. **模型持久化**：训练后自动保存模型，重启页面**自动加载**，无需重复训练
    5. **本地模型**：模型为本地运行，安全稳定
    """)

#  加载数据
df = data_uploader()
if df is None:
    st.warning("⚠️ 请先上传数据！")
    st.stop()

# 数据预处理
numeric_df = df.select_dtypes(include=[np.number]).dropna()
if len(numeric_df) < 10 or len(numeric_df.columns) < 2:
    st.error("❌ 数据无效！需要至少2列数值数据 + 10行数据")
    st.stop()

#2. 自动数据分析 + 智能网络结构（防过拟合/欠拟合） 
st.subheader("📊 数据自动分析与模型配置")
col1, col2 = st.columns(2)

with col1:
    target_col = st.selectbox("选择预测目标列（y）", numeric_df.columns, index=len(numeric_df.columns)-1)
    feature_cols = [col for col in numeric_df.columns if col != target_col]
    n_features = len(feature_cols)
    n_samples = len(numeric_df)
    st.info(f"✅ 数据：{n_samples} 行 | {n_features} 个特征")

with col2:
    # 自适应网络设计 
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
    batch_size = st.selectbox("Batch Size", [4, 8, 16, 32, 64, 128], index=1)

test_size = 0.2
patience = 10  # 早停：10轮不提升就停止

# PyTorch 防过拟合模型
class RegressionNet(nn.Module):
    def __init__(self, input_dim, h1, h2, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 防过拟合
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 防过拟合
            nn.Linear(h2, 1)
        )
    def forward(self, x):
        return self.net(x)

# 自动加载已保存的模型 
def load_saved_model():
    if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, CONFIG_PATH]):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)

            saved_features = config['features']
            saved_target = config['target']
            current_cols = list(numeric_df.columns)

            # 列名校验：保存的列必须与当前数据完全匹配
            if set(saved_features + [saved_target]) != set(current_cols):
                st.warning("⚠️ 保存的模型列名与当前数据不匹配，无法加载！")
                return False

            # 加载模型
            model = RegressionNet(n_features, hidden1, hidden2, dropout_rate).to(device)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            # 加载标准化器
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)

            st.session_state.reg_model = model
            st.session_state.reg_scaler = scaler
            st.session_state.reg_features = saved_features
            st.session_state.reg_target = saved_target
            return True
        except:
            return False
    return False

# 页面加载时自动加载模型
if "reg_model" not in st.session_state:
    load_success = load_saved_model()
    if load_success:
        st.toast("✅ 自动加载已保存的模型！", icon="🎉")

# ===================== 模型训练 =====================
st.subheader("🚀 模型训练")
train_col, clear_col = st.columns(2)
with train_col:
    if st.button("开始训练 / 重新训练模型", type="primary", use_container_width=True):
        with st.spinner("训练中（自动防过拟合+硬件加速）..."):
            # 数据处理 — 三层拆分：训练集 / 验证集(早停) / 测试集(最终评估)
            X = numeric_df[feature_cols].values
            y = numeric_df[target_col].values.reshape(-1, 1)

            # 先留出 20% 作为最终测试集
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # 再从剩余数据中分 20% 作为验证集（占总数据 16%）
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

            # 张量转换
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

            st.info(f"📊 数据拆分：训练集 {len(X_train)} | 验证集 {len(X_val)} | 测试集 {len(X_test)}")

            # 自适应 batch_size：不超过训练集大小
            actual_batch = min(batch_size, len(X_train))
            if actual_batch != batch_size:
                st.info(f"⚠️ 训练集较小，batch_size 自动调整为 {actual_batch}")
            train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                                      batch_size=actual_batch, shuffle=True)

            # 初始化模型（局部变量）
            model = RegressionNet(n_features, hidden1, hidden2, dropout_rate).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # L2正则

            # 早停机制 + 损失记录
            best_loss = float('inf')
            early_stop_count = 0
            train_losses, val_losses = [], []

            progress_bar = st.progress(0)
            status_text = st.empty()

            # 训练循环
            model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(batch_x), batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_train_loss = epoch_loss / len(train_loader)

                # 验证（使用验证集，不参与训练和最终评估）
                model.eval()
                with torch.no_grad():
                    val_loss = criterion(model(X_val_tensor), y_val_tensor).item()
                model.train()

                train_losses.append(avg_train_loss)
                val_losses.append(val_loss)

                # 进度条
                progress_bar.progress((epoch + 1) / epochs)
                status_text.text(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

                # 早停判断
                if val_loss < best_loss:
                    best_loss = val_loss
                    early_stop_count = 0
                    best_state = model.state_dict()
                else:
                    early_stop_count += 1
                    if early_stop_count >= patience:
                        st.info(f"⏹️ 早停触发！已训练 {epoch+1} 轮，防止过拟合")
                        break

            progress_bar.empty()
            status_text.empty()

            # 加载最优权重
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test_tensor).cpu().numpy()
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            #  保存模型到本地
            torch.save(model.state_dict(), MODEL_PATH)
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(scaler, f)
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump({"features": feature_cols, "target": target_col}, f, ensure_ascii=False)

            # 保存到会话
            st.session_state.reg_model = model
            st.session_state.reg_scaler = scaler
            st.session_state.reg_features = feature_cols
            st.session_state.reg_target = target_col

            st.success(f"训练完成！模型已自动保存 | R² = {r2:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f}")

            # 训练/验证损失曲线
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='训练损失', line=dict(color='#1f77b4')))
            fig.add_trace(go.Scatter(y=val_losses, mode='lines', name='验证损失', line=dict(color='#ff7f0e')))
            fig.update_layout(title="训练 & 验证损失曲线", xaxis_title="Epoch", yaxis_title="Loss (MSE)",
                              template="plotly_white", height=350, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

# 清除模型按钮
with clear_col:
    if st.button("清除已保存模型", use_container_width=True):
        for p in [MODEL_PATH, SCALER_PATH, CONFIG_PATH]:
            if os.path.exists(p):
                os.remove(p)
        for k in ["reg_model", "reg_scaler", "reg_features", "reg_target"]:
            if k in st.session_state:
                del st.session_state[k]
        st.warning("已清除所有保存的模型！")

#  预测功能 
st.subheader("🎯 数据预测")
if "reg_model" not in st.session_state:
    st.warning("请先训练模型，模型会自动保存并加载~")
else:
    model = st.session_state.reg_model
    scaler = st.session_state.reg_scaler
    features = st.session_state.reg_features
    target = st.session_state.reg_target

    st.info(f"✅ 加载模型成功 | 预测目标：{target}")
    st.write("请输入特征值进行预测：")
    input_data = []
    cols = st.columns(len(features))
    
    for i, col in enumerate(cols):
        val = col.number_input(f"{features[i]}", value=0.0, step=0.1)
        input_data.append(val)

    if st.button("执行预测", use_container_width=True):
        model.to(device)
        model.eval()
        with torch.no_grad():
            input_arr = np.array([input_data])
            input_scaled = scaler.transform(input_arr)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
            pred = model(input_tensor).item()
            st.success(f"预测结果：{pred:.4f}")

    # 批量预测
    st.divider()
    st.subheader("📦 批量预测 (CSV)")
    batch_file = st.file_uploader("上传包含特征列的 CSV 文件", type=["csv"], key="reg_batch")
    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        missing_cols = set(features) - set(batch_df.columns)
        if missing_cols:
            st.error(f"缺少特征列：{missing_cols}")
        else:
            batch_X = batch_df[features].values
            model.to(device)
            model.eval()
            with torch.no_grad():
                batch_scaled = scaler.transform(batch_X)
                batch_tensor = torch.tensor(batch_scaled, dtype=torch.float32).to(device)
                batch_pred = model(batch_tensor).cpu().numpy().flatten()
            result_df = batch_df.copy()
            result_df[f"预测_{target}"] = batch_pred
            st.dataframe(result_df, use_container_width=True)
            csv = result_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下载预测结果 CSV", csv, "predictions.csv", "text/csv", use_container_width=True)