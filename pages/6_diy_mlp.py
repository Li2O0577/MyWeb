import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from pages._prepare import render_sidebar, data_uploader

#  页面基础配置 
st.set_page_config(page_title="Classification", layout="wide", initial_sidebar_state="collapsed")
# 隐藏侧边栏导航栏
st.markdown(
"""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)
render_sidebar("pages/5_classification.py")
st.title("🧠 Classification (Decision Model)")

# 模型存储配置 
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
# 分类模型相关路径
MODEL_PATH = os.path.join(MODEL_DIR, "cls_best_model.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "cls_scaler.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "cls_config.json")

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
    
# 功能说明 
with st.expander("📢 功能介绍", expanded=True):
    st.markdown("""
    ### 智能神经网络分类决策模型
    1. **任务场景**：二分类/多分类任务（标签判断、类别决策、结果分类等）
    2. **自适应设计**：自动识别类别数，适配Sigmoid(二分类)/Softmax(多分类)激活函数
    3. **防过拟合策略**：自适应网络结构 + Dropout + 早停机制 + L2正则化
    4. **硬件兼容**：全平台CPU适配，无环境依赖问题
    5. **模型持久化**：训练后自动保存，重启页面自动加载，无需重复训练
    """)

# 数据加载与校验 
df = data_uploader()
if df is None:
    st.warning("⚠️ 请先上传数据文件！")
    st.stop()

# 数据预处理：仅保留数值列 + 剔除缺失值
numeric_df = df.select_dtypes(include=[np.number]).dropna()
# 基础数据校验
if len(numeric_df) < 10 or len(numeric_df.columns) < 2:
    st.error("❌ 数据格式无效！要求：至少2列数值型数据 + 不少于10行有效数据")
    st.stop()

# 统一列名类型（兼容int/str混合列名场景）
col_dtype = type(numeric_df.columns[0])
col_options = [col for col in numeric_df.columns]

# 数据自动分析与模型配置 
st.subheader("📊 数据自动分析与模型配置")
col1, col2 = st.columns(2)

with col1:
    # 目标列/特征列选择
    target_col = st.selectbox("选择分类目标列（y）", col_options, index=len(col_options)-1)
    feature_cols = [col for col in col_options if col != target_col]
    # 基础数据维度统计
    n_features = len(feature_cols)
    n_samples = len(numeric_df)
    n_classes = len(numeric_df[target_col].unique())
    
    # 合法性校验
    if n_features == 0:
        st.warning("⚠️ 目标列覆盖所有列！无可用特征列，请重新选择目标列")
        st.stop()
    if n_classes < 2:
        st.warning(f"⚠️ 目标列「{target_col}」仅包含{n_classes}个类别，无法执行分类任务！")
        st.stop()
    if n_classes > n_samples * 0.5:
        st.warning(f"⚠️ 类别数({n_classes})接近样本数({n_samples})，易引发过拟合！建议更换目标列")
    
    # 数据维度信息展示
    st.info(f"✅ 数据概览：{n_samples} 行 | {n_features} 个特征 | {n_classes} 个类别")

with col2:
    # 自适应网络结构设计（根据样本量动态调整）
    if n_samples < 500:
        hidden1, hidden2 = max(8, n_features), max(4, n_features//2)
        dropout_rate = 0.1
    elif n_samples < 5000:
        hidden1, hidden2 = n_features*2, n_features
        dropout_rate = 0.2
    else:
        hidden1, hidden2 = n_features*3, n_features*2
        dropout_rate = 0.3

# 训练超参数 
test_size = 0.2    # 测试集比例
epochs = 100       # 最大训练轮数
patience = 10      # 早停耐心值

#  分类模型定义（MLP） 
class ClassificationNet(nn.Module):
    def __init__(self, input_dim, h1, h2, dropout_rate, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h2, num_classes)
        )
        # 激活函数自适应
        self.activation = nn.Sigmoid() if num_classes == 2 else nn.Softmax(dim=1)

    def forward(self, x):
        x = self.net(x)
        return self.activation(x)

#  模型加载函数
def load_saved_model():
    """加载本地保存的分类模型及配置"""
    if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, CONFIG_PATH]):
        try:
            # 加载配置文件
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 列名类型兼容转换
            saved_features = [col_dtype(col) for col in config['features']]
            saved_target = col_dtype(config['target'])
            saved_n_classes = config['n_classes']
            reverse_label_map = config.get('reverse_label_map', {})

            # 列名一致性校验
            if set(saved_features + [saved_target]) != set(col_options):
                st.warning("⚠️ 保存的模型列名与当前数据不匹配，无法加载！")
                return False

            # 加载模型/标准化器
            model = ClassificationNet(n_features, hidden1, hidden2, dropout_rate, saved_n_classes).to(device)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            
            # 模型状态存入session
            st.session_state.cls_model = model
            st.session_state.cls_scaler = scaler
            st.session_state.cls_features = saved_features
            st.session_state.cls_target = saved_target
            st.session_state.cls_classes = saved_n_classes
            st.session_state.reverse_label_map = reverse_label_map
            return True
        except Exception as e:
            st.error(f"模型加载失败：{str(e)}")
            return False
    return False

# 页面初始化时自动加载模型
if "cls_model" not in st.session_state:
    load_success = load_saved_model()
    if load_success:
        st.toast("✅ 自动加载已保存的分类模型！", icon="🎉")

#模型训练逻辑 
st.subheader("🚀 模型训练")
train_col, clear_col = st.columns(2)

with train_col:
    if st.button("开始训练 / 重新训练模型", type="primary", use_container_width=True):
        with st.spinner("分类模型训练中..."):
            # 类别样本数校验（每个类别至少2个样本）
            class_counts = numeric_df[target_col].value_counts()
            if class_counts.min() < 2:
                st.error("❌ 训练失败：每个类别至少需要2个有效样本！")
                st.stop()

            # 标签映射（转换为0开始的连续整数，避免CUDA索引越界）
            y_raw = numeric_df[target_col].values
            unique_labels = np.unique(y_raw)
            label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
            reverse_label_map = {i: lbl for lbl, i in label_map.items()}
            y = np.array([label_map[lbl] for lbl in y_raw])

            # 数据拆分（分层抽样，保证类别分布一致）
            X = numeric_df[feature_cols].values
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            # 数据标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 张量转换（CPU适配）
            X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

            # 数据加载器
            train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=8, shuffle=True)

            # 模型初始化
            model = ClassificationNet(n_features, hidden1, hidden2, dropout_rate, n_classes).to(device)
            # 损失函数自适应
            criterion = nn.CrossEntropyLoss() if n_classes > 2 else nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2正则

            # 早停机制初始化
            best_loss = float('inf')
            early_stop_count = 0
            best_model = model.state_dict()

            # 训练循环
            model.train()
            for epoch in range(epochs):
                # 批次训练
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    pred = model(batch_x)
                    # 二分类场景维度适配
                    if n_classes == 2:
                        pred = pred.squeeze()
                        batch_y = batch_y.float()
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    optimizer.step()

                # 验证集评估
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_test_tensor)
                    if n_classes == 2:
                        val_pred = val_pred.squeeze()
                        val_loss = criterion(val_pred, y_test_tensor.float())
                    else:
                        val_loss = criterion(val_pred, y_test_tensor)
                model.train()

                # 早停判断
                if val_loss < best_loss:
                    best_loss = val_loss
                    early_stop_count = 0
                    best_model = model.state_dict()
                else:
                    early_stop_count += 1
                    if early_stop_count >= patience:
                        st.info(f"⏹️ 早停触发！已训练 {epoch+1} 轮")
                        break

            # 加载最优模型
            model.load_state_dict(best_model)
            model.eval()

            # 模型评估（计算准确率）
            with torch.no_grad():
                y_pred = model(X_test_tensor)
                if n_classes == 2:
                    y_pred = (y_pred.squeeze() > 0.5).cpu().numpy()
                else:
                    y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
            acc = accuracy_score(y_test, y_pred)

            # 模型持久化保存
            torch.save(model.state_dict(), MODEL_PATH)
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(scaler, f)
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump({
                    "features": [str(col) for col in feature_cols],
                    "target": str(target_col),
                    "n_classes": n_classes,
                    "label_map": label_map,
                    "reverse_label_map": reverse_label_map
                }, f, ensure_ascii=False)

            # 训练结果存入session
            st.session_state.cls_model = model
            st.session_state.cls_scaler = scaler
            st.session_state.cls_features = feature_cols
            st.session_state.cls_target = target_col
            st.session_state.cls_classes = n_classes
            st.session_state.reverse_label_map = reverse_label_map

            st.success(f"训练完成！准确率 = {acc:.4f}")

with clear_col:
    # 清除已保存模型
    if st.button("清除已保存分类模型", use_container_width=True):
        # 删除模型文件
        for p in [MODEL_PATH, SCALER_PATH, CONFIG_PATH]:
            if os.path.exists(p):
                os.remove(p)
        # 清空session状态
        keys = ["cls_model", "cls_scaler", "cls_features", "cls_target", "cls_classes", "reverse_label_map"]
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]
        st.warning("✅ 已清除所有分类模型文件及缓存！")

# 模型预测功能
st.subheader("🎯 决策预测")
if "cls_model" not in st.session_state:
    st.warning("⚠️ 请先完成模型训练！")
else:
    # 加载session中的模型状态
    model = st.session_state.cls_model
    scaler = st.session_state.cls_scaler
    features = st.session_state.cls_features
    target = st.session_state.cls_target
    n_classes = st.session_state.cls_classes
    reverse_label_map = st.session_state.reverse_label_map

    st.info(f"✅ 模型加载成功 | 分类目标：{target} | 类别数：{n_classes}")
    st.write("请输入特征值进行决策预测：")
    
    # 特征输入框（按列排布）
    input_data = []
    cols = st.columns(len(features))
    for i, col in enumerate(cols):
        val = col.number_input(f"特征 {features[i]}", value=0.0, step=0.1)
        input_data.append(val)

    # 预测执行
    if st.button("执行决策预测", use_container_width=True):
        model.eval()
        with torch.no_grad():
            # 数据标准化 + 张量转换
            input_arr = np.array([input_data])
            input_scaled = scaler.transform(input_arr)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
            output = model(input_tensor)

            # 预测结果解析
            if n_classes == 2:
                prob = output.item()
                pred_idx = 1 if prob > 0.5 else 0
            else:
                prob = torch.max(output).item()
                pred_idx = torch.argmax(output, dim=1).item()

            # 还原原始标签
            pred_class = reverse_label_map.get(pred_idx, pred_idx)
            st.success(f"🎯 预测类别：{pred_class} | 置信度：{prob:.4f}")