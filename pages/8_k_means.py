import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pages._prepare import render_sidebar, data_uploader

# 页面配置
st.set_page_config(page_title="K-means 聚类", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)
render_sidebar("pages/8_k_means.py")
st.title("📊 K-means 聚类")

# 模型保存路径（统一存放）
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "kmeans_scaler.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "kmeans_config.json")

# 功能介绍
with st.expander("📢 功能介绍", expanded=True):
    st.markdown("""
    ### KMeans 无监督聚类模型
    1. **任务说明**：自动将数据分为 K 个簇，无需标签
    2. **数据标准化**：自动对特征进行标准化，提升聚类效果
    3. **聚类可视化**：PCA 降维展示聚类分布
    4. **模型持久化**：训练后自动保存，重启页面一键加载
    5. **聚类预测**：输入新数据，自动判断所属簇类别
    """)

# 加载数据
df = data_uploader()
if df is None:
    st.warning("⚠️ 请先上传数据！")
    st.stop()

# 数据预处理
numeric_df = df.select_dtypes(include=[np.number]).dropna()
if len(numeric_df) < 10 or numeric_df.shape[1] < 1:
    st.error("❌ 数据无效！需要至少10行数值数据")
    st.stop()

# 统一列名类型
col_dtype = type(numeric_df.columns[0])
feature_cols = [col for col in numeric_df.columns]
n_features = len(feature_cols)
n_samples = len(numeric_df)

# 聚类配置
st.subheader("⚙️ 聚类参数配置")
col1, col2 = st.columns(2)
with col1:
    n_clusters = st.number_input("设置聚类数量 K", min_value=2, max_value=10, value=3, step=1)
    st.info(f"✅ 数据：{n_samples} 行 | {n_features} 个特征 | 聚类数 K={n_clusters}")
with col2:
    st.info("💡 提示：K 值越小，簇数量越少；建议根据业务需求选择")

#  模型加载函数 
def load_saved_kmeans():
    if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, CONFIG_PATH]):
        try:
            # 加载配置
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
            saved_features = [col_dtype(col) for col in config['features']]
            saved_k = config['n_clusters']

            # 校验列匹配
            if set(saved_features) != set(feature_cols):
                st.warning("⚠️ 模型特征与当前数据不匹配！")
                return False

            # 加载模型
            with open(MODEL_PATH, 'rb') as f:
                kmeans_model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)

            # 保存到会话
            st.session_state.kmeans_model = kmeans_model
            st.session_state.kmeans_scaler = scaler
            st.session_state.kmeans_features = saved_features
            st.session_state.kmeans_clusters = saved_k
            return True
        except Exception as e:
            st.error(f"加载模型失败：{str(e)}")
            return False
    return False

# 自动加载模型
if "kmeans_model" not in st.session_state:
    load_success = load_saved_kmeans()
    if load_success:
        st.toast("✅ 自动加载已保存的 KMeans 模型！", icon="🎉")

# 模型训练 
st.subheader("🚀 模型训练")
train_col, clear_col = st.columns(2)
with train_col:
    if st.button("开始聚类训练", type="primary", use_container_width=True):
        with st.spinner("KMeans 聚类训练中..."):
            # 数据标准化
            X = numeric_df[feature_cols].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 训练 KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(X_scaled)

            # 保存结果到会话
            st.session_state.kmeans_model = kmeans
            st.session_state.kmeans_scaler = scaler
            st.session_state.kmeans_features = feature_cols
            st.session_state.kmeans_clusters = n_clusters
            st.session_state.cluster_data = numeric_df.copy()
            st.session_state.cluster_data["聚类标签"] = cluster_labels

            # 保存模型文件
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(kmeans, f)
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(scaler, f)
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump({
                    "features": [str(col) for col in feature_cols],
                    "n_clusters": n_clusters
                }, f, ensure_ascii=False)

            st.success(f"✅ 聚类完成！数据已分为 {n_clusters} 个簇")

# 清除模型
with clear_col:
    if st.button("清除已保存聚类模型", use_container_width=True):
        for p in [MODEL_PATH, SCALER_PATH, CONFIG_PATH]:
            if os.path.exists(p):
                os.remove(p)
        keys = ["kmeans_model", "kmeans_scaler", "kmeans_features", "kmeans_clusters", "cluster_data"]
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]
        st.warning("已清除 KMeans 模型！")

# 聚类结果展示 
st.subheader("📈 聚类结果展示")
if "cluster_data" in st.session_state:
    data = st.session_state.cluster_data
    k = st.session_state.kmeans_clusters

    # 展示带标签的数据
    st.markdown("#### 聚类结果数据")
    st.dataframe(data.head(15), use_container_width=True)

    # 统计每个簇的样本数量
    st.markdown("#### 簇样本数量统计")
    count_df = data["聚类标签"].value_counts().sort_index()
    count_df = count_df.rename("样本数量").to_frame()
    count_df.index.name = "簇标签"
    st.dataframe(count_df, use_container_width=True)

    # 聚类可视化（PCA降维）
    st.markdown("#### 聚类分布可视化")
    X = data[feature_cols].values
    X_scaled = st.session_state.kmeans_scaler.transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=data["聚类标签"], cmap="viridis", s=50)
    ax.set_title(f"K-means 聚类可视化 (K={k})")
    ax.set_xlabel("主成分 1")
    ax.set_ylabel("主成分 2")
    plt.colorbar(scatter, label="簇标签")
    st.pyplot(fig)

#  聚类预测 
st.subheader("🎯 聚类预测")
if "kmeans_model" not in st.session_state:
    st.warning("请先训练 KMeans 模型！")
else:
    model = st.session_state.kmeans_model
    scaler = st.session_state.kmeans_scaler
    features = st.session_state.kmeans_features
    k = st.session_state.kmeans_clusters

    st.info(f"✅ 模型加载成功 | 聚类数 K={k}")
    st.write("请输入特征值，预测所属簇：")

    # 生成输入框
    input_data = []
    cols = st.columns(len(features))
    for i, col in enumerate(cols):
        val = col.number_input(f"特征 {features[i]}", value=0.0, step=0.1)
        input_data.append(val)

    if st.button("执行聚类预测", use_container_width=True):
        with st.no_rerun():
            # 数据预处理 + 预测
            input_arr = np.array([input_data])
            input_scaled = scaler.transform(input_arr)
            pred_cluster = model.predict(input_scaled)[0]

            st.success(f"🎯 预测结果：该数据属于 **簇 {pred_cluster}**")