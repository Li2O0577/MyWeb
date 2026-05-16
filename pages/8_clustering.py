import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pages._prepare import render_sidebar, data_uploader
import plotly.graph_objects as go

# 页面配置
st.set_page_config(page_title="聚类分析", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
""", unsafe_allow_html=True)
render_sidebar("pages/8_clustering.py")
st.title("🧪 聚类分析")

# 模型保存路径
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "cluster_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "cluster_scaler.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "cluster_config.json")

# 功能介绍
with st.expander("📢 功能介绍", expanded=True):
    st.markdown("""
    ### 无监督聚类模型 — K-means & DBSCAN
    1. **双算法支持**：K-means（球状簇、需预设 K）+ DBSCAN（任意形状、自动发现簇数、识别噪声）
    2. **肘部法则**（K-means）：自动遍历 K 值计算 Inertia，绘制肘部曲线辅助选择最优 K
    3. **质量评估**：轮廓系数（Silhouette Score），评估聚类效果
    4. **数据标准化**：自动对特征进行标准化，提升聚类效果
    5. **聚类可视化**：PCA 降维 + Plotly 交互式散点图展示聚类分布
    6. **模型持久化**：训练后自动保存，重启页面一键加载
    7. **聚类预测**（K-means）：输入新数据，自动判断所属簇类别
    """)

# 加载数据
df = data_uploader()
if df is None:
    st.warning("⚠️ 请先上传数据！")
    st.stop()

numeric_df = df.select_dtypes(include=[np.number]).dropna()
if len(numeric_df) < 10 or numeric_df.shape[1] < 1:
    st.error("❌ 数据无效！需要至少10行数值数据")
    st.stop()

col_dtype = type(numeric_df.columns[0])
feature_cols = [col for col in numeric_df.columns]
n_features = len(feature_cols)
n_samples = len(numeric_df)

# 初始化算法类型
if "cluster_algorithm" not in st.session_state:
    st.session_state.cluster_algorithm = "kmeans"

# 算法选择
st.subheader("⚙️ 算法选择与参数配置")
alg_col, *param_cols = st.columns([1, 1, 1, 2])
with alg_col:
    algorithm = st.selectbox("聚类算法", ["K-means", "DBSCAN"],
                             index=0 if st.session_state.cluster_algorithm == "kmeans" else 1)
    st.session_state.cluster_algorithm = "kmeans" if algorithm == "K-means" else "dbscan"

is_kmeans = (st.session_state.cluster_algorithm == "kmeans")

# K-means 参数
if is_kmeans:
    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        n_clusters = st.number_input("聚类数量 K", min_value=2, max_value=15, value=3, step=1)
    with col2:
        st.info(f"✅ 数据：{n_samples} 行 | {n_features} 个特征 | K={n_clusters}")

    # 肘部法则（仅 K-means）
    st.divider()
    st.subheader("📈 肘部法则 — 选择最优 K 值")
    st.caption("遍历不同 K 值计算 Inertia（簇内平方和），曲线拐点即为建议的 K 值。")

    max_k_limit = min(15, max(3, n_samples // 3))
    elbow_col1, elbow_col2 = st.columns([1, 3])
    with elbow_col1:
        max_k_elbow = st.number_input("最大 K 值", min_value=3, max_value=max_k_limit, value=min(10, max_k_limit), step=1)
        run_elbow = st.button("运行肘部分析", use_container_width=True, type="secondary")

    if run_elbow:
        with st.spinner("正在计算不同 K 值的 Inertia..."):
            X_elbow = numeric_df[feature_cols].values
            scaler_elbow = StandardScaler()
            X_elbow_scaled = scaler_elbow.fit_transform(X_elbow)

            ks = list(range(1, max_k_elbow + 1))
            inertias = []
            for k_val in ks:
                km = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
                km.fit(X_elbow_scaled)
                inertias.append(km.inertia_)

            st.session_state.elbow_ks = ks
            st.session_state.elbow_inertias = inertias

    if "elbow_ks" in st.session_state:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=st.session_state.elbow_ks, y=st.session_state.elbow_inertias,
            mode='lines+markers', marker=dict(size=8, color='#636efa'),
            line=dict(width=2, color='#636efa'), name='Inertia'
        ))
        fig_elbow.update_layout(
            title="肘部法则 — Inertia vs K",
            xaxis=dict(title="聚类数量 K", dtick=1),
            yaxis_title="Inertia (簇内平方和)",
            template="plotly_white", height=380,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.caption("💡 曲线在某个 K 值后下降变缓，该拐点（\"肘部\"）就是推荐的 K 值。")

# DBSCAN 参数
else:
    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        eps = st.number_input("邻域半径 ε (eps)", min_value=0.05, max_value=5.0, value=0.5, step=0.05,
                              help="两点间最大距离，小于此距离视为邻居。值越大簇越少。")
    with col2:
        min_samples = st.number_input("最小样本数 (min_samples)", min_value=2, max_value=50, value=5, step=1,
                                       help="核心点所需的最小邻居数。值越大噪声点越多。")
    st.info(f"✅ 数据：{n_samples} 行 | {n_features} 个特征 | ε={eps} | min_samples={min_samples}")
    st.caption("💡 DBSCAN 自动发现簇数量，无需预设 K。标签为 -1 的点为噪声（不属于任何簇）。")

st.divider()

# 模型加载
def load_saved_model():
    if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, CONFIG_PATH]):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
            saved_features = [col_dtype(col) for col in config['features']]
            if set(saved_features) != set(feature_cols):
                st.warning("⚠️ 模型特征与当前数据不匹配！")
                return False

            with open(MODEL_PATH, 'rb') as f:
                cluster_model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)

            st.session_state.cluster_model = cluster_model
            st.session_state.cluster_scaler = scaler
            st.session_state.cluster_features = saved_features
            st.session_state.cluster_algorithm = config.get("algorithm", "kmeans")
            st.session_state.cluster_params = config.get("params", {})
            if config.get("algorithm", "kmeans") == "kmeans":
                st.session_state.cluster_n_clusters = config["params"]["n_clusters"]
            else:
                st.session_state.cluster_n_clusters = config.get("n_clusters_found", None)
            return True
        except Exception as e:
            st.error(f"加载模型失败：{str(e)}")
            return False
    return False

if "cluster_model" not in st.session_state:
    if load_saved_model():
        algo_name = "K-means" if st.session_state.cluster_algorithm == "kmeans" else "DBSCAN"
        st.toast(f"✅ 自动加载已保存的 {algo_name} 模型！", icon="🎉")

# 模型训练
st.subheader("🚀 模型训练")
train_col, clear_col = st.columns(2)
with train_col:
    btn_label = "开始聚类训练" if "cluster_model" not in st.session_state else "重新训练"
    if st.button(btn_label, type="primary", use_container_width=True):
        algo_label = "K-means" if is_kmeans else "DBSCAN"
        with st.spinner(f"{algo_label} 聚类训练中..."):
            X = numeric_df[feature_cols].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if is_kmeans:
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                cluster_labels = model.fit_predict(X_scaled)
                inertia = model.inertia_
                n_found = n_clusters
            else:
                model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
                cluster_labels = model.fit_predict(X_scaled)
                inertia = None
                n_found = len(set(cluster_labels) - {-1})

                if n_found == 0:
                    st.error("❌ DBSCAN 将所有点标记为噪声！请增大 eps 或减小 min_samples 后重试。")
                    st.stop()

            # 轮廓系数（排除噪声点）
            if is_kmeans or n_found >= 2:
                valid_mask = cluster_labels != -1
                if valid_mask.sum() >= 2:
                    if valid_mask.sum() <= 5000:
                        sil = silhouette_score(X_scaled[valid_mask], cluster_labels[valid_mask])
                    else:
                        rng = np.random.default_rng(42)
                        sample_idx = rng.choice(valid_mask.sum(), size=5000, replace=False)
                        valid_indices = np.where(valid_mask)[0]
                        sil = silhouette_score(
                            X_scaled[valid_indices[sample_idx]],
                            cluster_labels[valid_indices[sample_idx]]
                        )
                else:
                    sil = None
            else:
                sil = None

            # 保存会话
            st.session_state.cluster_model = model
            st.session_state.cluster_scaler = scaler
            st.session_state.cluster_features = feature_cols
            st.session_state.cluster_n_clusters = n_found
            st.session_state.cluster_silhouette = sil
            st.session_state.cluster_inertia = inertia

            cluster_df = numeric_df.copy()
            cluster_df["聚类标签"] = cluster_labels
            st.session_state.cluster_data = cluster_df

            # 保存模型文件
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(model, f)
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(scaler, f)
            config_dict = {
                "features": [str(col) for col in feature_cols],
                "algorithm": st.session_state.cluster_algorithm,
                "params": {"n_clusters": n_clusters} if is_kmeans else {"eps": eps, "min_samples": min_samples},
                "n_clusters_found": n_found,
            }
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False)

            if is_kmeans:
                msg = f"✅ 聚类完成！分为 {n_clusters} 个簇 | 轮廓系数: {sil:.4f} | Inertia: {inertia:,.2f}" if sil else f"✅ 聚类完成！分为 {n_clusters} 个簇"
            else:
                noise_count = int((cluster_labels == -1).sum())
                msg = f"✅ 聚类完成！发现 {n_found} 个簇 + {noise_count} 个噪声点"
                if sil:
                    msg += f" | 轮廓系数: {sil:.4f}"
            st.success(msg)

# 清除模型
with clear_col:
    if st.button("清除已保存聚类模型", use_container_width=True):
        for p in [MODEL_PATH, SCALER_PATH, CONFIG_PATH]:
            if os.path.exists(p):
                os.remove(p)
        keys = ["cluster_model", "cluster_scaler", "cluster_features", "cluster_n_clusters",
                "cluster_silhouette", "cluster_inertia", "cluster_data", "cluster_algorithm",
                "cluster_params"]
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]
        st.warning("已清除聚类模型！")

# 聚类结果展示
st.subheader("📈 聚类结果展示")
if "cluster_data" in st.session_state:
    data = st.session_state.cluster_data
    n_found = st.session_state.cluster_n_clusters
    saved_algo = st.session_state.get("cluster_algorithm", "kmeans")
    is_kmeans_result = (saved_algo == "kmeans")

    # 数据预览
    st.markdown("#### 聚类结果数据")
    st.dataframe(data.head(15), use_container_width=True)

    # 簇样本数量统计
    st.markdown("#### 簇样本数量统计")
    counts = data["聚类标签"].value_counts().sort_index()
    count_df = counts.rename("样本数量").to_frame()
    count_df.index.name = "簇标签"
    st.dataframe(count_df, use_container_width=True)

    if not is_kmeans_result and -1 in counts.index:
        st.caption(f"🔸 标签 **-1** 为噪声点（{counts.get(-1, 0)} 个），不属于任何簇。")

    # 质量指标
    st.markdown("#### 聚类质量评估")
    sil = st.session_state.get("cluster_silhouette", None)
    inertia = st.session_state.get("cluster_inertia", None)

    q_col1, q_col2 = st.columns(2)
    with q_col1:
        if sil is not None:
            st.metric("轮廓系数 (Silhouette Score)", f"{sil:.4f}",
                      help="取值范围 [-1, 1]，越接近 1 表示簇内紧密、簇间分离好。> 0.5 为良好。")
        else:
            st.metric("轮廓系数", "N/A", help="噪声点过多或簇数不足，无法计算。")
    with q_col2:
        if inertia is not None:
            st.metric("惯性值 (Inertia)", f"{inertia:,.2f}",
                      help="簇内平方和，值越小越紧密。仅 K-means 有此指标。")
        else:
            st.metric("簇数量 (自动发现)", f"{n_found}",
                      help="DBSCAN 自动发现的簇数量（不含噪声点）。")

    if sil is not None and sil < 0.25:
        st.warning(f"⚠️ 轮廓系数较低（{sil:.3f}），聚类结构可能不明显。"
                   f"{'建议使用肘部法则重新选择 K 值。' if is_kmeans_result else '建议调整 eps 和 min_samples 参数。'}")

    st.divider()

    # 聚类可视化
    st.markdown("#### 聚类分布可视化")
    X = data[feature_cols].values
    X_scaled = st.session_state.cluster_scaler.transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    ev1, ev2 = pca.explained_variance_ratio_

    fig_cluster = go.Figure()
    labels = data["聚类标签"].values
    unique_labels = sorted(set(labels))

    for lbl in unique_labels:
        mask = labels == lbl
        if lbl == -1:
            name, color, opacity, size = "噪声", "#888888", 0.3, 5
        else:
            name, color, opacity, size = f"簇 {lbl}", None, 0.7, 7

        trace_kwargs = dict(
            x=X_pca[mask, 0], y=X_pca[mask, 1],
            mode='markers', name=name,
            marker=dict(size=size, opacity=opacity),
            hovertemplate=f'PC1: %{{x:.3f}}<br>PC2: %{{y:.3f}}<extra></extra>'
        )
        if color:
            trace_kwargs["marker"]["color"] = color
        fig_cluster.add_trace(go.Scatter(**trace_kwargs))

    algo_title = "K-means" if is_kmeans_result else "DBSCAN"
    fig_cluster.update_layout(
        title=f"{algo_title} 聚类可视化 (K={n_found})" if is_kmeans_result else f"DBSCAN 聚类可视化 ({n_found} 簇)",
        xaxis_title=f"主成分 1 ({ev1:.1%} 方差)",
        yaxis_title=f"主成分 2 ({ev2:.1%} 方差)",
        template="plotly_white", height=420,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

# 预测
st.subheader("🎯 聚类预测")
predict_algo = st.session_state.get("cluster_algorithm", "kmeans")

if "cluster_model" not in st.session_state:
    st.warning("请先训练模型！")
elif predict_algo != "kmeans":
    st.info("💡 DBSCAN 是一种基于密度的聚类算法，不提供 `predict()` 方法。如需对新数据进行聚类，请重新训练模型并包含新数据，或切换到 K-means 算法。")
else:
    model = st.session_state.cluster_model
    scaler = st.session_state.cluster_scaler
    features = st.session_state.cluster_features
    k = st.session_state.cluster_n_clusters

    st.info(f"✅ K-means 模型已加载 | 聚类数 K={k}")
    st.write("请输入特征值，预测所属簇：")

    input_data = []
    cols = st.columns(len(features))
    for i, col in enumerate(cols):
        val = col.number_input(f"特征 {features[i]}", value=0.0, step=0.1)
        input_data.append(val)

    if st.button("执行聚类预测", use_container_width=True):
        input_arr = np.array([input_data])
        input_scaled = scaler.transform(input_arr)
        pred_cluster = model.predict(input_scaled)[0]
        st.success(f"🎯 预测结果：该数据属于 **簇 {pred_cluster}**")
