"""Clustering page — Flask backend for training, Streamlit for UI."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pages._prepare import render_sidebar, data_uploader
from pages._mlp_common import cluster_explanation, render_version_selector, validate_input_array, render_ml_status_bar, render_ml_workbench_overview, render_stable_prediction_panel, render_small_dataset_warning
from pages._api import train_clustering, elbow_clustering, predict_clustering, clear_clustering, ensure_session, clustering_status, backend_status_badge, render_backend_sync_panel, list_clustering_versions, activate_clustering_version, delete_clustering_version
from pages._ui_common import render_metric_row, render_page_header, render_section_header

st.set_page_config(page_title="聚类分析", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>[data-testid="stSidebarNav"] {display: none;}</style>""", unsafe_allow_html=True)
render_sidebar("pages/8_clustering.py")
render_page_header("聚类分析", "使用 K-means 或 DBSCAN 发现数据分组，并管理聚类模型版本。")

backend_status_badge()

with st.expander("功能介绍", expanded=False):
    st.markdown("""
    ### 无监督聚类模型 — K-means & DBSCAN
    1. **双算法支持**：K-means（球状簇、需预设 K）+ DBSCAN（任意形状、自动发现簇数、识别噪声）
    2. **肘部法则**（K-means）：自动遍历 K 值计算 Inertia
    3. **前后端分离**：Flask 后端训练，Streamlit 前端展示
    """)

df = data_uploader()
if df is None:
    st.warning("⚠️ 请先上传数据！")
    st.stop()

numeric_df = df.select_dtypes(include=[np.number]).dropna()
if len(numeric_df) < 10 or numeric_df.shape[1] < 1:
    st.error("❌ 数据无效！需要至少10行数值数据")
    st.stop()

backend_synced = render_backend_sync_panel(numeric_df, compact=True)

feature_cols = [col for col in numeric_df.columns]
n_features = len(feature_cols)
n_samples = len(numeric_df)
render_small_dataset_warning(n_samples)

if "cluster_algorithm" not in st.session_state:
    st.session_state.cluster_algorithm = "kmeans"

# Version selector
active_vid = render_version_selector(
    "聚类",
    list_clustering_versions,
    activate_clustering_version,
    delete_clustering_version,
    prediction_keys=["cluster_last_prediction"],
)

# Auto-detect saved model — reloads when active version changes
status = clustering_status()
render_ml_status_bar(numeric_df, backend_synced, status, "聚类")
current_vid = status.get("version_id", "") if status else ""
if st.session_state.get("cluster_version_id") != current_vid:
    if status and status.get("has_model"):
        st.session_state.cluster_features = status.get("features", [])
        algo = status.get("params", {}).get("algorithm", "kmeans")
        if algo == "dbscan":
            st.session_state.cluster_algorithm = "dbscan"
        st.session_state.cluster_version_id = current_vid
        ds = status.get("dataset_name", "")
        created = status.get("created_at", "")[:16].replace("T", " ")
        st.info(f"📦 已加载聚类模型版本（{ds} | {created} | {algo}）。重新训练以查看聚类结果。")

render_section_header("算法配置", "选择聚类算法、特征列和核心参数。")
alg_col, *param_cols = st.columns([1, 1, 1, 2])
with alg_col:
    algorithm = st.selectbox("聚类算法", ["K-means", "DBSCAN"],
                             index=0 if st.session_state.cluster_algorithm == "kmeans" else 1)
    st.session_state.cluster_algorithm = "kmeans" if algorithm == "K-means" else "dbscan"

is_kmeans = (st.session_state.cluster_algorithm == "kmeans")

if is_kmeans:
    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        max_clusters = min(15, n_samples)
        n_clusters = st.number_input("聚类数量 K", min_value=2, max_value=max_clusters, value=min(3, max_clusters), step=1)
    with col2:
        st.info(f"✅ 数据：{n_samples} 行 | {n_features} 个特征 | K={n_clusters}")

    st.divider()
    render_section_header("肘部法则", "计算不同 K 值的惯性，用于辅助选择聚类数量。")
    max_k_limit = min(15, max(3, n_samples // 3))
    elbow_col1, elbow_col2 = st.columns([1, 3])
    with elbow_col1:
        max_k_elbow = st.number_input("最大 K 值", min_value=3, max_value=max_k_limit, value=min(10, max_k_limit), step=1)
        run_elbow = st.button("运行肘部分析", use_container_width=True, type="secondary")

    if run_elbow:
        with st.spinner("正在计算..."):
            sid = ensure_session(numeric_df)
            if not sid:
                st.toast("无法连接到 Flask 后端 (http://localhost:5001)。请确保后端已启动。", icon="❌")
            else:
                result = elbow_clustering(sid, [str(c) for c in feature_cols], max_k_elbow)
                if result:
                    st.session_state.elbow_result = result

    if "elbow_result" in st.session_state:
        eb = st.session_state.elbow_result
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=eb["ks"], y=eb["inertias"], mode='lines+markers', marker=dict(size=8), line=dict(width=2)))
        fig_elbow.update_layout(title="肘部法则 — Inertia vs K", xaxis=dict(title="K", dtick=1), yaxis_title="Inertia", template="plotly_white", height=380, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_elbow, use_container_width=True)

else:
    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        eps = st.number_input("邻域半径 ε (eps)", min_value=0.05, max_value=5.0, value=0.5, step=0.05)
    with col2:
        min_samples = st.number_input("最小样本数 (min_samples)", min_value=2, max_value=50, value=5, step=1)
    st.info(f"✅ 数据：{n_samples} 行 | {n_features} 个特征 | ε={eps} | min_samples={min_samples}")
    st.caption("💡 DBSCAN 自动发现簇数量，无需预设 K。标签为 -1 的点为噪声。")

if is_kmeans:
    cluster_extra_items = [("算法", "K-means"), ("K", n_clusters)]
else:
    cluster_extra_items = [("算法", "DBSCAN"), ("eps/min_samples", f"{eps} / {min_samples}")]

render_ml_workbench_overview(
    "聚类",
    numeric_df,
    backend_synced=backend_synced,
    model_status=status,
    task_text="无监督聚类",
    target_text="无目标列",
    feature_count=n_features,
    sample_count=n_samples,
    extra_items=cluster_extra_items,
)

st.divider()

render_section_header("模型训练", "训练聚类模型或清除当前聚类结果。")
train_col, clear_col = st.columns(2)

with train_col:
    if st.button("开始聚类训练" if "cluster_result" not in st.session_state else "重新训练", type="primary", use_container_width=True):
        with st.spinner("聚类训练中（后端 Flask 计算）..."):
            sid = ensure_session(numeric_df)
            if not sid:
                st.toast("无法连接到 Flask 后端 (http://localhost:5001)。请确保后端已启动。", icon="❌")
                st.stop()
            algo = "kmeans" if is_kmeans else "dbscan"
            params = {"n_clusters": n_clusters} if is_kmeans else {"eps": eps, "min_samples": min_samples}
            resp = train_clustering(sid, [str(c) for c in feature_cols], algo, params)
            if resp:
                if "error" not in resp:
                    result = resp
                    st.session_state.cluster_result = result
                    st.session_state.cluster_features = feature_cols
                    st.session_state.cluster_version_id = result.get("version_id", "")

                    if is_kmeans:
                        msg = f"✅ 聚类完成！分为 {result['n_found']} 个簇"
                        if result.get("silhouette"):
                            msg += f" | 轮廓系数: {result['silhouette']:.4f}"
                            msg += f" | Inertia: {result['inertia']:,.2f}"
                        st.toast(msg, icon="✅")
                    else:
                        noise = sum(1 for l in result["labels"] if l == -1)
                        msg = f"✅ 聚类完成！发现 {result['n_found']} 个簇 + {noise} 个噪声点"
                        if result.get("silhouette"):
                            msg += f" | 轮廓系数: {result['silhouette']:.4f}"
                        st.toast(msg, icon="✅")
                else:
                    st.toast(resp["error"], icon="❌")

with clear_col:
    if st.button("清除已保存聚类模型", use_container_width=True):
        clear_clustering()
        for k in ["cluster_result", "cluster_features", "elbow_result", "cluster_version_id", "cluster_last_prediction"]:
            if k in st.session_state: del st.session_state[k]
        st.warning("已清除聚类模型！")

render_section_header("聚类结果", "查看聚类分布、轮廓系数和二维可视化。")
if st.session_state.get("cluster_result"):
    data = st.session_state.cluster_result
    feature_cols = st.session_state.cluster_features

    counts = {int(k): int(v) for k, v in data["cluster_counts"].items()}
    count_df = pd.DataFrame({"样本数量": counts}).rename_axis("簇标签")
    st.markdown("#### 簇样本数量统计")
    st.dataframe(count_df, use_container_width=True)

    result_metrics = []
    if data.get("silhouette"):
        result_metrics.append(("轮廓系数", f"{data['silhouette']:.4f}"))
    if data.get("inertia"):
        result_metrics.append(("惯性值", f"{data['inertia']:,.2f}"))
    render_metric_row(result_metrics)

    st.markdown("#### 聚类分布可视化")
    pca = data["pca"]
    fig_cluster = go.Figure()
    labels = data["labels"]
    unique_labels = sorted(set(labels))
    for lbl in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == lbl]
        xs = [pca["x"][i] for i in indices]
        ys = [pca["y"][i] for i in indices]
        if lbl == -1:
            name, color, size = "噪声", "#888888", 5
        else:
            name, color, size = f"簇 {lbl}", None, 7
        kwargs = dict(x=xs, y=ys, mode='markers', name=name, marker=dict(size=size, opacity=0.7))
        if color: kwargs["marker"]["color"] = color
        fig_cluster.add_trace(go.Scatter(**kwargs))
    fig_cluster.update_layout(title=f"{'K-means' if data.get('algorithm')=='kmeans' else 'DBSCAN'} 聚类可视化", xaxis_title=f"PC1 ({pca.get('ev1',0):.1%})", yaxis_title=f"PC2 ({pca.get('ev2',0):.1%})", template="plotly_white", height=420, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_cluster, use_container_width=True)

# Prediction (K-means only)
render_section_header("聚类预测", "使用当前 K-means 版本预测新样本所属簇。")
if not st.session_state.get("cluster_result"):
    st.warning("请先训练模型！")
elif data.get("algorithm") != "kmeans":
    st.info("💡 DBSCAN 不提供 predict() 方法。如需对新数据进行聚类，请切换到 K-means 算法。")
else:
    features = st.session_state.cluster_features
    st.info(f"✅ K-means 模型已训练")
    st.write("请输入特征值，预测所属簇：")
    input_data = []
    cols = st.columns(min(len(features), 5))
    for i, col in enumerate(cols):
        val = col.number_input(f"特征 {features[i]}", value=0.0, step=0.1)
        input_data.append(val)
    if len(features) > 5:
        for row_start in range(5, len(features), 5):
            cols = st.columns(5)
            for j, col in enumerate(cols):
                idx = row_start + j
                if idx < len(features):
                    val = col.number_input(f"特征 {features[idx]}", value=0.0, step=0.1, key=f"cluster_pred_{idx}")
                    input_data.append(val)
    if st.button("执行聚类预测", use_container_width=True):
        err = validate_input_array(np.array([input_data]), "聚类预测", expected_features=len(features))
        if err:
            st.toast(err, icon="❌")
        else:
            result = predict_clustering(input_data, version_id=st.session_state.get("cluster_version_id"))
            if result and "cluster" in result:
                cluster_id = result["cluster"]
                st.session_state.cluster_last_prediction = {
                    "main": f"簇 {cluster_id}",
                    "details": [
                        ("算法", "K-means"),
                        ("簇说明", cluster_explanation(cluster_id, st.session_state.get("cluster_result"))),
                        ("模型版本", (st.session_state.get("cluster_version_id") or "当前激活版本")[:24]),
                    ],
                }
                st.toast(f"🎯 预测结果：该数据属于 **簇 {result['cluster']}**", icon="✅")

    if st.session_state.get("cluster_last_prediction"):
        pred = st.session_state.cluster_last_prediction
        if isinstance(pred, dict):
            render_stable_prediction_panel(
                "预测结果",
                pred["main"],
                details=pred["details"],
                model_status=status,
                fallback_result=st.session_state.get("cluster_result"),
                description="聚类预测结果表示样本所属分组，簇编号不代表好坏或顺序。",
            )
        else:
            st.session_state.pop("cluster_last_prediction", None)
