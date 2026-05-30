"""Clustering (K-means & DBSCAN) training & prediction."""
import os
import json
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from models.registry import (
    get_model_paths, create_version_dir, register_version, generate_version_id,
)
from services._safe_serialize import (
    save_scaler, load_scaler,
    save_kmeans, load_kmeans,
    safe_load_pickle,
)


def train(df, feature_cols, algorithm, params, dataset_name="", session_id=""):
    """Train clustering model. Returns labels, metrics, PCA coords, and version_id."""
    df = df[feature_cols].dropna()
    if len(df) < 10:
        return None, f"Insufficient clean data: {len(df)} rows after dropping NaN"
    X = df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if algorithm == "kmeans":
        model = KMeans(n_clusters=params["n_clusters"], random_state=42, n_init='auto')
        cluster_labels = model.fit_predict(X_scaled)
        inertia = float(model.inertia_)
        n_found = params["n_clusters"]
    else:
        model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"], n_jobs=-1)
        cluster_labels = model.fit_predict(X_scaled)
        inertia = None
        n_found = int(len(set(cluster_labels) - {-1}))

        if n_found == 0:
            return None, "DBSCAN 将所有点标记为噪声！请增大 eps 或减小 min_samples 后重试。"

    # Silhouette score
    sil = None
    valid_mask = cluster_labels != -1
    valid_labels = cluster_labels[valid_mask]
    if valid_mask.sum() >= 2 and len(np.unique(valid_labels)) >= 2:
        if valid_mask.sum() <= 5000:
            sil = float(silhouette_score(X_scaled[valid_mask], valid_labels))
        else:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(valid_mask.sum(), size=5000, replace=False)
            valid_indices = np.where(valid_mask)[0]
            sil = float(silhouette_score(
                X_scaled[valid_indices[sample_idx]],
                cluster_labels[valid_indices[sample_idx]]
            ))

    # PCA for visualization
    if X_scaled.shape[1] >= 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        ev1, ev2 = pca.explained_variance_ratio_
        pca_result = {"x": X_pca[:, 0].tolist(), "y": X_pca[:, 1].tolist(),
                      "ev1": float(ev1), "ev2": float(ev2)}
    else:
        pca_result = {"x": X_scaled[:, 0].tolist(),
                      "y": [0.0] * len(X_scaled),
                      "ev1": 1.0, "ev2": 0.0}

    labels = cluster_labels.tolist()
    cluster_counts = {int(l): int((np.array(labels) == l).sum()) for l in sorted(set(labels))}

    # Persist as version
    version_id = generate_version_id()
    vdir = create_version_dir("clustering", version_id)

    # KMeans → safe .npz, DBSCAN → restricted pickle
    if algorithm == "kmeans":
        model_name = "model.npz"
        model_path = os.path.join(vdir, model_name)
        save_kmeans(model, model_path)
    else:
        model_name = "model.pkl"
        model_path = os.path.join(vdir, model_name)
        with open(model_path, "wb") as f:
            import pickle
            pickle.dump(model, f)

    scaler_name = "scaler.npz"
    scaler_path = os.path.join(vdir, scaler_name)
    config_path = os.path.join(vdir, "config.json")

    save_scaler(scaler, scaler_path)

    config_dict = {
        "features": [str(c) for c in feature_cols],
        "algorithm": algorithm,
        "params": params,
        "n_clusters_found": n_found,
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False)

    register_version("clustering", version_id, {
        "dataset_name": dataset_name,
        "session_id": session_id,
        "features": [str(c) for c in feature_cols],
        "target": "",
        "metrics": {"silhouette": sil, "n_clusters": n_found},
        "params": {"algorithm": algorithm, "params": params},
    }, {"model": model_name, "scaler": scaler_name, "config": "config.json"})

    return {
        "labels": labels, "cluster_counts": cluster_counts, "n_found": n_found,
        "silhouette": sil, "inertia": inertia, "pca": pca_result,
        "algorithm": algorithm, "version_id": version_id
    }, None


def elbow(df, feature_cols, max_k):
    """Compute inertia for K values 1..max_k. Capped at n_samples-1."""
    df_clean = df[feature_cols].dropna()
    X = df_clean.values
    n = len(X)
    max_valid = min(max_k, n - 1)
    if max_valid < 2:
        return {"ks": [], "inertias": [], "warning": "数据量不足以计算肘部法则（至少需要 3 行有效数据）"}
    capped = max_valid < max_k
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ks = list(range(1, max_valid + 1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))
    result = {"ks": ks, "inertias": inertias}
    if capped:
        result["warning"] = f"max_k 已从 {max_k} 调整为 {max_valid}（不能超过样本数）"
    return result


def predict_one(feature_values, version_id=None):
    """Predict cluster for a new data point (K-means only)."""
    paths, meta = get_model_paths("clustering", version_id)
    if not paths:
        return None, "没有找到已保存的聚类模型，请先训练模型或切换到有效版本。"

    with open(paths["config"], 'r') as f:
        config = json.load(f)
    if config.get("algorithm") != "kmeans":
        return None, "Only K-means supports prediction."

    model_path = paths.get("model", "")
    if model_path.endswith(".npz"):
        model = load_kmeans(model_path)
    else:
        # Legacy DBSCAN model saved as pickle — load via restricted unpickler
        model = safe_load_pickle(model_path, KMeans)
    scaler = load_scaler(paths["scaler"])

    input_arr = np.array([feature_values])
    input_scaled = scaler.transform(input_arr)
    pred = int(model.predict(input_scaled)[0])
    return {"cluster": pred}, None
