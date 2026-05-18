"""Clustering (K-means & DBSCAN) training & prediction."""
import os
import pickle
import json
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "cluster_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "cluster_scaler.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "cluster_config.json")


def train(df, feature_cols, algorithm, params):
    """Train clustering model. Returns labels, metrics, and PCA coords for plotting."""
    X = df[feature_cols].values
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
    if valid_mask.sum() >= 2:
        if valid_mask.sum() <= 5000:
            sil = float(silhouette_score(X_scaled[valid_mask], cluster_labels[valid_mask]))
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

    # Persist
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    config_dict = {
        "features": [str(c) for c in feature_cols],
        "algorithm": algorithm,
        "params": params,
        "n_clusters_found": n_found,
    }
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False)

    return {
        "labels": labels, "cluster_counts": cluster_counts, "n_found": n_found,
        "silhouette": sil, "inertia": inertia, "pca": pca_result,
        "algorithm": algorithm
    }, None


def elbow(df, feature_cols, max_k):
    """Compute inertia for K values 1..max_k."""
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ks = list(range(1, max_k + 1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))
    return {"ks": ks, "inertias": inertias}


def predict_one(feature_values):
    """Predict cluster for a new data point (K-means only)."""
    if not os.path.exists(MODEL_PATH):
        return None, "No saved model found."
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    if config.get("algorithm") != "kmeans":
        return None, "Only K-means supports prediction."
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    input_arr = np.array([feature_values])
    input_scaled = scaler.transform(input_arr)
    pred = int(model.predict(input_scaled)[0])
    return {"cluster": pred}, None
