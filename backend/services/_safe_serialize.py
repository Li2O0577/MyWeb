"""Safe serialization for sklearn objects — replacing pickle with numpy .npz.

StandardScaler, KMeans → numpy .npz format (no code execution risk)
Pipeline, DBSCAN    → restricted unpickler (sklearn/numpy modules only)
"""
import os
import io
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ═══════════════════════════════════════════════════════════════════════════════
# StandardScaler — save/load via np.savez (no pickle)
# ═══════════════════════════════════════════════════════════════════════════════

def save_scaler(scaler, path):
    """Persist StandardScaler parameters as .npz."""
    np.savez(
        path,
        mean=scaler.mean_,
        scale=scaler.scale_,
        var=scaler.var_,
        n_features_in=scaler.n_features_in_,
        n_samples_seen=scaler.n_samples_seen_,
    )


def load_scaler(path):
    """Reconstruct a fitted StandardScaler from .npz (new) or .pkl (legacy).

    New models use np.savez (no code execution).  Legacy models use pickle
    loaded via RestrictedUnpickler so pre-existing model versions continue
    to work without retraining.
    """
    if path.endswith(".npz"):
        data = np.load(path)
        scaler = StandardScaler()
        scaler.mean_ = data["mean"]
        scaler.scale_ = data["scale"]
        scaler.var_ = data["var"]
        scaler.n_features_in_ = int(data["n_features_in"])
        scaler.n_samples_seen_ = int(data["n_samples_seen"])
        return scaler
    # Legacy pickle-format scaler (.pkl) — loaded via restricted unpickler
    return safe_load_pickle(path, StandardScaler)


# ═══════════════════════════════════════════════════════════════════════════════
# KMeans — save/load via np.savez (no pickle)
# ═══════════════════════════════════════════════════════════════════════════════

def save_kmeans(model, path):
    """Persist fitted KMeans parameters as .npz."""
    np.savez(
        path,
        cluster_centers=model.cluster_centers_,
        n_clusters=model.n_clusters,
        n_features_in=model.n_features_in_,
        n_iter=model.n_iter_,
    )


def load_kmeans(path):
    """Reconstruct a fitted KMeans from .npz parameters."""
    data = np.load(path)
    n_clusters = int(data["n_clusters"])
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    kmeans.cluster_centers_ = data["cluster_centers"]
    kmeans.n_features_in_ = int(data["n_features_in"])
    kmeans.n_iter_ = int(data["n_iter"])
    kmeans._n_threads = os.cpu_count() or 1
    return kmeans


# ═══════════════════════════════════════════════════════════════════════════════
# Restricted pickle — for complex objects (Pipeline, DBSCAN)
# ═══════════════════════════════════════════════════════════════════════════════

class RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows sklearn, numpy, pandas and builtin types."""

    _SAFE_PREFIXES = (
        "sklearn",
        "numpy",
        "pandas",
        "scipy",
        "pyarrow",
    )

    def find_class(self, module, name):
        if module == "builtins":
            return super().find_class(module, name)
        for prefix in self._SAFE_PREFIXES:
            if module == prefix or module.startswith(prefix + "."):
                return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Unpickling of {module}.{name} blocked — not in safe-module allowlist"
        )


def safe_load_pickle(path, expected_type):
    """Load a pickle file via RestrictedUnpickler + post-load type check."""
    with open(path, "rb") as f:
        obj = RestrictedUnpickler(f).load()
    if not isinstance(obj, expected_type):
        raise TypeError(
            f"Expected {expected_type.__name__}, got {type(obj).__name__}"
        )
    return obj
