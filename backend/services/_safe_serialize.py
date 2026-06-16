"""Small serialization helpers for model artifacts."""
import pickle


def save_scaler(scaler, path):
    with open(path, "wb") as fh:
        pickle.dump(scaler, fh)


def load_scaler(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_kmeans(model, path):
    with open(path, "wb") as fh:
        pickle.dump(model, fh)


def load_kmeans(path):
    from sklearn.cluster import KMeans
    return safe_load_pickle(path, KMeans)


def safe_load_pickle(path, expected_type):
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if not isinstance(obj, expected_type):
        raise TypeError(f"Unexpected artifact type: {type(obj).__name__}")
    return obj
