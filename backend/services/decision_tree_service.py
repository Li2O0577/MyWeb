"""Decision tree training & prediction (classification + regression)."""
import os
import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error

from models.registry import (
    get_model_paths, create_version_dir, register_version, generate_version_id,
)
from services._safe_serialize import safe_load_pickle


def train(df, target_col, feature_cols, task_type, criterion, max_depth,
          dataset_name="", session_id=""):
    """Train decision tree. Returns metrics + version_id."""
    is_cls = (task_type == "classification")
    cols = feature_cols + [target_col]
    df = df[cols].dropna()
    if len(df) < 10:
        return None, f"Insufficient clean data: {len(df)} rows after dropping NaN"
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if is_cls else None
    )

    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

    num_cols = [c for c in numeric_cols if c in feature_cols]
    cat_cols = [c for c in categorical_cols if c in feature_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

    if is_cls:
        tree_model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
    else:
        tree_model = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth, random_state=42)

    pipeline = Pipeline([("preprocess", preprocessor), ("tree", tree_model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    result = {}

    if is_cls:
        result["acc"] = float(accuracy_score(y_test, y_pred))
        result["cm"] = confusion_matrix(y_test, y_pred).tolist()
        unique_labels = sorted(set(y_test) | set(y_pred))
        result["label_names"] = [str(l) for l in unique_labels]
    else:
        result["r2"] = float(r2_score(y_test, y_pred))
        result["mae"] = float(mean_absolute_error(y_test, y_pred))
        result["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # Persist as version
    version_id = generate_version_id()
    vdir = create_version_dir("decision_tree", version_id)

    model_path = os.path.join(vdir, "model.pkl")
    config_path = os.path.join(vdir, "config.json")

    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)

    config_dict = {
        "features": feature_cols,
        "target": target_col,
        "categorical_features": cat_cols,
        "criterion": criterion,
        "max_depth": max_depth,
        "task_type": task_type
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False)

    metrics = {}
    if is_cls:
        metrics["acc"] = result.get("acc")
    else:
        metrics["r2"] = result.get("r2")
        metrics["mae"] = result.get("mae")
        metrics["rmse"] = result.get("rmse")

    register_version("decision_tree", version_id, {
        "dataset_name": dataset_name,
        "session_id": session_id,
        "features": feature_cols,
        "target": target_col,
        "categorical_features": cat_cols,
        "metrics": metrics,
        "params": {"criterion": criterion, "max_depth": max_depth, "task_type": task_type},
    }, {"model": "model.pkl", "config": "config.json"})

    # Tree rules
    tree = None
    for _, step in pipeline.named_steps.items():
        if hasattr(step, 'tree_'):
            tree = step
            break

    if tree is not None:
        tree_ = tree.tree_
        try:
            expanded_names = list(preprocessor.get_feature_names_out())
        except Exception:
            expanded_names = [f"x{i}" for i in range(tree_.n_features_)]
        result["tree_rules"] = export_text(tree, feature_names=expanded_names)

        criterion_cn = {"entropy": "信息熵", "gini": "基尼系数",
                        "squared_error": "平方误差", "friedman_mse": "Friedman MSE",
                        "absolute_error": "绝对误差", "poisson": "Poisson偏差"}
        criterion_name = criterion_cn.get(criterion, criterion)

        nodes = []
        for i in range(tree_.node_count):
            nodes.append({
                "id": i,
                "feature": expanded_names[tree_.feature[i]] if tree_.feature[i] != -2 else "叶子节点",
                "threshold": round(float(tree_.threshold[i]), 4) if tree_.feature[i] != -2 else "-",
                "impurity": round(float(tree_.impurity[i]), 4),
                "samples": int(tree_.n_node_samples[i]),
                "type": "内部节点" if tree_.children_left[i] != -1 else "叶子节点"
            })
        result["tree_nodes"] = nodes
        result["criterion_name"] = criterion_name

    result["version_id"] = version_id
    return result, None


def _safe_load_pickle(path):
    """Load a pickle file with type validation for sklearn models."""
    from sklearn.pipeline import Pipeline
    return safe_load_pickle(path, Pipeline)


def predict_one(input_dict, task_type, version_id=None):
    """Single prediction. Returns (result, None) or (None, (code, message))."""
    import pandas as pd
    paths, meta = get_model_paths("decision_tree", version_id)
    if not paths:
        return None, ("MODEL_NOT_FOUND", "没有找到已保存的决策树模型，请先训练模型或切换到有效版本。")

    model_task = meta.get("params", {}).get("task_type") if meta else None
    if model_task and model_task != task_type:
        task_names = {"classification": "分类", "regression": "回归"}
        return None, ("TASK_MISMATCH",
            f"当前决策树版本是{task_names.get(model_task, model_task)}模型，"
            f"但本次请求按{task_names.get(task_type, task_type)}任务预测。"
            "请切换到匹配的模型版本，或重新训练当前任务。"
        )

    pipeline = _safe_load_pickle(paths["model"])
    input_df = pd.DataFrame(input_dict)
    pred = pipeline.predict(input_df)[0]
    if task_type == "classification":
        return {"pred_class": str(pred)}, None
    else:
        return {"pred_value": float(pred)}, None
