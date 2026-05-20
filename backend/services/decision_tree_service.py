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

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "dt_model.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "dt_config.json")


def train(df, target_col, feature_cols, task_type, criterion, max_depth):
    """Train decision tree. Returns metrics."""
    is_cls = (task_type == "classification")
    cols = feature_cols + [target_col]
    df = df[cols].dropna()
    if len(df) < 10:
        return {"error": f"Insufficient clean data: {len(df)} rows after dropping NaN"}
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

    # Persist
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            "features": feature_cols,
            "target": target_col,
            "criterion": criterion,
            "max_depth": max_depth,
            "task_type": task_type
        }, f, ensure_ascii=False)

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

    return result


def predict_one(input_dict, task_type):
    """Single prediction. Returns predicted value/class."""
    if not os.path.exists(MODEL_PATH):
        return None, "No saved model found."
    import pandas as pd
    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    input_df = pd.DataFrame(input_dict)
    pred = pipeline.predict(input_df)[0]
    if task_type == "classification":
        return {"pred_class": str(pred)}, None
    else:
        return {"pred_value": float(pred)}, None
