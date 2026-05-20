"""Friendly pre-training validation for model endpoints."""
import numpy as np
import pandas as pd


MAX_MISSING_RATIO = 0.5
MAX_CLASS_COUNT = 50


def _col_name(col):
    return str(col)


def _message(errors):
    return "；".join(errors)


def _selected_frame(df, target_col, feature_cols):
    cols = list(feature_cols)
    if target_col is not None:
        cols.append(target_col)
    return df[cols]


def _basic_checks(df, target_col, feature_cols, *, require_numeric_features=True,
                  min_clean_rows=10, batch_size=None):
    errors = []

    if df is None or df.empty:
        return ["当前数据为空，请先上传或同步有效数据。"], None, 0

    if not feature_cols:
        errors.append("至少需要选择 1 个特征列。")

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        errors.append(f"特征列不存在：{', '.join(_col_name(c) for c in missing_features)}。")

    if target_col is not None and target_col not in df.columns:
        errors.append(f"目标列不存在：{_col_name(target_col)}。")

    if target_col is not None and target_col in feature_cols:
        errors.append("目标列不能同时作为特征列，请从特征中移除目标列。")

    if len(set(feature_cols)) != len(feature_cols):
        errors.append("特征列存在重复项，请重新选择特征列。")

    if errors:
        return errors, None, 0

    selected = _selected_frame(df, target_col, feature_cols)
    missing_ratios = selected.isna().mean()
    high_missing = [
        f"{_col_name(col)}({ratio:.0%})"
        for col, ratio in missing_ratios.items()
        if ratio >= MAX_MISSING_RATIO
    ]
    if high_missing:
        errors.append(
            "以下列空值比例过高，请先在“数据处理”页面填充或删除空值："
            + "、".join(high_missing)
            + "。"
        )

    clean = selected.dropna()
    clean_rows = len(clean)
    if clean_rows < min_clean_rows:
        errors.append(
            f"删除空值后仅剩 {clean_rows} 行，至少需要 {min_clean_rows} 行可用于训练。"
        )

    if require_numeric_features:
        non_numeric = [
            c for c in feature_cols
            if not pd.api.types.is_numeric_dtype(clean[c])
        ]
        if non_numeric:
            errors.append(
                "模型训练要求特征列全部为数值型，请先编码这些特征列："
                + "、".join(_col_name(c) for c in non_numeric)
                + "。"
            )

    if clean_rows > 0:
        numeric_part = clean[feature_cols].select_dtypes(include=[np.number])
        if len(numeric_part.columns) == len(feature_cols):
            values = numeric_part.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                errors.append("特征列中包含无穷大或非法数值，请先清洗数据。")

    if batch_size is not None:
        try:
            batch = int(batch_size)
        except (TypeError, ValueError):
            errors.append("批次大小必须是正整数。")
        else:
            if batch <= 0:
                errors.append("批次大小必须大于 0。")
            elif clean_rows > 0 and batch > clean_rows:
                errors.append(f"批次大小为 {batch}，但清洗后只有 {clean_rows} 行数据，请调小 batch size。")

    return errors, clean, clean_rows


def validate_regression_training(df, target_col, feature_cols, *, batch_size=None,
                                 require_numeric_features=True, min_clean_rows=10):
    errors, clean, clean_rows = _basic_checks(
        df, target_col, feature_cols,
        require_numeric_features=require_numeric_features,
        min_clean_rows=min_clean_rows,
        batch_size=batch_size,
    )
    if clean is not None and target_col in clean.columns:
        if not pd.api.types.is_numeric_dtype(clean[target_col]):
            errors.append(f"回归目标列「{_col_name(target_col)}」必须是数值型。")
        else:
            y = clean[target_col].to_numpy(dtype=float)
            if not np.isfinite(y).all():
                errors.append("目标列中包含无穷大或非法数值，请先清洗数据。")
            if pd.Series(y).nunique(dropna=True) < 2:
                errors.append("回归目标列至少需要 2 个不同数值。")

    if errors:
        return {"error": _message(errors)}
    return None


def validate_classification_training(df, target_col, feature_cols, *, batch_size=None,
                                     require_numeric_features=True, min_clean_rows=10):
    errors, clean, clean_rows = _basic_checks(
        df, target_col, feature_cols,
        require_numeric_features=require_numeric_features,
        min_clean_rows=min_clean_rows,
        batch_size=batch_size,
    )
    if clean is not None and target_col in clean.columns:
        counts = clean[target_col].value_counts(dropna=True)
        n_classes = len(counts)
        if n_classes < 2:
            errors.append("分类目标列至少需要 2 个类别。")
        if n_classes > MAX_CLASS_COUNT or (clean_rows > 0 and n_classes > clean_rows * 0.5):
            errors.append(
                f"目标列共有 {n_classes} 个类别，过于接近样本数，像是 ID 或连续值；"
                "请改选真正的分类标签，或先合并/分箱类别。"
            )
        if n_classes >= 2 and counts.min() < 2:
            rare = counts[counts < 2].index.tolist()[:5]
            errors.append(
                "每个类别至少需要 2 个样本，样本过少的类别包括："
                + "、".join(_col_name(v) for v in rare)
                + "。"
            )

    if errors:
        return {"error": _message(errors)}
    return None


def validate_mlp_training(df, target_col, feature_cols, *, task_type, batch_size,
                          val_split=None, n_classes=None):
    if task_type == "classification":
        error = validate_classification_training(
            df, target_col, feature_cols, batch_size=batch_size
        )
    else:
        error = validate_regression_training(
            df, target_col, feature_cols, batch_size=batch_size
        )
    if error:
        return error

    if val_split is not None:
        try:
            split = float(val_split)
        except (TypeError, ValueError):
            return {"error": "验证集比例必须是 0 到 1 之间的小数。"}
        if not 0 < split < 0.8:
            return {"error": "验证集比例需要大于 0 且小于 0.8。"}

    if task_type == "classification" and n_classes is not None:
        try:
            declared = int(n_classes)
        except (TypeError, ValueError):
            return {"error": "分类类别数必须是正整数。"}
        actual = df[list(feature_cols) + [target_col]].dropna()[target_col].nunique()
        if declared != actual:
            return {"error": f"前端传入的类别数为 {declared}，但目标列实际有 {actual} 个类别，请刷新后重试。"}

    return None


def validate_feature_training(df, feature_cols, *, min_clean_rows=10):
    errors, _clean, _clean_rows = _basic_checks(
        df, None, feature_cols,
        require_numeric_features=True,
        min_clean_rows=min_clean_rows,
        batch_size=None,
    )
    if errors:
        return {"error": _message(errors)}
    return None
