"""Structured chart data generation for the React visualization workspace."""
import math
import pandas as pd
import numpy as np


LARGE_POINT_ROWS = 30000
VERY_LARGE_POINT_ROWS = 100000
HIGH_CARDINALITY = 50
VERY_HIGH_CARDINALITY = 200
PIE_MAX_USEFUL_CATEGORIES = 20
GROUP_MAX_USEFUL_CATEGORIES = 30
MAX_POINTS = 1200


AGG_MAP = {
    "mean": "mean",
    "sum": "sum",
    "count": "count",
    "median": "median",
    "min": "min",
    "max": "max",
}


def _json_value(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _resolve_column(df, name, required=True):
    if name in (None, "", "无", "none", "__index__"):
        if required:
            raise ValueError("缺少必要字段。")
        return None
    for col in df.columns:
        if str(col) == str(name):
            return col
    if required:
        raise ValueError(f"字段不存在：{name}")
    return None


def _resolve_columns(df, names):
    return [_resolve_column(df, name) for name in (names or []) if name not in (None, "", "无", "none")]


def _numeric_columns(df):
    return list(df.select_dtypes(include=[np.number]).columns)


def _column_cardinality(df, col):
    col = _resolve_column(df, col, required=False)
    if col is None:
        return 0
    return int(df[col].nunique(dropna=True))


def _is_id_like(df, col):
    col = _resolve_column(df, col, required=False)
    if col is None or len(df) == 0:
        return False
    unique = _column_cardinality(df, col)
    return unique > HIGH_CARDINALITY and unique / max(len(df), 1) > 0.85


def _warning(level, title, detail):
    return {"level": level, "title": title, "detail": detail}


def _common_large_point_warnings(df, chart_name, color_col=None, size_col=None):
    warnings = []
    n_rows = len(df)
    if n_rows > VERY_LARGE_POINT_ROWS:
        warnings.append(_warning(
            "error",
            f"{chart_name}数据量过大",
            f"当前有 {n_rows:,} 行，接口会抽样返回，建议先筛选、聚合或换成分箱类图表。",
        ))
    elif n_rows > LARGE_POINT_ROWS:
        warnings.append(_warning(
            "warning",
            f"{chart_name}点数较多",
            f"当前有 {n_rows:,} 行，已限制返回点数以避免浏览器卡顿。",
        ))

    color_unique = _column_cardinality(df, color_col)
    if color_unique > VERY_HIGH_CARDINALITY:
        warnings.append(_warning(
            "error",
            "颜色分组类别过多",
            f"「{color_col}」有 {color_unique:,} 个不同取值，图例会失控且颜色不可读。",
        ))
    elif color_unique > GROUP_MAX_USEFUL_CATEGORIES:
        warnings.append(_warning(
            "warning",
            "颜色分组较多",
            f"「{color_col}」有 {color_unique:,} 个类别，颜色区分会变弱。",
        ))

    if size_col and _is_id_like(df, size_col):
        warnings.append(_warning(
            "warning",
            "气泡大小字段疑似 ID",
            f"「{size_col}」几乎每行都不同，用作气泡大小通常没有解释意义。",
        ))
    return warnings


def _category_warnings(df, col, chart_name, max_useful=GROUP_MAX_USEFUL_CATEGORIES):
    warnings = []
    unique = _column_cardinality(df, col)
    if unique > VERY_HIGH_CARDINALITY:
        warnings.append(_warning(
            "error",
            f"{chart_name}类别数过多",
            f"「{col}」有 {unique:,} 个不同取值，直接绘制会不可读，建议先聚合 Top N 或筛选。",
        ))
    elif unique > max_useful:
        warnings.append(_warning(
            "warning",
            f"{chart_name}类别较多",
            f"「{col}」有 {unique:,} 个类别，标签和图例可能拥挤。",
        ))
    if _is_id_like(df, col):
        warnings.append(_warning(
            "warning",
            "字段疑似唯一标识",
            f"「{col}」接近一行一个取值，通常不适合作为类别轴或分组字段。",
        ))
    return warnings


def _sample(df, max_points=MAX_POINTS):
    if len(df) <= max_points:
        return df.copy()
    return df.sample(n=max_points, random_state=7).sort_index()


def _xy_records(df, x_col, y_col, color_col=None, size_col=None, z_col=None):
    cols = [col for col in [x_col, y_col, color_col, size_col, z_col] if col is not None]
    sample = _sample(df[cols].dropna(subset=[x_col, y_col]))
    records = []
    for _, row in sample.iterrows():
        item = {"x": _json_value(row[x_col]), "y": _json_value(row[y_col])}
        if z_col is not None:
            item["z"] = _json_value(row[z_col])
        if color_col is not None:
            item["color"] = str(_json_value(row[color_col]))
        if size_col is not None:
            item["size"] = _json_value(row[size_col])
        records.append(item)
    return records, len(sample)


def _line_series(df, x_col, y_cols, group_col=None, agg="none"):
    plot_df = df.copy()
    x_name = "__index__" if x_col is None else str(x_col)
    if x_col is None:
        plot_df["__index__"] = np.arange(len(plot_df))
        x_col = "__index__"

    if agg in AGG_MAP and x_col != "__index__":
        group_keys = [x_col]
        if group_col is not None:
            group_keys.append(group_col)
        plot_df = plot_df.groupby(group_keys, as_index=False)[y_cols].agg(AGG_MAP[agg])
    else:
        plot_df = _sample(plot_df, max_points=MAX_POINTS)

    series = []
    for y_col in y_cols:
        if group_col is None:
            points = [
                {"x": _json_value(row[x_col]), "y": _json_value(row[y_col])}
                for _, row in plot_df[[x_col, y_col]].dropna().iterrows()
            ]
            series.append({"name": str(y_col), "points": points})
        else:
            for group_value, group_df in plot_df.groupby(group_col, dropna=False):
                label = f"{y_col} · {group_value}"
                points = [
                    {"x": _json_value(row[x_col]), "y": _json_value(row[y_col])}
                    for _, row in group_df[[x_col, y_col]].dropna().iterrows()
                ]
                series.append({"name": str(label), "points": points})
    return {"x_label": x_name, "series": series}


def _bar_data(df, x_col, y_col, color_col=None, agg="mean", top_n=30):
    agg = agg if agg in AGG_MAP else "mean"
    group_keys = [x_col]
    if color_col is not None and color_col != x_col:
        group_keys.append(color_col)
    grouped = df.groupby(group_keys, dropna=False, as_index=False)[y_col].agg(AGG_MAP[agg])
    grouped = grouped.sort_values(y_col, ascending=False).head(int(top_n))
    bars = []
    for _, row in grouped.iterrows():
        bars.append({
            "label": str(_json_value(row[x_col])),
            "value": _json_value(row[y_col]),
            "group": str(_json_value(row[color_col])) if color_col is not None and color_col in row else None,
        })
    return {"bars": bars, "x_label": str(x_col), "y_label": str(y_col)}


def _histogram_data(df, col, bins=40, group_col=None, cumulative=False, norm="count"):
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    bins = max(5, min(int(bins), 200))
    if values.empty:
        return {"bins": []}
    counts, edges = np.histogram(values, bins=bins)
    if cumulative:
        counts = np.cumsum(counts)
    if norm == "percent":
        counts = counts / max(counts.sum(), 1) * 100
    elif norm == "probability":
        counts = counts / max(counts.sum(), 1)
    elif norm == "density":
        counts, edges = np.histogram(values, bins=bins, density=True)
        if cumulative:
            counts = np.cumsum(counts)
    result = []
    for i, count in enumerate(counts):
        start = float(edges[i])
        end = float(edges[i + 1])
        result.append({
            "label": f"{start:.2f}-{end:.2f}",
            "start": start,
            "end": end,
            "count": float(count),
        })
    return {"bins": result, "x_label": str(col), "group_col": str(group_col) if group_col else None}


def _box_groups(df, y_col, x_col=None, max_groups=30):
    groups = [(None, df)] if x_col is None else list(df.groupby(x_col, dropna=False))[:max_groups]
    output = []
    for label, group_df in groups:
        values = pd.to_numeric(group_df[y_col], errors="coerce").dropna()
        if values.empty:
            continue
        q1 = float(values.quantile(0.25))
        median = float(values.quantile(0.5))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower = max(float(values.min()), q1 - 1.5 * iqr)
        upper = min(float(values.max()), q3 + 1.5 * iqr)
        outliers = values[(values < lower) | (values > upper)].head(60).tolist()
        hist_counts, hist_edges = np.histogram(values, bins=min(20, max(5, int(np.sqrt(len(values))))))
        output.append({
            "label": str(_json_value(label)) if label is not None else str(y_col),
            "q1": q1,
            "median": median,
            "q3": q3,
            "lower": lower,
            "upper": upper,
            "outliers": [_json_value(v) for v in outliers],
            "density": [
                {"start": float(hist_edges[i]), "end": float(hist_edges[i + 1]), "count": int(hist_counts[i])}
                for i in range(len(hist_counts))
            ],
        })
    return {"groups": output, "y_label": str(y_col)}


def _density_heatmap(df, x_col, y_col, bins=24):
    plot_df = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if plot_df.empty:
        return {"x_bins": [], "y_bins": [], "cells": []}
    heat, x_edges, y_edges = np.histogram2d(plot_df[x_col], plot_df[y_col], bins=max(8, min(int(bins), 60)))
    cells = []
    for xi in range(heat.shape[0]):
        for yi in range(heat.shape[1]):
            if heat[xi, yi] > 0:
                cells.append({"x": xi, "y": yi, "value": int(heat[xi, yi])})
    return {
        "x_label": str(x_col),
        "y_label": str(y_col),
        "x_bins": [float(v) for v in x_edges],
        "y_bins": [float(v) for v in y_edges],
        "cells": cells,
    }


def _pie_data(df, names_col, values_col=None, top_n=20):
    if values_col is None:
        grouped = df[names_col].value_counts(dropna=False).reset_index()
        grouped.columns = [names_col, "value"]
    else:
        grouped = df.groupby(names_col, dropna=False, as_index=False)[values_col].sum()
        grouped = grouped.rename(columns={values_col: "value"})
    grouped = grouped.sort_values("value", ascending=False)
    top_n = max(3, min(int(top_n), 50))
    top = grouped.head(top_n).copy()
    if len(grouped) > top_n:
        other = pd.DataFrame([{names_col: "其他", "value": grouped.iloc[top_n:]["value"].sum()}])
        top = pd.concat([top, other], ignore_index=True)
    return {
        "slices": [{"label": str(_json_value(row[names_col])), "value": _json_value(row["value"])} for _, row in top.iterrows()]
    }


def _scatter_matrix(df, cols, color_col=None):
    cols = cols[:6]
    sample = _sample(df[[*cols, *([color_col] if color_col else [])]].dropna(subset=cols), max_points=350)
    pairs = []
    for x_col in cols:
        for y_col in cols:
            points = []
            for _, row in sample.iterrows():
                item = {"x": _json_value(row[x_col]), "y": _json_value(row[y_col])}
                if color_col:
                    item["color"] = str(_json_value(row[color_col]))
                points.append(item)
            pairs.append({"x_col": str(x_col), "y_col": str(y_col), "points": points})
    return {"columns": [str(c) for c in cols], "pairs": pairs}


def _corr_heatmap(df, cols=None):
    cols = cols or _numeric_columns(df)
    cols = cols[:50]
    corr = df[cols].corr(numeric_only=True)
    cells = []
    for y_col in corr.index:
        for x_col in corr.columns:
            cells.append({"x": str(x_col), "y": str(y_col), "value": _json_value(corr.loc[y_col, x_col])})
    return {"columns": [str(c) for c in corr.columns], "cells": cells}


def build_visualization_payload(df, config):
    """Return structured chart data for React rendering."""
    chart_type = str(config.get("chart_type") or "scatter")
    numeric_cols = _numeric_columns(df)
    all_cols = list(df.columns)
    warnings = []
    title = str(config.get("title") or "")

    if chart_type == "scatter":
        x_col = _resolve_column(df, config.get("x_col") or (numeric_cols[0] if numeric_cols else all_cols[0]))
        y_col = _resolve_column(df, config.get("y_col") or (numeric_cols[1] if len(numeric_cols) > 1 else x_col))
        color_col = _resolve_column(df, config.get("color_col"), required=False)
        size_col = _resolve_column(df, config.get("size_col"), required=False)
        warnings.extend(_common_large_point_warnings(df, "散点图", color_col, size_col))
        if x_col == y_col:
            warnings.append(_warning("warning", "X/Y 轴相同", f"「{x_col}」同时用于 X 和 Y，散点会退化为对角线。"))
        points, sampled = _xy_records(df, x_col, y_col, color_col, size_col)
        data = {"points": points, "x_label": str(x_col), "y_label": str(y_col), "sampled_rows": sampled}

    elif chart_type in {"line", "area"}:
        x_col = _resolve_column(df, config.get("x_col"), required=False)
        y_cols = _resolve_columns(df, config.get("y_cols") or numeric_cols[:2])
        group_col = _resolve_column(df, config.get("group_col"), required=False)
        agg = str(config.get("agg") or "none")
        if not y_cols:
            raise ValueError("折线图/面积图至少需要一个数值 Y 轴字段。")
        warnings.extend(_category_warnings(df, group_col, "分组字段"))
        if agg == "none" and len(df) > LARGE_POINT_ROWS:
            warnings.append(_warning("warning", "未聚合且行数较多", "已抽样返回点位，建议按时间或类别聚合。"))
        data = _line_series(df, x_col, y_cols, group_col, agg)

    elif chart_type == "bar":
        x_col = _resolve_column(df, config.get("x_col") or all_cols[0])
        y_col = _resolve_column(df, config.get("y_col") or (numeric_cols[0] if numeric_cols else all_cols[0]))
        color_col = _resolve_column(df, config.get("color_col"), required=False)
        agg = str(config.get("agg") or "mean")
        warnings.extend(_category_warnings(df, x_col, "柱状图"))
        warnings.extend(_category_warnings(df, color_col, "颜色分组"))
        data = _bar_data(df, x_col, y_col, color_col, agg, config.get("top_n", 30))

    elif chart_type == "histogram":
        col = _resolve_column(df, config.get("col") or (numeric_cols[0] if numeric_cols else all_cols[0]))
        group_col = _resolve_column(df, config.get("group_col"), required=False)
        bins = int(config.get("bins", 40))
        if bins > 120:
            warnings.append(_warning("warning", "直方图柱数过多", "柱数过多会放大噪声，建议先使用 20-80 个柱。"))
        warnings.extend(_category_warnings(df, group_col, "直方图分组"))
        data = _histogram_data(df, col, bins, group_col, bool(config.get("cumulative", False)), str(config.get("norm") or "count"))

    elif chart_type in {"box", "violin"}:
        y_col = _resolve_column(df, config.get("y_col") or (numeric_cols[0] if numeric_cols else all_cols[0]))
        x_col = _resolve_column(df, config.get("x_col"), required=False)
        warnings.extend(_category_warnings(df, x_col, "分组字段"))
        data = _box_groups(df, y_col, x_col)

    elif chart_type == "density_heatmap":
        x_col = _resolve_column(df, config.get("x_col") or numeric_cols[0])
        y_col = _resolve_column(df, config.get("y_col") or numeric_cols[min(1, len(numeric_cols) - 1)])
        if x_col == y_col:
            raise ValueError("二维密度热力图需要不同的 X/Y 字段。")
        if len(df) > VERY_LARGE_POINT_ROWS:
            warnings.append(_warning("warning", "二维密度数据量较大", "密度分箱可能较慢，建议先筛选范围。"))
        data = _density_heatmap(df, x_col, y_col, config.get("bins", 24))

    elif chart_type == "pie":
        names_col = _resolve_column(df, config.get("names_col") or all_cols[0])
        values_col = _resolve_column(df, config.get("values_col"), required=False)
        unique_names = _column_cardinality(df, names_col)
        if unique_names > PIE_MAX_USEFUL_CATEGORIES:
            warnings.append(_warning("warning", "饼图类别较多", f"「{names_col}」超过常规可读范围，接口会合并长尾为“其他”。"))
        data = _pie_data(df, names_col, values_col, config.get("top_n", PIE_MAX_USEFUL_CATEGORIES))

    elif chart_type == "scatter_matrix":
        cols = _resolve_columns(df, config.get("cols") or numeric_cols[:4])
        color_col = _resolve_column(df, config.get("color_col"), required=False)
        if len(cols) < 2:
            raise ValueError("成对关系图至少需要 2 个数值列。")
        if len(cols) > 6:
            warnings.append(_warning("warning", "成对关系图列数较多", "最多返回前 6 个字段以保持可读性。"))
        warnings.extend(_category_warnings(df, color_col, "颜色分组"))
        data = _scatter_matrix(df, cols, color_col)

    elif chart_type == "corr_heatmap":
        cols = _resolve_columns(df, config.get("cols") or numeric_cols)
        if len(cols) < 2:
            raise ValueError("相关性热力图至少需要 2 个数值列。")
        if len(cols) > 30:
            warnings.append(_warning("warning", "相关性矩阵列数较多", "标签会拥挤，建议选择关键字段。"))
        data = _corr_heatmap(df, cols)

    elif chart_type == "scatter3d":
        if len(numeric_cols) < 3:
            raise ValueError("3D 散点图至少需要 3 个数值列。")
        x_col = _resolve_column(df, config.get("x_col") or numeric_cols[0])
        y_col = _resolve_column(df, config.get("y_col") or numeric_cols[1])
        z_col = _resolve_column(df, config.get("z_col") or numeric_cols[2])
        color_col = _resolve_column(df, config.get("color_col"), required=False)
        size_col = _resolve_column(df, config.get("size_col"), required=False)
        if len({x_col, y_col, z_col}) < 3:
            raise ValueError("3D 散点图的 X/Y/Z 轴不能重复。")
        warnings.extend(_common_large_point_warnings(df, "3D 散点图", color_col, size_col))
        points, sampled = _xy_records(df, x_col, y_col, color_col, size_col, z_col)
        data = {"points": points, "x_label": str(x_col), "y_label": str(y_col), "z_label": str(z_col), "sampled_rows": sampled}

    else:
        raise ValueError(f"不支持的图表类型：{chart_type}")

    return {
        "chart_type": chart_type,
        "title": title,
        "n_rows": int(len(df)),
        "warnings": warnings,
        "data": data,
    }
