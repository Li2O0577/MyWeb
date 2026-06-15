"""Data parsing, processing, outlier detection, and summary generation."""
import ast
import io
import pandas as pd
import numpy as np


def parse_file(file_bytes, filename):
    """Parse uploaded CSV/Excel into a DataFrame."""
    if filename.endswith('.csv'):
        return pd.read_csv(io.BytesIO(file_bytes))
    elif filename.endswith(('.xlsx', '.xls')):
        return pd.read_excel(io.BytesIO(file_bytes))
    else:
        raise ValueError(f"不支持的文件格式: {filename}。请上传 CSV (.csv) 或 Excel (.xlsx) 文件。")


_ALLOWED_EXPR_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_ALLOWED_EXPR_UNARYOPS = (ast.UAdd, ast.USub)


def _resolve_column(df, name):
    """Resolve frontend string column names back to the DataFrame column object."""
    for col in df.columns:
        if str(col) == str(name):
            return col
    raise KeyError(f"字段不存在：{name}")


def _resolve_columns(df, names):
    return [_resolve_column(df, name) for name in (names or [])]


def _ensure_new_column_name(df, name):
    name = str(name or "").strip()
    if not name:
        raise ValueError("新列名不能为空。")
    if name in [str(c) for c in df.columns]:
        raise ValueError(f"列名已存在：{name}")
    return name


def _cast_series(series, dtype):
    dtype = str(dtype or "").lower()
    if dtype in {"int", "integer"}:
        return pd.to_numeric(series, errors="raise").round().astype("Int64")
    if dtype in {"float", "number", "numeric"}:
        return pd.to_numeric(series, errors="raise")
    if dtype in {"str", "string", "text"}:
        return series.astype("string")
    if dtype in {"datetime", "date", "time"}:
        return pd.to_datetime(series, errors="raise")
    if dtype in {"category", "categorical"}:
        return series.astype("category")
    if dtype in {"bool", "boolean"}:
        normalized = series.astype("string").str.strip().str.lower()
        mapping = {
            "true": True, "t": True, "yes": True, "y": True, "1": True,
            "false": False, "f": False, "no": False, "n": False, "0": False,
        }
        converted = normalized.map(mapping)
        if converted.isna().any() and series.notna().any():
            raise ValueError("布尔转换只接受 true/false、yes/no 或 1/0。")
        return converted.astype("boolean")
    raise ValueError(f"不支持的目标类型：{dtype}")


def _fill_series(series, method, value=None):
    method = str(method or "").lower()
    if method == "mean":
        return series.fillna(pd.to_numeric(series, errors="raise").mean())
    if method == "median":
        return series.fillna(pd.to_numeric(series, errors="raise").median())
    if method == "mode":
        mode = series.mode(dropna=True)
        return series.fillna(mode.iloc[0] if not mode.empty else value)
    if method in {"zero", "0"}:
        return series.fillna(0)
    if method in {"constant", "value"}:
        return series.fillna(value)
    if method == "ffill":
        return series.ffill()
    if method == "bfill":
        return series.bfill()
    raise ValueError(f"不支持的缺失值填充方法：{method}")


def _scale_columns(df, cols, method):
    method = str(method or "standard").lower()
    for col in cols:
        values = pd.to_numeric(df[col], errors="raise")
        if method in {"standard", "standardize", "zscore"}:
            std = values.std(ddof=0)
            df[col] = (values - values.mean()) / (std if std and not np.isnan(std) else 1)
        elif method in {"minmax", "normalize"}:
            span = values.max() - values.min()
            df[col] = (values - values.min()) / (span if span and not np.isnan(span) else 1)
        else:
            raise ValueError(f"不支持的缩放方法：{method}")
    return df


def _validate_formula_expr(expr, allowed_names):
    if not expr or len(expr) > 500:
        raise ValueError("表达式不能为空，且长度不能超过 500 个字符。")
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Expression):
            continue
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_EXPR_BINOPS):
                raise ValueError("表达式只支持基础算术运算。")
            continue
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_EXPR_UNARYOPS):
                raise ValueError("表达式只支持基础一元运算。")
            continue
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ValueError(f"表达式中包含不可用列名或变量：{node.id}")
            continue
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("表达式中只能使用数字常量。")
            continue
        if isinstance(node, (ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.UAdd, ast.USub)):
            continue
        raise ValueError("表达式不能调用函数、访问属性或使用非算术语法。")
    return tree


def _apply_formula(df, new_col, expr):
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    namespace = {}
    for col in numeric_cols:
        key = str(col)
        if key.isidentifier():
            namespace[key] = pd.to_numeric(df[col], errors="raise")
    tree = _validate_formula_expr(expr, set(namespace))
    if not namespace:
        raise ValueError("当前没有可用于表达式的数值列。")
    df[new_col] = eval(compile(tree, "<formula>", "eval"), {"__builtins__": {}}, namespace)
    return df


def _apply_unary_calc(df, op):
    col = _resolve_column(df, op.get("col"))
    new_col = _ensure_new_column_name(df, op.get("new_col"))
    operator = str(op.get("operator") or "").lower()
    values = pd.to_numeric(df[col], errors="raise")
    if operator in {"square", "平方"}:
        result = values ** 2
    elif operator in {"sqrt", "开方"}:
        result = np.sqrt(values)
    elif operator in {"log", "取对数"}:
        result = np.log(values)
    elif operator in {"abs", "绝对值"}:
        result = values.abs()
    elif operator in {"round", "四舍五入"}:
        result = values.round(int(op.get("decimals", 2)))
    elif operator in {"neg", "negative", "取负"}:
        result = -values
    else:
        raise ValueError(f"不支持的一元运算：{operator}")
    df[new_col] = result
    return df


def _apply_binary_calc(df, op):
    left_col = _resolve_column(df, op.get("left_col"))
    new_col = _ensure_new_column_name(df, op.get("new_col"))
    left = pd.to_numeric(df[left_col], errors="raise")
    if op.get("right_col"):
        right = pd.to_numeric(df[_resolve_column(df, op.get("right_col"))], errors="raise")
    else:
        right = float(op.get("right_value", 0))
    operator = str(op.get("operator") or "+")
    if operator == "+":
        result = left + right
    elif operator in {"-", "−"}:
        result = left - right
    elif operator in {"*", "×"}:
        result = left * right
    elif operator in {"/", "÷"}:
        result = left / right
    elif operator == "//":
        result = left // right
    elif operator == "%":
        result = left % right
    elif operator in {"**", "^"}:
        result = left ** right
    else:
        raise ValueError(f"不支持的二元运算：{operator}")
    df[new_col] = result
    return df


def _apply_pca(df, op):
    cols = _resolve_columns(df, op.get("cols"))
    if len(cols) < 2:
        raise ValueError("PCA 至少需要 2 个数值列。")
    n_components = int(op.get("n_components", 2))
    if n_components < 1 or n_components > len(cols):
        raise ValueError("PCA 维度必须介于 1 和所选列数量之间。")
    matrix = df[cols].apply(pd.to_numeric, errors="raise")
    matrix = matrix.fillna(matrix.median(numeric_only=True))
    values = matrix.to_numpy(dtype=float)
    std = values.std(axis=0)
    std[std == 0] = 1
    values = (values - values.mean(axis=0)) / std
    _, _, vt = np.linalg.svd(values, full_matrices=False)
    transformed = values @ vt[:n_components].T
    prefix = str(op.get("prefix") or "PCA").strip() or "PCA"
    old_pca_cols = [col for col in df.columns if str(col).startswith(f"{prefix}_")]
    if old_pca_cols:
        df = df.drop(columns=old_pca_cols)
    if bool(op.get("drop_original", False)):
        df = df.drop(columns=cols)
    for i in range(n_components):
        df[f"{prefix}_{i + 1}"] = transformed[:, i]
    return df


def _outlier_columns(df, op):
    cols = op.get("cols")
    if cols:
        return _resolve_columns(df, cols)
    return list(df.select_dtypes(include=[np.number]).columns)


def _outlier_mask(series, info):
    lower = info.get("lower_bound")
    upper = info.get("upper_bound")
    if lower is None or upper is None:
        return pd.Series(False, index=series.index)
    values = pd.to_numeric(series, errors="raise")
    return (values < lower) | (values > upper)


def _apply_outlier_operation(df, op, mode):
    cols = _outlier_columns(df, op)
    coefficient = float(op.get("coefficient", 1.5) or 1.5)
    info_map = detect_outliers(df[cols], coefficient=coefficient)
    touched = 0
    drop_indices = set()

    for col in cols:
        info = info_map.get(col)
        if not info or int(info.get("count", 0)) <= 0:
            continue
        mask = _outlier_mask(df[col], info)
        if not mask.any():
            continue
        touched += int(mask.sum())
        if mode == "winsorize":
            df[col] = pd.to_numeric(df[col], errors="raise").clip(
                lower=info.get("lower_bound"),
                upper=info.get("upper_bound"),
            )
        elif mode == "replace":
            method = str(op.get("method") or "median").lower()
            clean_values = pd.to_numeric(df.loc[~mask, col], errors="raise")
            if len(clean_values) == 0:
                clean_values = pd.to_numeric(df[col], errors="raise")
            if method == "mean":
                replacement = clean_values.mean()
            elif method == "median":
                replacement = clean_values.median()
            else:
                raise ValueError(f"不支持的异常值替换方法：{method}")
            df.loc[mask, col] = replacement
        elif mode == "drop":
            drop_indices.update(info.get("indices", []))
        else:
            raise ValueError(f"不支持的异常值处理模式：{mode}")

    if mode == "drop" and drop_indices:
        df = df.drop(index=list(drop_indices), errors="ignore").reset_index(drop=True)
        touched = len(drop_indices)

    return df, touched


def apply_processing_operations(df, operations):
    """Apply structured processing operations and return (new_df, messages)."""
    df = df.copy()
    messages = []
    for op in operations or []:
        op_type = op.get("op")
        if op_type == "select_cols":
            cols = _resolve_columns(df, op.get("cols"))
            df = df.loc[:, cols]
            messages.append(f"已保留 {len(cols)} 个字段")
        elif op_type == "drop_na":
            before = len(df)
            df = df.dropna()
            messages.append(f"已删除 {before - len(df)} 行含缺失值记录")
        elif op_type == "drop_duplicates":
            before = len(df)
            df = df.drop_duplicates()
            messages.append(f"已删除 {before - len(df)} 行重复记录")
        elif op_type == "drop_cols":
            cols = _resolve_columns(df, op.get("cols"))
            df = df.drop(columns=cols, errors="ignore")
            messages.append(f"已删除 {len(cols)} 个字段")
        elif op_type == "drop_rows":
            indices = op.get("indices", [])
            positions = op.get("positions", [])
            if positions:
                df = df.drop(index=df.index[[int(i) for i in positions if 0 <= int(i) < len(df)]])
            elif indices:
                df = df.drop(index=indices, errors="ignore")
            messages.append("已删除指定行")
        elif op_type == "rename_col":
            old = _resolve_column(df, op.get("old"))
            new = str(op.get("new") or "").strip()
            if not new:
                raise ValueError("新列名不能为空。")
            df = df.rename(columns={old: new})
            messages.append(f"已重命名字段 {old} -> {new}")
        elif op_type in {"astype", "cast_type"}:
            col = _resolve_column(df, op.get("col"))
            df[col] = _cast_series(df[col], op.get("dtype"))
            messages.append(f"已转换字段类型：{col}")
        elif op_type == "fill_na":
            cols = _resolve_columns(df, op.get("cols") or [op.get("col")])
            for col in cols:
                df[col] = _fill_series(df[col], op.get("method"), op.get("value"))
            messages.append(f"已填充 {len(cols)} 个字段的缺失值")
        elif op_type == "scale":
            cols = _resolve_columns(df, op.get("cols"))
            df = _scale_columns(df, cols, op.get("method"))
            messages.append(f"已缩放 {len(cols)} 个数值字段")
        elif op_type == "add_noise":
            cols = _resolve_columns(df, op.get("cols") or [op.get("col")])
            rng = np.random.default_rng(op.get("seed"))
            level = float(op.get("level", 0.1))
            for col in cols:
                values = pd.to_numeric(df[col], errors="raise")
                df[col] = values + rng.normal(0, level, size=len(df))
            messages.append(f"已为 {len(cols)} 个字段添加噪声")
        elif op_type == "label_encode":
            cols = _resolve_columns(df, op.get("cols") or [op.get("col")])
            for col in cols:
                df[col] = pd.factorize(df[col], sort=True)[0]
            messages.append(f"已标签编码 {len(cols)} 个类别字段")
        elif op_type == "one_hot_encode":
            cols = _resolve_columns(df, op.get("cols") or [op.get("col")])
            max_new = int(op.get("max_new_columns", 200))
            estimated = sum(max(0, df[col].nunique(dropna=not bool(op.get("dummy_na", False))) - (1 if op.get("drop_first", True) else 0)) for col in cols)
            if estimated > max_new:
                raise ValueError(f"独热编码将新增约 {estimated} 列，超过限制 {max_new}。")
            df = pd.get_dummies(df, columns=cols, drop_first=bool(op.get("drop_first", True)), dummy_na=bool(op.get("dummy_na", False)))
            messages.append(f"已独热编码 {len(cols)} 个类别字段")
        elif op_type in {"formula", "custom_formula"}:
            new_col = _ensure_new_column_name(df, op.get("new_col"))
            df = _apply_formula(df, new_col, str(op.get("expr") or ""))
            messages.append(f"已生成计算列：{new_col}")
        elif op_type == "unary_calc":
            df = _apply_unary_calc(df, op)
            messages.append(f"已生成计算列：{op.get('new_col')}")
        elif op_type == "binary_calc":
            df = _apply_binary_calc(df, op)
            messages.append(f"已生成计算列：{op.get('new_col')}")
        elif op_type == "pca":
            df = _apply_pca(df, op)
            messages.append("已完成 PCA 降维")
        elif op_type in {"winsorize_outliers", "outlier_winsorize"}:
            df, touched = _apply_outlier_operation(df, op, "winsorize")
            messages.append(f"已缩尾处理 {touched} 个异常值")
        elif op_type in {"replace_outliers", "outlier_replace"}:
            df, touched = _apply_outlier_operation(df, op, "replace")
            messages.append(f"已替换 {touched} 个异常值")
        elif op_type in {"drop_outliers", "outlier_drop"}:
            df, touched = _apply_outlier_operation(df, op, "drop")
            messages.append(f"已删除 {touched} 行包含异常值的记录")
        else:
            raise ValueError(f"不支持的数据处理操作：{op_type}")

    df = df.replace([np.inf, -np.inf], np.nan)
    return df, messages


def _mad_outliers(series, coefficient=1.5):
    """MAD-based outlier detection. Fallback when IQR == 0.

    Returns (mask, lower_bound, upper_bound, method_label).
    Uses modified Z-score with MAD scaling factor 0.6745.
    If MAD is also zero, falls back to standard deviation.
    """
    median = series.median()
    mad = np.median(np.abs(series - median))

    if mad == 0 or np.isnan(mad):
        # MAD is zero: most values are identical. Last resort: standard deviation.
        std = series.std()
        if std == 0 or np.isnan(std) or len(series) < 4:
            # Truly constant column or too small: flag everything != median as "outlier"
            mask = series != median
            if mask.sum() == 0:
                return pd.Series(False, index=series.index), None, None, "none"
            # Use a small epsilon to define bounds around the median
            eps = max(1e-8, np.finfo(np.float64).eps * abs(median) * 10)
            lower_bound = round(float(median) - eps, 4)
            upper_bound = round(float(median) + eps, 4)
            return mask, lower_bound, upper_bound, "std"
        # Std is non-zero: use 2.5 std threshold (wider than usual since std is inflated by outliers)
        threshold = 2.5 if coefficient <= 1.5 else 4.0
        lower_bound = round(median - threshold * std, 4)
        upper_bound = round(median + threshold * std, 4)
        mask = (series < lower_bound) | (series > upper_bound)
        return mask, lower_bound, upper_bound, "std"

    # 0.6745 * (x - median) / mad ~ N(0, 1) for normal data
    # Use 3.5 as threshold -> roughly equivalent to IQR * 1.5 in terms of false positive rate
    threshold = 3.5 if coefficient <= 1.5 else 5.0
    modified_z = 0.6745 * (series - median) / mad
    mask = modified_z.abs() > threshold
    lower_bound = round(median - threshold * mad / 0.6745, 4)
    upper_bound = round(median + threshold * mad / 0.6745, 4)
    return mask, lower_bound, upper_bound, "mad"


def detect_outliers(df, coefficient=1.5):
    """Detect outliers in numeric columns.

    Uses IQR (Q1 - coeff*IQR, Q3 + coeff*IQR) as primary method.
    Falls back to MAD when IQR == 0. NaN values are counted but excluded from detection.

    Args:
        df: pandas DataFrame
        coefficient: IQR multiplier (1.5 = standard, 3.0 = extreme)

    Returns:
        {col_name: {
            count, indices, values, severities, directions,
            lower_bound, upper_bound, nan_count, method
        }}
    """
    if coefficient <= 0:
        coefficient = 1.5
    outliers = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        # Drop NaN for detection, but count and report them
        col_series = df[col]
        valid = col_series.dropna()
        nan_count = int(col_series.isna().sum())

        if len(valid) < 4:
            continue  # too few values to detect outliers meaningfully

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        method = "iqr"

        if iqr == 0:
            # Fallback to MAD
            mask, lower, upper, method = _mad_outliers(valid, coefficient)
            if method == "none":
                if nan_count > 0:
                    outliers[col] = {
                        "count": 0, "indices": [], "values": [],
                        "severities": [], "directions": [],
                        "lower_bound": None, "upper_bound": None,
                        "nan_count": nan_count, "method": "none"
                    }
                continue
        else:
            lower = q1 - coefficient * iqr
            upper = q3 + coefficient * iqr
            mask = (valid < lower) | (valid > upper)

        outlier_count = int(mask.sum())
        if outlier_count > 0 or nan_count > 0:
            outlier_indices = valid.index[mask].tolist()
            outlier_values = valid.loc[mask].tolist()

            # Compute severity: how far beyond the fence relative to the acceptable range width.
            # A severity of 1.0 means "as far out as the entire acceptable range is wide."
            fence_width = upper - lower
            scale = max(fence_width, 1e-10)
            severities = []
            directions = []
            for val in outlier_values:
                if val > upper:
                    severities.append(round((val - upper) / scale, 2))
                    directions.append("high")
                else:
                    severities.append(round((lower - val) / scale, 2))
                    directions.append("low")

            outliers[col] = {
                "count": outlier_count,
                "indices": outlier_indices,
                "values": outlier_values,
                "severities": severities,
                "directions": directions,
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
                "nan_count": nan_count,
                "method": method,
            }

    return outliers


def build_data_summary(df, max_chars=8000):
    """Generate a text summary of the dataset (for LLM analysis). Truncated to max_chars."""
    lines = []
    lines.append("## Data Summary")
    lines.append(f"- Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    col_names = ', '.join(str(c) for c in df.columns)
    if len(col_names) > 500:
        col_names = col_names[:497] + "..."
    lines.append(f"- Column names: {col_names}")
    lines.append("")

    lines.append("### Column Types")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        lines.append(f"- {dtype}: {count} columns")
    lines.append("")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0 and len(numeric_cols) <= 100:
        desc = df[numeric_cols].describe().to_string()
        if len(desc) > 2000:
            desc = desc[:1997] + "..."
        lines.append("### Numeric Columns Statistics")
        lines.append(desc)
        lines.append("")

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            corr_upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            corr_pairs = corr_upper.stack().reset_index()
            corr_pairs.columns = ["Col A", "Col B", "Correlation"]
            corr_pairs["AbsCorr"] = corr_pairs["Correlation"].abs()
            corr_top = corr_pairs.sort_values("AbsCorr", ascending=False).head(5)
            if len(corr_top) > 0:
                lines.append("### Top Correlations")
                for _, row in corr_top.iterrows():
                    lines.append(f"- {row['Col A']} vs {row['Col B']}: {row['Correlation']:.3f}")
                lines.append("")
    elif len(numeric_cols) > 100:
        lines.append(f"### Numeric Columns: {len(numeric_cols)} (statistics skipped)")
        lines.append("")

    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        lines.append("### Categorical Columns")
        shown = 0
        for col in cat_cols:
            n_unique = df[col].nunique()
            n_total = df[col].notna().sum()
            lines.append(f"- **{col}**: {n_unique} unique / {n_total} non-null")
            shown += 1
            if shown >= 20:
                lines.append(f"- ... and {len(cat_cols) - shown} more columns")
                break
        lines.append("")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        lines.append("### Missing Values")
        shown = 0
        for col, cnt in missing.items():
            lines.append(f"- {col}: {cnt} ({cnt/len(df)*100:.1f}%)")
            shown += 1
            if shown >= 10:
                lines.append(f"- ... and {len(missing) - shown} more")
                break
        lines.append("")
    else:
        lines.append("### Missing Values: None")
        lines.append("")

    lines.append("### First 5 Rows (Preview)")
    preview = df.head(5).to_string()
    if len(preview) > 1500:
        preview = preview[:1497] + "..."
    lines.append(preview)

    full = "\n".join(lines)
    if len(full) > max_chars:
        full = full[:max_chars - 3] + "..."
    return full


def serialize_preview(df, rows=100):
    """Return JSON-safe preview of DataFrame."""
    preview = df.head(rows).copy()
    for col in preview.select_dtypes(include=['datetime64', 'datetimetz']).columns:
        preview[col] = preview[col].astype(str)
    # Convert any remaining non-serializable types
    for col in preview.columns:
        if preview[col].dtype == 'object':
            preview[col] = preview[col].apply(lambda x: str(x) if not isinstance(x, (str, int, float, bool, type(None), list, dict)) else x)
    records = preview.to_dict(orient="records")

    # Ensure all values are JSON-serializable (numpy scalars survive to_dict)
    for row in records:
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                row[k] = int(v)
            elif isinstance(v, (np.floating,)):
                row[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                row[k] = bool(v)
            elif isinstance(v, np.ndarray):
                row[k] = v.tolist()

    return records


def build_data_profile(df, session_id=None, session_meta=None, preview_rows=100):
    """Return a structured, JSON-safe dataset profile for frontend screens."""
    numeric_cols = [str(c) for c in df.select_dtypes(include=[np.number]).columns.tolist()]
    categorical_cols = [str(c) for c in df.select_dtypes(exclude=[np.number]).columns.tolist()]
    outliers = detect_outliers(df, coefficient=1.5)
    missing_counts = df.isnull().sum()
    dtypes = {str(col): str(dtype) for col, dtype in df.dtypes.items()}

    column_profiles = []
    for col in df.columns:
        name = str(col)
        dtype = str(df[col].dtype)
        missing_count = int(missing_counts[col])
        non_null = df[col].dropna()
        if name in numeric_cols:
            kind = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            kind = "datetime"
        elif pd.api.types.is_bool_dtype(df[col]):
            kind = "boolean"
        else:
            kind = "categorical"

        sample_values = []
        for value in non_null.head(5).tolist():
            if isinstance(value, (np.integer,)):
                sample_values.append(int(value))
            elif isinstance(value, (np.floating,)):
                sample_values.append(float(value))
            elif isinstance(value, (np.bool_,)):
                sample_values.append(bool(value))
            else:
                sample_values.append(str(value))

        stats = None
        if kind == "numeric":
            numeric_values = pd.to_numeric(non_null, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(numeric_values):
                stats = {
                    "mean": round(float(numeric_values.mean()), 6),
                    "median": round(float(numeric_values.median()), 6),
                    "var": round(float(numeric_values.var()), 6) if len(numeric_values) > 1 else 0.0,
                    "std": round(float(numeric_values.std()), 6) if len(numeric_values) > 1 else 0.0,
                    "min": round(float(numeric_values.min()), 6),
                    "max": round(float(numeric_values.max()), 6),
                }

        column_profiles.append({
            "name": name,
            "dtype": dtype,
            "kind": kind,
            "missing_count": missing_count,
            "missing_pct": round((missing_count / len(df) * 100), 2) if len(df) else 0,
            "unique_count": int(df[col].nunique(dropna=True)),
            "sample_values": sample_values,
            "outlier_count": int(outliers.get(col, {}).get("count", 0)),
            "nan_count": int(outliers.get(col, {}).get("nan_count", missing_count)),
            "stats": stats,
        })

    profile = {
        "session_id": session_id,
        "session_meta": session_meta,
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "columns": [str(c) for c in df.columns],
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "dtypes": dtypes,
        "missing_counts": {str(k): int(v) for k, v in missing_counts.items() if int(v) > 0},
        "missing_total": int(missing_counts.sum()),
        "outliers": {str(k): v for k, v in outliers.items()},
        "column_profiles": column_profiles,
        "preview": serialize_preview(df, rows=preview_rows),
        "summary": build_data_summary(df),
    }
    return profile
