"""Data parsing, outlier detection, and summary generation."""
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
