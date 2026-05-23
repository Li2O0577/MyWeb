"""Data parsing, outlier detection, and summary generation."""
import io
import pandas as pd
import numpy as np


def parse_file(file_bytes, filename):
    """Parse uploaded CSV/Excel into a DataFrame."""
    if filename.endswith('.csv'):
        return pd.read_csv(io.BytesIO(file_bytes))
    else:
        return pd.read_excel(io.BytesIO(file_bytes))


def detect_outliers(df):
    """IQR-based outlier detection. Returns {col: {count, indices, values, lower_bound, upper_bound}}."""
    outliers = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        if mask.any():
            outliers[col] = {
                "count": int(mask.sum()),
                "indices": df.index[mask].tolist(),
                "values": df.loc[mask, col].tolist(),
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
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
    return preview.to_dict(orient="records")
