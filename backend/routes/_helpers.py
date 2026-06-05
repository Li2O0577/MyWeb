"""Small helpers shared by route modules."""
import numpy as np

try:
    from routes._responses import service_error
except ModuleNotFoundError:
    from backend.routes._responses import service_error


def coerce_columns_like(df, columns):
    """Convert request column names to the DataFrame's column type when possible."""
    if df.empty or len(df.columns) == 0:
        return list(columns)

    col_type = type(df.columns[0])
    coerced = []
    for col in columns:
        try:
            coerced.append(col_type(col))
        except (ValueError, TypeError):
            coerced.append(col)
    return coerced


def coerce_column_like(df, column):
    """Convert one request column name to the DataFrame's column type when possible."""
    if df.empty or len(df.columns) == 0:
        return column

    col_type = type(df.columns[0])
    try:
        return col_type(column)
    except (ValueError, TypeError):
        return column


def numeric_prediction_error(values, context="预测输入", batch=False, max_rows=10000):
    """Return a Chinese validation error for numeric prediction payloads."""
    try:
        raw = np.asarray(values, dtype=object)
    except Exception:
        return f"{context}无法读取，请检查请求格式。"

    if raw.size == 0:
        return f"{context}为空，请提供特征值。"

    if batch:
        if raw.ndim != 2:
            return f"{context}应为二维数组 rows，例如 [[1, 2], [3, 4]]。"
        if raw.shape[0] > max_rows:
            return f"{context}单次最多支持 {max_rows} 行，当前为 {raw.shape[0]} 行。"
    elif raw.ndim != 1:
        return f"{context}应为一维特征数组，例如 [1, 2, 3]。"

    try:
        numeric = raw.astype(float)
    except (TypeError, ValueError):
        return f"{context}包含非数值内容，请先转换为数值或完成编码。"

    if np.isnan(numeric).any():
        return f"{context}包含空值 NaN，请补全后再预测。"
    if np.isinf(numeric).any():
        return f"{context}包含无穷值 Inf，请清洗后再预测。"
    return None


def is_model_missing_error(err):
    """Heuristic for service errors that mean no usable model/version exists."""
    text = str(err or "")
    return (
        "没有找到已保存" in text
        or "请先训练模型" in text
        or "切换到有效版本" in text
        or "not found" in text.lower()
    )


def prediction_exception_message():
    return "预测输入无法处理，请检查特征数量、顺序和数值类型是否与当前模型一致。"


def prediction_service_error(err):
    """Return a route response for a model-missing or prediction failure error."""
    code = "MODEL_NOT_FOUND" if is_model_missing_error(err) else "PREDICTION_FAILED"
    status = 404 if code == "MODEL_NOT_FOUND" else 400
    return service_error(code, err, status)
