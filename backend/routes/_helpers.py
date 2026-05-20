"""Small helpers shared by route modules."""


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
