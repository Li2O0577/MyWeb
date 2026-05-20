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


def coerce_column_like(df, column):
    """Convert one request column name to the DataFrame's column type when possible."""
    if df.empty or len(df.columns) == 0:
        return column

    col_type = type(df.columns[0])
    try:
        return col_type(column)
    except (ValueError, TypeError):
        return column
