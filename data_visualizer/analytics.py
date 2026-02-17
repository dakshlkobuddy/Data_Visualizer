import pandas as pd


def build_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Column": df.columns,
            "Missing Count": df.isna().sum().values,
            "Missing %": ((df.isna().sum() / max(len(df), 1)) * 100).round(2).values,
        }
    ).sort_values("Missing Count", ascending=False)


def build_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return df.describe(include="all", datetime_is_numeric=True).transpose()
    except TypeError:
        return df.describe(include="all").transpose()


def build_grouped_aggregation(
    df: pd.DataFrame, group_col: str, value_col: str, agg_fn: str
) -> pd.DataFrame:
    grouped_series = df.groupby(group_col, dropna=False)[value_col].agg(agg_fn)
    if grouped_series.name == group_col:
        grouped_series = grouped_series.rename(f"{value_col}_{agg_fn}")
    return grouped_series.reset_index()

