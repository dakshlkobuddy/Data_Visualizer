import pandas as pd


def get_column_groups(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    low_cardinality_numeric = [
        col for col in numeric_cols if df[col].nunique(dropna=False) <= 20 and col not in categorical_cols
    ]
    categorical_cols.extend(low_cardinality_numeric)
    categorical_cols = list(dict.fromkeys(categorical_cols))

    return numeric_cols, categorical_cols


def transform_data(
    df: pd.DataFrame,
    missing_strategy: str,
    remove_duplicates: bool,
    numeric_convert_cols: tuple[str, ...],
    datetime_convert_cols: tuple[str, ...],
    category_convert_cols: tuple[str, ...],
    apply_outlier_filter: bool,
    outlier_cols: tuple[str, ...],
    iqr_factor: float,
    enable_sampling: bool,
    sample_size: int,
    sample_seed: int,
) -> pd.DataFrame:
    cleaned_df = df.copy()

    for col in numeric_convert_cols:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")

    for col in datetime_convert_cols:
        cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors="coerce")

    for col in category_convert_cols:
        cleaned_df[col] = cleaned_df[col].astype("category")

    if missing_strategy == "Drop rows with missing values":
        cleaned_df = cleaned_df.dropna()
    elif missing_strategy == "Fill missing values":
        for col in cleaned_df.columns:
            if cleaned_df[col].isna().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            else:
                modes = cleaned_df[col].mode(dropna=True)
                fill_value = modes.iloc[0] if not modes.empty else "Unknown"
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)

    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()

    if apply_outlier_filter and outlier_cols:
        for col in outlier_cols:
            if not pd.api.types.is_numeric_dtype(cleaned_df[col]):
                continue
            q1 = cleaned_df[col].quantile(0.25)
            q3 = cleaned_df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - (iqr_factor * iqr)
            upper = q3 + (iqr_factor * iqr)
            cleaned_df = cleaned_df[(cleaned_df[col] >= lower) & (cleaned_df[col] <= upper)]

    if enable_sampling and 0 < sample_size < len(cleaned_df):
        cleaned_df = cleaned_df.sample(n=sample_size, random_state=sample_seed)

    return cleaned_df

