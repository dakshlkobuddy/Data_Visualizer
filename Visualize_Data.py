import io
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data_visualizer.analytics import (
    build_descriptive_stats,
    build_grouped_aggregation,
    build_missing_summary,
)
from data_visualizer.data_processing import get_column_groups, transform_data
from data_visualizer.plotting import PLOT_TYPES, create_plot, get_axis_config

st.set_page_config(
    page_title="Data Visualizer",
    layout="centered",
    page_icon=":bar_chart:",
)
st.title("Visual Data Hub")

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKING_DIR, "data")


@st.cache_data
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_csv_from_upload(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content))


@st.cache_data
def list_sample_files(folder_path: str) -> list[str]:
    if not os.path.isdir(folder_path):
        return []
    return sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".csv")])


@st.cache_data
def cached_transform_data(
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
    return transform_data(
        df=df,
        missing_strategy=missing_strategy,
        remove_duplicates=remove_duplicates,
        numeric_convert_cols=numeric_convert_cols,
        datetime_convert_cols=datetime_convert_cols,
        category_convert_cols=category_convert_cols,
        apply_outlier_filter=apply_outlier_filter,
        outlier_cols=outlier_cols,
        iqr_factor=iqr_factor,
        enable_sampling=enable_sampling,
        sample_size=sample_size,
        sample_seed=sample_seed,
    )


def show_dataset_picker() -> tuple[pd.DataFrame | None, str | None]:
    source = st.radio("Choose data source", ["Sample dataset", "Upload CSV"], horizontal=True)

    if source == "Sample dataset":
        files = list_sample_files(DATA_DIR)
        if not files:
            st.warning("No sample CSV files found in the data folder.")
            return None, None

        selected_file = st.selectbox("Select a sample file", files, index=None)
        if not selected_file:
            return None, None

        file_path = os.path.join(DATA_DIR, selected_file)
        try:
            return load_csv_from_path(file_path), selected_file
        except Exception as err:
            st.error(f"Could not read sample file: {err}")
            return None, None

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if not uploaded_file:
        return None, None

    try:
        return load_csv_from_upload(uploaded_file.getvalue()), uploaded_file.name
    except Exception as err:
        st.error(f"Could not read uploaded CSV: {err}")
        return None, None


def apply_data_cleaning_controls(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander("Data Cleaning Controls", expanded=False):
        missing_strategy = st.selectbox(
            "Missing value handling",
            ["Do nothing", "Drop rows with missing values", "Fill missing values"],
        )
        remove_duplicates = st.checkbox("Remove duplicate rows", value=False)

        st.caption("Type conversion (optional)")
        numeric_convert_cols = st.multiselect("Convert to numeric", options=df.columns.tolist())
        datetime_convert_cols = st.multiselect("Convert to datetime", options=df.columns.tolist())
        category_convert_cols = st.multiselect("Convert to category", options=df.columns.tolist())

        apply_outlier_filter = st.checkbox("Filter outliers (IQR)", value=False)
        outlier_cols = []
        iqr_factor = 1.5
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if apply_outlier_filter and numeric_cols:
            outlier_cols = st.multiselect(
                "Outlier filter columns",
                options=numeric_cols,
                default=numeric_cols[:1],
            )
            iqr_factor = st.slider("IQR factor", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

        st.caption("Performance (large datasets)")
        enable_sampling = st.checkbox("Use row sampling", value=False)
        sample_size = min(5000, len(df))
        sample_seed = 42
        if enable_sampling:
            sample_size = st.slider(
                "Sample rows",
                min_value=100 if len(df) >= 100 else 1,
                max_value=len(df),
                value=min(5000, len(df)),
                step=100 if len(df) >= 100 else 1,
            )
            sample_seed = st.number_input("Sample seed", min_value=0, value=42, step=1)

    cleaned_df = cached_transform_data(
        df=df,
        missing_strategy=missing_strategy,
        remove_duplicates=remove_duplicates,
        numeric_convert_cols=tuple(numeric_convert_cols),
        datetime_convert_cols=tuple(datetime_convert_cols),
        category_convert_cols=tuple(category_convert_cols),
        apply_outlier_filter=apply_outlier_filter,
        outlier_cols=tuple(outlier_cols),
        iqr_factor=iqr_factor,
        enable_sampling=enable_sampling,
        sample_size=int(sample_size),
        sample_seed=int(sample_seed),
    )

    st.caption(
        f"After cleaning: Rows {len(cleaned_df)} (was {len(df)}), Columns {len(cleaned_df.columns)}"
    )
    return cleaned_df


def render_analytics_panels(df: pd.DataFrame) -> None:
    st.subheader("Analytics")

    overview_col1, overview_col2, overview_col3 = st.columns(3)
    with overview_col1:
        st.metric("Rows", len(df))
    with overview_col2:
        st.metric("Columns", len(df.columns))
    with overview_col3:
        st.metric("Duplicate Rows", int(df.duplicated().sum()))

    with st.expander("Missing Values Summary", expanded=False):
        st.dataframe(build_missing_summary(df), width="stretch")

    with st.expander("Descriptive Statistics", expanded=False):
        st.dataframe(build_descriptive_stats(df), width="stretch")

    with st.expander("Groupby Aggregation", expanded=False):
        if len(df.columns) < 2:
            st.info("Need at least two columns for groupby analytics.")
            return

        group_col = st.selectbox("Group column", options=df.columns.tolist(), key="group_col")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            st.info("No numeric columns available for aggregation.")
            return

        value_col = st.selectbox("Value column", options=numeric_cols, key="value_col")
        agg_fn = st.selectbox(
            "Aggregation", options=["mean", "sum", "median", "min", "max", "count"], key="agg_fn"
        )
        grouped = build_grouped_aggregation(df, group_col, value_col, agg_fn)
        st.dataframe(grouped, width="stretch")


df, data_name = show_dataset_picker()
if df is None:
    st.info("Select a sample dataset or upload a CSV to begin.")
    st.stop()

st.caption(f"Dataset: {data_name} | Rows: {len(df)} | Columns: {len(df.columns)}")
df = apply_data_cleaning_controls(df)
if df.empty:
    st.warning("No rows remain after cleaning. Adjust cleaning options and try again.")
    st.stop()

export_name = os.path.splitext(data_name)[0] if data_name else "dataset"
csv_export = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download cleaned dataset (CSV)",
    data=csv_export,
    file_name=f"{export_name}_cleaned.csv",
    mime="text/csv",
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Preview")
    st.dataframe(df.head(), width="stretch")

numeric_cols, categorical_cols = get_column_groups(df)

with col2:
    st.subheader("Plot Setup")
    plot_type = st.selectbox("Select the type of plot", options=PLOT_TYPES)
    x_axis = None
    y_axis = None

    try:
        config = get_axis_config(df, plot_type)
    except ValueError as err:
        st.warning(str(err))
        st.stop()

    if config["x_fixed"] is not None:
        x_axis = str(config["x_fixed"])
    else:
        x_axis = st.selectbox("Select the X-axis", options=config["x_options"])
    if config["requires_y"]:
        y_axis = st.selectbox("Select the Y-axis", options=config["y_options"])

    with st.expander("Plot Customization", expanded=False):
        style = st.selectbox("Theme", options=["whitegrid", "darkgrid", "white", "dark", "ticks"], index=0)
        palette_options = ["deep", "muted", "bright", "pastel", "dark", "colorblind", "Blues", "viridis"]
        palette = st.selectbox("Color palette", options=palette_options, index=0)
        fig_width = st.slider("Figure width", min_value=6.0, max_value=16.0, value=10.0, step=0.5)
        fig_height = st.slider("Figure height", min_value=4.0, max_value=10.0, value=5.5, step=0.5)
        bins = st.slider("Histogram bins", min_value=5, max_value=100, value=30, step=1)
        marker_size = st.slider("Marker size", min_value=10, max_value=300, value=60, step=10)
        show_legend = st.checkbox("Show legend", value=True)
        x_scale = st.selectbox("X-axis scale", options=["linear", "log"], index=0)
        y_scale = st.selectbox("Y-axis scale", options=["linear", "log"], index=0)

if st.button("Generate Plot"):
    try:
        fig = create_plot(
            df=df,
            plot_type=plot_type,
            x_axis=x_axis,
            y_axis=y_axis,
            style=style,
            palette=palette,
            fig_width=fig_width,
            fig_height=fig_height,
            bins=bins,
            marker_size=marker_size,
            show_legend=show_legend,
            x_scale=x_scale,
            y_scale=y_scale,
        )
        st.pyplot(fig)

        png_buffer = io.BytesIO()
        fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight")
        png_buffer.seek(0)

        svg_buffer = io.BytesIO()
        fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
        svg_buffer.seek(0)
        plt.close(fig)

        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                "Download plot (PNG)",
                data=png_buffer,
                file_name=f"{export_name}_{plot_type.replace(' ', '_').lower()}.png",
                mime="image/png",
            )
        with export_col2:
            st.download_button(
                "Download plot (SVG)",
                data=svg_buffer,
                file_name=f"{export_name}_{plot_type.replace(' ', '_').lower()}.svg",
                mime="image/svg+xml",
            )
    except Exception as err:
        st.error(f"Plot generation failed: {err}")

render_analytics_panels(df)

