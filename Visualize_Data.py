import os
import io
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Data Visualizer",
    layout="centered",
    page_icon=":bar_chart:",
)
st.title("Visual Data Hub")

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKING_DIR, "data")
PLOT_TYPES = [
    "Line Plot",
    "Bar Chart",
    "Scatter Plot",
    "Distribution Plot",
    "Count Plot",
    "Box Plot",
    "Violin Plot",
    "Correlation Heatmap",
]


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


def get_column_groups(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Add low-cardinality numeric columns as category-like options for count plots.
    low_cardinality_numeric = [
        col for col in numeric_cols if df[col].nunique(dropna=False) <= 20 and col not in categorical_cols
    ]
    categorical_cols.extend(low_cardinality_numeric)

    # Keep order stable and remove duplicates.
    categorical_cols = list(dict.fromkeys(categorical_cols))

    return numeric_cols, categorical_cols


@st.cache_data
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

    cleaned_df = transform_data(
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
        missing_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Missing Count": df.isna().sum().values,
                "Missing %": ((df.isna().sum() / max(len(df), 1)) * 100).round(2).values,
            }
        )
        st.dataframe(missing_df.sort_values("Missing Count", ascending=False), width="stretch")

    with st.expander("Descriptive Statistics", expanded=False):
        try:
            summary_df = df.describe(include="all", datetime_is_numeric=True).transpose()
        except TypeError:
            summary_df = df.describe(include="all").transpose()
        st.dataframe(summary_df, width="stretch")

    with st.expander("Groupby Aggregation", expanded=False):
        if len(df.columns) >= 2:
            group_col = st.selectbox("Group column", options=df.columns.tolist(), key="group_col")
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                value_col = st.selectbox("Value column", options=numeric_cols, key="value_col")
                agg_fn = st.selectbox("Aggregation", options=["mean", "sum", "median", "min", "max", "count"], key="agg_fn")
                grouped_series = df.groupby(group_col, dropna=False)[value_col].agg(agg_fn)
                if grouped_series.name == group_col:
                    grouped_series = grouped_series.rename(f"{value_col}_{agg_fn}")
                grouped = grouped_series.reset_index()
                st.dataframe(grouped, width="stretch")
            else:
                st.info("No numeric columns available for aggregation.")
        else:
            st.info("Need at least two columns for groupby analytics.")


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
        content = uploaded_file.getvalue()
        df = load_csv_from_upload(content)
        return df, uploaded_file.name
    except Exception as err:
        st.error(f"Could not read uploaded CSV: {err}")
        return None, None


def render_plot(
    df: pd.DataFrame,
    plot_type: str,
    x_axis: str,
    y_axis: str | None,
    style: str,
    palette: str,
    fig_width: float,
    fig_height: float,
    bins: int,
    marker_size: int,
    show_legend: bool,
    x_scale: str,
    y_scale: str,
) -> plt.Figure:
    sns.set_style(style)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if plot_type == "Line Plot":
        sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax, palette=palette)
    elif plot_type == "Bar Chart":
        sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax, palette=palette)
    elif plot_type == "Scatter Plot":
        sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax, s=marker_size, legend=show_legend, palette=palette)
    elif plot_type == "Distribution Plot":
        sns.histplot(data=df, x=x_axis, kde=True, bins=bins, ax=ax, color=sns.color_palette(palette)[0])
    elif plot_type == "Count Plot":
        sns.countplot(data=df, x=x_axis, ax=ax, palette=palette)
    elif plot_type == "Box Plot":
        sns.boxplot(data=df, x=x_axis, y=y_axis, ax=ax, palette=palette)
    elif plot_type == "Violin Plot":
        sns.violinplot(data=df, x=x_axis, y=y_axis, ax=ax, palette=palette)
    elif plot_type == "Correlation Heatmap":
        corr = df.select_dtypes(include="number").corr(numeric_only=True)
        heatmap_cmap = palette if palette in ["Blues", "viridis"] else "Blues"
        sns.heatmap(corr, cmap=heatmap_cmap, annot=False, linewidths=0.3, ax=ax)

    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=10)

    # Keep crowded x-axis labels readable for high-cardinality columns.
    labels = ax.get_xticklabels()
    if len(labels) > 12:
        for label in labels:
            label.set_rotation(45)
            label.set_horizontalalignment("right")
    if len(labels) > 16:
        step = max(1, len(labels) // 12)
        for i, label in enumerate(labels):
            if i % step != 0:
                label.set_visible(False)

    if plot_type in ["Distribution Plot", "Count Plot", "Correlation Heatmap"]:
        ax.set_title(f"{plot_type} of {x_axis}", fontsize=12)
    else:
        ax.set_title(f"{plot_type}: {y_axis} vs {x_axis}", fontsize=12)

    if plot_type == "Correlation Heatmap":
        ax.set_xlabel("", fontsize=10)
        ax.set_ylabel("", fontsize=10)
    else:
        ax.set_xlabel(x_axis, fontsize=10)

    if plot_type == "Distribution Plot":
        ax.set_ylabel("Density", fontsize=10)
    elif plot_type == "Count Plot":
        ax.set_ylabel("Count", fontsize=10)
    elif plot_type == "Correlation Heatmap":
        ax.set_ylabel("", fontsize=10)
    else:
        ax.set_ylabel(str(y_axis), fontsize=10)

    if x_scale == "log":
        try:
            ax.set_xscale("log")
        except ValueError:
            st.warning("X-axis cannot be displayed in log scale for this data.")
    if y_scale == "log":
        try:
            ax.set_yscale("log")
        except ValueError:
            st.warning("Y-axis cannot be displayed in log scale for this data.")

    if not show_legend and ax.get_legend() is not None:
        ax.get_legend().remove()

    fig.tight_layout()
    return fig


# Data source selection (sample files or upload)
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

    if plot_type in ["Line Plot", "Scatter Plot"]:
        valid_x = numeric_cols
        valid_y = numeric_cols
        if not valid_x or not valid_y:
            st.warning("This plot requires numeric columns for both X and Y.")
            st.stop()
        x_axis = st.selectbox("Select the X-axis", options=valid_x)
        y_axis = st.selectbox("Select the Y-axis", options=valid_y)

    elif plot_type == "Bar Chart":
        valid_x = df.columns.tolist()
        valid_y = numeric_cols
        if not valid_y:
            st.warning("Bar chart requires at least one numeric column for Y-axis.")
            st.stop()
        x_axis = st.selectbox("Select the X-axis", options=valid_x)
        y_axis = st.selectbox("Select the Y-axis", options=valid_y)

    elif plot_type == "Distribution Plot":
        valid_x = numeric_cols
        if not valid_x:
            st.warning("Distribution plot requires a numeric X-axis column.")
            st.stop()
        x_axis = st.selectbox("Select the X-axis", options=valid_x)

    elif plot_type == "Count Plot":
        valid_x = categorical_cols
        if not valid_x:
            st.warning("Count plot requires a categorical or low-cardinality column.")
            st.stop()
        x_axis = st.selectbox("Select the X-axis", options=valid_x)
    elif plot_type in ["Box Plot", "Violin Plot"]:
        valid_x = df.columns.tolist()
        valid_y = numeric_cols
        if not valid_y:
            st.warning(f"{plot_type} requires at least one numeric column for Y-axis.")
            st.stop()
        x_axis = st.selectbox("Select the X-axis", options=valid_x)
        y_axis = st.selectbox("Select the Y-axis", options=valid_y)
    elif plot_type == "Correlation Heatmap":
        if len(numeric_cols) < 2:
            st.warning("Correlation heatmap requires at least two numeric columns.")
            st.stop()
        x_axis = "Numeric Features"
        y_axis = None

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
        fig = render_plot(
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

