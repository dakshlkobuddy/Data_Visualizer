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
PLOT_TYPES = ["Line Plot", "Bar Chart", "Scatter Plot", "Distribution Plot", "Count Plot"]


@st.cache_data
def load_csv_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


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
        df = pd.read_csv(io.BytesIO(content))
        return df, uploaded_file.name
    except Exception as err:
        st.error(f"Could not read uploaded CSV: {err}")
        return None, None


def render_plot(df: pd.DataFrame, plot_type: str, x_axis: str, y_axis: str | None) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))

    if plot_type == "Line Plot":
        sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax)
    elif plot_type == "Bar Chart":
        sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax)
    elif plot_type == "Scatter Plot":
        sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
    elif plot_type == "Distribution Plot":
        sns.histplot(data=df, x=x_axis, kde=True, ax=ax)
    elif plot_type == "Count Plot":
        sns.countplot(data=df, x=x_axis, ax=ax)

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

    if plot_type in ["Distribution Plot", "Count Plot"]:
        ax.set_title(f"{plot_type} of {x_axis}", fontsize=12)
    else:
        ax.set_title(f"{plot_type}: {y_axis} vs {x_axis}", fontsize=12)

    ax.set_xlabel(x_axis, fontsize=10)
    if plot_type == "Distribution Plot":
        ax.set_ylabel("Density", fontsize=10)
    elif plot_type == "Count Plot":
        ax.set_ylabel("Count", fontsize=10)
    else:
        ax.set_ylabel(str(y_axis), fontsize=10)

    fig.tight_layout()
    st.pyplot(fig)


# Data source selection (sample files or upload)
df, data_name = show_dataset_picker()
if df is None:
    st.info("Select a sample dataset or upload a CSV to begin.")
    st.stop()

st.caption(f"Dataset: {data_name} | Rows: {len(df)} | Columns: {len(df.columns)}")

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

if st.button("Generate Plot"):
    try:
        render_plot(df=df, plot_type=plot_type, x_axis=x_axis, y_axis=y_axis)
    except Exception as err:
        st.error(f"Plot generation failed: {err}")

