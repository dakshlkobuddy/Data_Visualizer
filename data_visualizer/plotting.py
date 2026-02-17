from collections.abc import Callable
from typing import TypedDict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_visualizer.data_processing import get_column_groups

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

HEATMAP_CMAPS = {"Blues", "viridis"}


class AxisConfig(TypedDict):
    x_options: list[str]
    y_options: list[str]
    requires_y: bool
    x_fixed: str | None


def get_axis_config(df: pd.DataFrame, plot_type: str) -> AxisConfig:
    numeric_cols, categorical_cols = get_column_groups(df)
    all_cols = df.columns.tolist()

    if plot_type in {"Line Plot", "Scatter Plot"}:
        if not numeric_cols:
            raise ValueError("This plot requires numeric columns for both X and Y.")
        return {"x_options": numeric_cols, "y_options": numeric_cols, "requires_y": True, "x_fixed": None}
    if plot_type == "Bar Chart":
        if not numeric_cols:
            raise ValueError("Bar chart requires at least one numeric column for Y-axis.")
        return {"x_options": all_cols, "y_options": numeric_cols, "requires_y": True, "x_fixed": None}
    if plot_type == "Distribution Plot":
        if not numeric_cols:
            raise ValueError("Distribution plot requires a numeric X-axis column.")
        return {"x_options": numeric_cols, "y_options": [], "requires_y": False, "x_fixed": None}
    if plot_type == "Count Plot":
        if not categorical_cols:
            raise ValueError("Count plot requires a categorical or low-cardinality column.")
        return {"x_options": categorical_cols, "y_options": [], "requires_y": False, "x_fixed": None}
    if plot_type in {"Box Plot", "Violin Plot"}:
        if not numeric_cols:
            raise ValueError(f"{plot_type} requires at least one numeric column for Y-axis.")
        return {"x_options": all_cols, "y_options": numeric_cols, "requires_y": True, "x_fixed": None}
    if plot_type == "Correlation Heatmap":
        if len(numeric_cols) < 2:
            raise ValueError("Correlation heatmap requires at least two numeric columns.")
        return {"x_options": [], "y_options": [], "requires_y": False, "x_fixed": "Numeric Features"}

    raise ValueError(f"Unsupported plot type: {plot_type}")


def validate_plot_request(df: pd.DataFrame, plot_type: str, x_axis: str | None, y_axis: str | None) -> None:
    config = get_axis_config(df, plot_type)
    if config["x_fixed"] is not None:
        return

    x_options = config["x_options"]
    y_options = config["y_options"]
    requires_y = config["requires_y"]

    if x_axis not in x_options:
        raise ValueError("Invalid X-axis for selected plot type.")
    if requires_y and y_axis not in y_options:
        raise ValueError("Invalid Y-axis for selected plot type.")


def _single_color(palette: str) -> tuple[float, float, float]:
    try:
        return sns.color_palette(palette, n_colors=1)[0]
    except Exception:
        return sns.color_palette("deep", n_colors=1)[0]


def _plot_line(ax: plt.Axes, df: pd.DataFrame, x_axis: str, y_axis: str | None, color: tuple[float, float, float], marker_size: int, bins: int, palette: str, show_legend: bool) -> None:
    sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax, color=color)


def _plot_bar(ax: plt.Axes, df: pd.DataFrame, x_axis: str, y_axis: str | None, color: tuple[float, float, float], marker_size: int, bins: int, palette: str, show_legend: bool) -> None:
    sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax, color=color)


def _plot_scatter(ax: plt.Axes, df: pd.DataFrame, x_axis: str, y_axis: str | None, color: tuple[float, float, float], marker_size: int, bins: int, palette: str, show_legend: bool) -> None:
    sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax, s=marker_size, legend=show_legend, color=color)


def _plot_distribution(ax: plt.Axes, df: pd.DataFrame, x_axis: str, y_axis: str | None, color: tuple[float, float, float], marker_size: int, bins: int, palette: str, show_legend: bool) -> None:
    sns.histplot(data=df, x=x_axis, kde=True, bins=bins, ax=ax, color=color)


def _plot_count(ax: plt.Axes, df: pd.DataFrame, x_axis: str, y_axis: str | None, color: tuple[float, float, float], marker_size: int, bins: int, palette: str, show_legend: bool) -> None:
    sns.countplot(data=df, x=x_axis, ax=ax, color=color)


def _plot_box(ax: plt.Axes, df: pd.DataFrame, x_axis: str, y_axis: str | None, color: tuple[float, float, float], marker_size: int, bins: int, palette: str, show_legend: bool) -> None:
    sns.boxplot(data=df, x=x_axis, y=y_axis, ax=ax, color=color)


def _plot_violin(ax: plt.Axes, df: pd.DataFrame, x_axis: str, y_axis: str | None, color: tuple[float, float, float], marker_size: int, bins: int, palette: str, show_legend: bool) -> None:
    sns.violinplot(data=df, x=x_axis, y=y_axis, ax=ax, color=color)


def _plot_heatmap(ax: plt.Axes, df: pd.DataFrame, x_axis: str, y_axis: str | None, color: tuple[float, float, float], marker_size: int, bins: int, palette: str, show_legend: bool) -> None:
    corr = df.select_dtypes(include="number").corr(numeric_only=True)
    cmap = palette if palette in HEATMAP_CMAPS else "Blues"
    sns.heatmap(corr, cmap=cmap, annot=False, linewidths=0.3, ax=ax)


PLOT_DISPATCH: dict[str, Callable[..., None]] = {
    "Line Plot": _plot_line,
    "Bar Chart": _plot_bar,
    "Scatter Plot": _plot_scatter,
    "Distribution Plot": _plot_distribution,
    "Count Plot": _plot_count,
    "Box Plot": _plot_box,
    "Violin Plot": _plot_violin,
    "Correlation Heatmap": _plot_heatmap,
}


def create_plot(
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
    validate_plot_request(df, plot_type, x_axis, y_axis)
    sns.set_style(style)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    color = _single_color(palette)

    plot_fn = PLOT_DISPATCH.get(plot_type)
    if plot_fn is None:
        raise ValueError(f"Unsupported plot type: {plot_type}")
    plot_fn(ax, df, x_axis, y_axis, color, marker_size, bins, palette, show_legend)

    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=10)

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

    if plot_type in {"Distribution Plot", "Count Plot", "Correlation Heatmap"}:
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
            pass
    if y_scale == "log":
        try:
            ax.set_yscale("log")
        except ValueError:
            pass

    if not show_legend and ax.get_legend() is not None:
        ax.get_legend().remove()

    fig.tight_layout()
    return fig
