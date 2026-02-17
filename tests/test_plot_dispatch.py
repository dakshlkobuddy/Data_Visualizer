import matplotlib
import pandas as pd
import pytest

from data_visualizer.plotting import PLOT_DISPATCH, PLOT_TYPES, create_plot

matplotlib.use("Agg")


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": ["a", "b", "a", "c"],
            "num1": [1, 2, 3, 4],
            "num2": [10, 20, 10, 30],
        }
    )


def test_dispatch_covers_all_supported_plot_types() -> None:
    assert set(PLOT_TYPES) == set(PLOT_DISPATCH.keys())


def test_create_plot_raises_for_unsupported_type(sample_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unsupported plot type"):
        create_plot(
            df=sample_df,
            plot_type="Unknown Plot",
            x_axis="num1",
            y_axis="num2",
            style="whitegrid",
            palette="deep",
            fig_width=8.0,
            fig_height=5.0,
            bins=20,
            marker_size=60,
            show_legend=True,
            x_scale="linear",
            y_scale="linear",
        )


def test_create_plot_returns_figure_for_scatter(sample_df: pd.DataFrame) -> None:
    fig = create_plot(
        df=sample_df,
        plot_type="Scatter Plot",
        x_axis="num1",
        y_axis="num2",
        style="whitegrid",
        palette="deep",
        fig_width=8.0,
        fig_height=5.0,
        bins=20,
        marker_size=60,
        show_legend=True,
        x_scale="linear",
        y_scale="linear",
    )
    assert fig is not None

