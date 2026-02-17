import pandas as pd
import pytest

from data_visualizer.plotting import get_axis_config, validate_plot_request


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": ["a", "b", "a", "c"],
            "num1": [1, 2, 3, 4],
            "num2": [10.5, 9.1, 8.2, 7.3],
        }
    )


def test_scatter_requires_numeric_axes(sample_df: pd.DataFrame) -> None:
    config = get_axis_config(sample_df, "Scatter Plot")
    assert config["requires_y"] is True
    assert "num1" in config["x_options"]
    assert "category" not in config["x_options"]


def test_correlation_heatmap_requires_two_numeric_columns() -> None:
    df = pd.DataFrame({"label": ["x", "y"], "value": [1, 2]})
    with pytest.raises(ValueError, match="Correlation heatmap requires at least two numeric columns."):
        get_axis_config(df, "Correlation Heatmap")


def test_validate_plot_request_rejects_invalid_y(sample_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Invalid Y-axis"):
        validate_plot_request(sample_df, "Line Plot", x_axis="num1", y_axis="category")

