import os

from src.data_fetcher import fetch_and_save_historical_prices
from src.figures import create_research_figures
from src.predict_support_levels import predict_support_levels
from src.support_levels import (
    deepsupp,
    fibonacci_support,
    fractal_support,
    hmm_support_levels,
    local_minima_support,
    moving_average_support,
    percentile_support,
    quantile_regression_support,
)


def make_dataset():
    # Create dataset folder if it doesn't exist
    os.makedirs("datasets", exist_ok=True)

    fetch_and_save_historical_prices(debug=True)


def make_support_levels():
    # Create output directory
    os.makedirs("predictions", exist_ok=True)

    # List of support functions to use - now includes all methods
    support_functions = [
        (quantile_regression_support, "quantile_regression"),
        (hmm_support_levels, "hmm"),
        (local_minima_support, "local_minima"),
        (moving_average_support, "moving_average"),
        (fibonacci_support, "fibonacci"),
        (fractal_support, "fractal"),
        (deepsupp, "deepsupp"),
    ]

    predict_support_levels(support_functions)


def make_figures():
    # Create output directory for figures
    os.makedirs("output/figures", exist_ok=True)

    # Create research figures
    create_research_figures()


if __name__ == "__main__":
    # make_dataset()
    # make_support_levels()
    make_figures()
