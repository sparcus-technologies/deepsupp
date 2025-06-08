import os

from src.data_fetcher import fetch_and_save_historical_prices
from src.figures import create_research_figures
from src.predict_support_levels import predict_support_levels
from src.support_methods.deepsupp import deepsupp
from src.support_methods.fibonacci import fibonacci_support
from src.support_methods.fractal import fractal_support
from src.support_methods.hmm import hmm_support_levels
from src.support_methods.local_minima import local_minima_support
from src.support_methods.moving_average import moving_average_support
from src.support_methods.quantile_regression import quantile_regression_support


def make_dataset():
    # Create dataset folder if it doesn't exist
    os.makedirs("datasets", exist_ok=True)

    fetch_and_save_historical_prices(debug=True)


def make_support_levels():
    # Create output directory
    os.makedirs("predictions", exist_ok=True)

    # List of support functions to use
    support_functions = [
        # (deepsupp, "deepsupp"),
        (fibonacci_support, "fibonacci"),
        (fractal_support, "fractal"),
        (hmm_support_levels, "hmm"),
        (local_minima_support, "local_minima"),
        (moving_average_support, "moving_average"),
        (quantile_regression_support, "quantile_regression"),
    ]

    predict_support_levels(support_functions)


def make_figures():
    # Create output directory for figures
    os.makedirs("output/figures", exist_ok=True)

    # Create research figures
    create_research_figures()


if __name__ == "__main__":
    # make_dataset()
    make_support_levels()
    # make_figures()
