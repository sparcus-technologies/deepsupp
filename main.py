import os

import pandas as pd

from src.data.data_fetcher import fetch_and_save_historical_prices
from src.data.make_date_filtered_dataset import make_date_filtered_dataset
from src.figures.visualize_support_levels import visualize_support_levels
from src.predict_support_levels import predict_support_levels
from src.support_methods.deepsupp import deepsupp
from src.support_methods.fibonacci import fibonacci_support
from src.support_methods.fractal import fractal_support
from src.support_methods.hmm import hmm_support_levels
from src.support_methods.local_minima import local_minima_support
from src.support_methods.moving_average import moving_average_support
from src.support_methods.quantile_regression import quantile_regression_support


def make_dataset():
    """
    Fetch historical prices for S&P 500 companies and create a dataset.
    This function fetches data for the last 2 years and saves it to a CSV file.
    It also creates a date-filtered dataset for the last 6 months.
    """

    # Create dataset folder if it doesn't exist
    os.makedirs("datasets", exist_ok=True)

    fetch_and_save_historical_prices(debug=True)


def make_support_levels():

    # Create output directory
    os.makedirs("predictions", exist_ok=True)

    # List of support functions to use
    support_functions = [
        (deepsupp, "deepsupp"),
        (fibonacci_support, "fibonacci"),
        (fractal_support, "fractal"),
        (hmm_support_levels, "hmm"),
        (local_minima_support, "local_minima"),
        (moving_average_support, "moving_average"),
        (quantile_regression_support, "quantile_regression"),
    ]

    predict_support_levels(support_functions)


def make_figures():
    fig_dir = "figures"

    # Create output directory for figures
    os.makedirs(fig_dir, exist_ok=True)

    print("Generating research paper figures...")

    # Generate support level visualization
    visualize_support_levels(fig_dir=fig_dir)


if __name__ == "__main__":
    # make_dataset()
    # make_date_filtered_dataset(date_period_months=6)
    # make_support_levels()
    make_figures()
