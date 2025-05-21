import os

from data_fetcher import fetch_and_save_historical_prices
from dataset_creator import calculate_and_save_support_levels
from support_levels import hmm_support_levels, quantile_regression_support

if __name__ == "__main__":

    # Create dataset folder if it doesn't exist
    os.makedirs("datasets", exist_ok=True)

    # fetch_and_save_historical_prices(debug=True)

    # List of support functions to use
    support_functions = [
        (quantile_regression_support, "quantile_regression"),
        # (hmm_support_levels, "hmm"),
    ]

    # Create datasets using all support functions
    calculate_and_save_support_levels(support_functions)
