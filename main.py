import os

from dataset_creator import create_sp500_support_dataset
from support_levels import (
    quantile_regression_support,
    hmm_support_levels
)

if __name__ == "__main__":
    try:
        # Create dataset folder if it doesn't exist
        os.makedirs("datasets", exist_ok=True)

        # List of support functions to use
        support_functions = [
            (quantile_regression_support, "quantile_regression"),
            (hmm_support_levels, "hmm")
        ]

        # Create datasets using all support functions
        datasets = create_sp500_support_dataset(
            support_functions=support_functions,
            period_years=2,
            base_output_file="datasets/sp500_support",
        )

        print("Dataset creation completed!")
    except Exception as e:
        print(f"Error during dataset creation: {e}")