import numpy as np
import pandas as pd


def calculate_and_save_support_levels(
    support_functions,
    historical_prices_csv="datasets/sp500_historial_prices.csv",
    output_dir="datasets",
):

    historical_prices_df = pd.read_csv(historical_prices_csv)
    historical_prices_df.drop(columns=["Date"], inplace=True)
    tickers = historical_prices_df.columns

    num_levels = 7

    for support_function, function_name in support_functions:

        # Create a dataframe to store results for this support function
        results_df = pd.DataFrame(
            columns=["Ticker"] + [f"Support Level {i+1}" for i in range(num_levels)]
        )

        for ticker in tickers:
            # Skip tickers with missing data
            if historical_prices_df[ticker].isna().any():
                continue

            prices = historical_prices_df[ticker]
            support_levels = support_function(prices, num_levels=num_levels)

            # Create a row for this ticker
            row_data = {"Ticker": ticker}
            for i, level in enumerate(support_levels):
                row_data[f"Support Level {i+1}"] = level

            # Append to results dataframe
            results_df.loc[len(results_df)] = row_data

        # Save results to CSV
        output_path = f"{output_dir}/{function_name}_support_levels.csv"
        results_df.to_csv(output_path, index=False)
