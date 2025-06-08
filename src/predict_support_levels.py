from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def predict_support_levels(
    support_functions,
    historical_prices_csv="dataset/sp500_historial_prices.csv",
    output_dir="predictions",
    date_period_months=6,
):

    historical_prices_df = pd.read_csv(historical_prices_csv)

    # Convert Date column to datetime
    historical_prices_df["Date"] = pd.to_datetime(historical_prices_df["Date"])

    # Filter to include only the last n months of data
    if date_period_months > 0:
        end_date = historical_prices_df["Date"].max()
        start_date = end_date - pd.DateOffset(months=date_period_months)
        historical_prices_df = historical_prices_df[
            historical_prices_df["Date"] >= start_date
        ]

    # Store tickers before dropping Date column
    tickers = [col for col in historical_prices_df.columns if col != "Date"]

    # Drop Date column after filtering
    historical_prices_df.drop(columns=["Date"], inplace=True)

    num_levels = 7

    for support_function, function_name in support_functions:

        print(f"Predicting support levels using {function_name}...")

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
