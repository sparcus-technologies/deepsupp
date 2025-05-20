import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from data_fetcher import fetch_stock_data, get_sp500_tickers
from support_levels import quantile_regression_support


def create_sp500_support_dataset(
    support_functions=None,
    period_years=2,
    base_output_file="datasets/sp500_support_dataset",
):
    """
    Create multiple datasets for S&P 500 companies using different support level identification methods.

    Args:
        support_functions: List of tuples containing (function, name) for support level identification.
                          Each function should accept price_data and return a list of support levels.
        period_years: Number of years of historical data to use.
        base_output_file: Base name for output files (will be appended with function names).

    Returns:
        Dictionary of dataframes for each support function.
    """
    # Default to just the built-in function if none provided
    if support_functions is None:
        support_functions = [(local_minima_support, "local_minima")]

    # Define date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=period_years * 365)).strftime(
        "%Y-%m-%d"
    )
    print(f"Creating S&P 500 dataset from {start_date} to {end_date}...")

    # Get S&P 500 tickers
    tickers = get_sp500_tickers()

    # Create a master dataframe with dates as index
    date_range = pd.date_range(
        start=start_date, end=end_date, freq="B"
    )  # Business days
    master_df = pd.DataFrame(index=date_range)
    master_df.index.name = "Date"

    # Fetch stock data once for all methods
    stock_data_dict = {}
    for i, ticker in enumerate(tickers):
        print(f"Fetching data for {ticker} ({i+1}/{len(tickers)})...")

        # Fetch stock data
        stock_data = fetch_stock_data(ticker, start_date, end_date)

        if stock_data is None or stock_data.empty:
            print(f"Skipping {ticker} - no data available")
            continue

        # Ensure dataset has proper date format
        if not pd.api.types.is_datetime64_any_dtype(stock_data["Date"]):
            stock_data["Date"] = pd.to_datetime(stock_data["Date"])

        # Set Date as index for merging
        stock_data.set_index("Date", inplace=True)

        # Store in dictionary
        stock_data_dict[ticker] = stock_data

        # Add closing price to the master dataframe
        master_df[ticker] = stock_data["Close"]

        # Be nice to the API and avoid rate limits
        time.sleep(0.5)

    print()

    # Dictionary to store results
    result_dfs = {}

    # Process each support function
    for support_func, func_name in support_functions:
        print(f"Processing with {func_name} support function...")

        # Create a copy of the master dataframe for this method
        method_df = master_df.copy()

        # Calculate support levels using this method for each ticker
        for ticker, stock_data in stock_data_dict.items():
            try:
                # Calculate support levels using the current function
                support_levels = support_func(stock_data["Close"], num_levels=3)

                # Add support levels information as a column
                support_str = ",".join([str(level) for level in support_levels])
                method_df[f"{ticker}_supports"] = support_str
            except Exception as e:
                print(f"Error calculating support for {ticker} with {func_name}: {e}")

        # Clean up the dataframe - fill NaN with previous values where possible
        method_df = method_df.ffill()

        # Remove rows that are still completely NaN
        method_df = method_df.dropna(how="all")

        # Generate output filenames
        output_file = f"{base_output_file}_{func_name}.csv"

        # Save to CSV
        method_df.to_csv(output_file)
        print(f"Dataset saved to {output_file}")

        # Create structured dataframe with separate support columns
        structured_df = method_df.copy()

        # Remove the support string columns and add individual support columns
        for ticker in tickers:
            support_col = f"{ticker}_supports"
            if support_col in structured_df.columns:
                # Get the most recent support values
                supports = structured_df[support_col].dropna()
                if not supports.empty:
                    latest_support_str = supports.iloc[-1]

                    # Remove the string support column
                    structured_df = structured_df.drop(columns=[support_col])

                    # Add individual support level columns if we have support data
                    if isinstance(latest_support_str, str):
                        support_values = latest_support_str.split(",")
                        for i, val in enumerate(support_values):
                            try:
                                structured_df[f"{ticker}_support_{i+1}"] = float(val)
                            except ValueError:
                                print(
                                    f"Warning: Could not convert support value '{val}' to float"
                                )

        # Save the structured format
        structured_output = output_file.replace(".csv", "_structured.csv")
        structured_df.to_csv(structured_output)
        print(f"Structured dataset saved to {structured_output}")

        # Store in results dictionary
        result_dfs[func_name] = method_df

        print()

    return result_dfs
