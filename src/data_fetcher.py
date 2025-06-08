import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from curl_cffi import requests


def get_sp500_tickers():
    with open("sp500_tickers.txt", "r") as file:
        tickers = [line.strip() for line in file.readlines()]
    return tickers


def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a given ticker.
    """
    try:
        session = requests.Session(impersonate="chrome")
        stock_data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            session=session,
            auto_adjust=True,
        )
        if stock_data.empty:
            print(f"No data found for {ticker}")
            return None
        stock_data.reset_index(inplace=True)
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def fetch_and_save_historical_prices(
    period_years=2,
    output_file="dataset/sp500_historial_prices.csv",
    debug=False,
):
    """
    Fetch historical price data for S&P 500 companies and save it to a CSV file.

    Args:
        period_years: Number of years of historical data to use.
        base_output_file: Base name for output file.

    Returns:
        Tuple of (master_dataframe, stock_data_dictionary, tickers)
    """
    # Define date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=period_years * 365)).strftime(
        "%Y-%m-%d"
    )
    print(f"Creating S&P 500 dataset from {start_date} to {end_date}...")

    # Get S&P 500 tickers
    tickers = get_sp500_tickers()

    if debug:
        tickers = tickers[:5]
        print(f"Debug mode: Using only {len(tickers)} tickers.")

    # Create a master dataframe with dates as index
    date_range = pd.date_range(
        start=start_date, end=end_date, freq="B"
    )  # Business days
    df = pd.DataFrame(index=date_range)
    df.index.name = "Date"

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
        df[ticker] = stock_data["Close"]

        # Be nice to the API and avoid rate limits
        time.sleep(0.5)

    # Drop rows with all NaN values
    df.dropna(how="all", inplace=True)

    # Save the master dataframe with just closing prices
    df.to_csv(output_file)
