import pandas as pd
import yfinance as yf
from curl_cffi import requests
import csv


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


def get_sp500_tickers():
    """
    Get the list of S&P 500 tickers from the local CSV file.
    Returns only the base tickers for data fetching.
    """
    try:
        # Read the first row of the CSV to get the column names
        with open('sp500_support_local_minima_structured.csv', 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
        
        # Extract only the base ticker symbols (no support columns)
        base_tickers = []
        for ticker in header[1:]:  # Skip 'Date' column
            # Add only if it's a base ticker (doesn't have "_support_" in the name)
            if "_support_" not in ticker:
                base_tickers.append(ticker)
        
        print(f"Retrieved {len(base_tickers)} S&P 500 tickers from local CSV file")
        return base_tickers
    except Exception as e:
        print(f"Error getting S&P 500 tickers from local file: {e}")
        # Return a few tickers as fallback
        return ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]