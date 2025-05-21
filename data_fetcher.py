import csv

import pandas as pd
import yfinance as yf
from curl_cffi import requests


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
    with open("sp500_tickers.txt", "r") as file:
        tickers = [line.strip() for line in file.readlines()]
    return tickers
