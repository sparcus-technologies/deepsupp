import pandas as pd


def make_date_filtered_dataset(
    original_csv="datasets/sp500_historial_prices.csv", date_period_months=6
):
    """
    Filter the prices DataFrame to only include data from the last n months.
    """

    prices_df = pd.read_csv(original_csv)
    prices_df["Date"] = pd.to_datetime(prices_df["Date"])

    if date_period_months > 0:
        end_date = prices_df["Date"].max()
        start_date = end_date - pd.DateOffset(months=date_period_months)
        prices_df = prices_df[prices_df["Date"] >= start_date]

    prices_df.to_csv("datasets/sp500_filtered_prices.csv", index=False)
