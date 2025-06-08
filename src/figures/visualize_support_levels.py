import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter, MonthLocator


def load_historical_prices(file_path):
    """Load historical price data from CSV."""
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df


def load_support_levels(method):
    """Load support level predictions based on method name."""
    file_path = f"predictions/{method}_support_levels.csv"
    df = pd.read_csv(file_path)
    return df


def get_available_methods():
    """Discover available prediction methods by scanning the predictions directory."""
    prediction_files = glob.glob("predictions/*_support_levels.csv")
    methods = [
        os.path.basename(f).replace("_support_levels.csv", "") for f in prediction_files
    ]
    if not methods:
        print("Warning: No prediction methods found in the predictions directory.")
    return methods


def visualize_support_levels(ticker="AAPL", fig_dir="figures"):
    """
    Visualize stock price with support levels from different prediction methods.

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol to visualize
    fig_dir : str
        Directory to save the figure
    """

    # Dynamically get available methods from predictions directory
    methods = get_available_methods()

    if not methods:
        raise ValueError(
            "No prediction methods found. Please ensure prediction files exist in the 'predictions/' directory."
        )

    # Load historical price data
    historical_prices = load_historical_prices("datasets/sp500_filtered_prices.csv")

    # Filter for the selected ticker and recent data
    if ticker in historical_prices.columns:
        price_data = historical_prices[ticker].dropna()
    else:
        raise ValueError(f"Ticker {ticker} not found in historical price data.")

    # Create figure with subplots - one for each method
    num_methods = len(methods)
    fig, axes = plt.subplots(num_methods, 1, figsize=(12, 4 * num_methods), sharex=True)

    # Make axes iterable even if there's only one method
    if num_methods == 1:
        axes = [axes]

    # Plot support levels from each method on its own subplot
    for i, (method, ax) in enumerate(zip(methods, axes)):
        # Plot the stock price on each subplot
        ax.plot(
            price_data.index,
            price_data.values,
            label=f"{ticker} Price",
            color="black",
            linewidth=1.5,
        )

        # Plot support levels for the current method
        try:
            support_df = load_support_levels(method)
            if ticker in support_df["Ticker"].values:
                ticker_row = support_df[support_df["Ticker"] == ticker].iloc[0]

                # Get support levels (filter out NaN values)
                support_levels = [
                    ticker_row[f"Support Level {j}"]
                    for j in range(1, 8)
                    if pd.notna(ticker_row[f"Support Level {j}"])
                ]

                # Plot horizontal lines for each support level
                for level in support_levels:
                    ax.axhline(
                        y=level, color="red", alpha=0.6, linestyle="--", linewidth=1
                    )

                # Plot a small marker at the right edge of the chart for each level
                for level in support_levels:
                    if price_data.index[-1] in price_data.index:
                        ax.plot(
                            price_data.index[-1],
                            level,
                            "o",
                            color="red",
                            markersize=6,
                            label=f"Support ({level:.2f})",
                        )
        except Exception as e:
            print(f"Error plotting {method} support levels: {e}")

        # Format subplot
        ax.set_title(f"{method.capitalize()} Support Levels for {ticker}", fontsize=12)
        ax.set_ylabel("Price ($)", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="best", frameon=True)

    # Format x-axis on the bottom subplot only
    axes[-1].set_xlabel("Date", fontsize=12)

    # Format x-axis to show months
    axes[-1].xaxis.set_major_locator(MonthLocator())
    axes[-1].xaxis.set_major_formatter(DateFormatter("%b %Y"))

    # Rotate x-tick labels
    plt.setp(axes[-1].get_xticklabels(), rotation=45)

    # Add overall title
    fig.suptitle(f"{ticker} Stock Price with Support Levels", fontsize=16, y=0.995)

    plt.tight_layout()

    # Save the figure
    fig_path = os.path.join(fig_dir, f"{ticker}_support_levels.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {fig_path}")

    return fig_path
