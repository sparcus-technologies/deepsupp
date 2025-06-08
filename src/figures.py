import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

    # Define available methods
    methods = [
        # "deepsupp",
        "fibonacci",
        "fractal",
        "hmm",
        "local_minima",
        "moving_average",
        "quantile_regression",
    ]

    # Load historical price data
    historical_prices = load_historical_prices("datasets/sp500_historial_prices.csv")

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


def plot_performance_comparison(fig_dir="figures"):
    """Create bar chart comparing method performance"""

    # Sample performance data based on the paper's results table
    methods = [
        "DeepSupp",
        "HMM",
        "Quantile Regression",
        "Local Minima",
        "Fractal",
        "Moving Average",
        "Fibonacci",
        "Percentile",
    ]

    metrics = {
        "Support Accuracy": [0.782, 0.753, 0.744, 0.726, 0.698, 0.682, 0.643, 0.624],
        "Price Proximity": [0.754, 0.731, 0.769, 0.685, 0.673, 0.654, 0.712, 0.748],
        "Volume Confirmation": [0.812, 0.692, 0.642, 0.689, 0.672, 0.683, 0.605, 0.512],
        "Market Sensitivity": [0.776, 0.704, 0.663, 0.642, 0.635, 0.712, 0.625, 0.583],
        "Support Duration": [0.685, 0.644, 0.718, 0.691, 0.643, 0.587, 0.615, 0.568],
        "Breakout Recovery": [0.721, 0.698, 0.685, 0.702, 0.714, 0.675, 0.687, 0.654],
        "Overall Score": [0.763, 0.720, 0.712, 0.689, 0.671, 0.665, 0.647, 0.615],
    }

    # Create DataFrame
    df = pd.DataFrame(metrics, index=methods)

    # Plot overall performance
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=df.index, y=df["Overall Score"], palette="viridis")
    plt.title("Overall Performance Comparison", fontsize=14)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Method", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    # Add value labels
    for i, v in enumerate(df["Overall Score"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "performance_comparison.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Saved performance comparison to {fig_path}")

    # Create a detailed comparison of all metrics
    plt.figure(figsize=(14, 8))
    df_melted = df.reset_index().melt(
        id_vars="index", var_name="Metric", value_name="Score"
    )

    # Don't include Overall Score in the detailed comparison
    df_melted = df_melted[df_melted["Metric"] != "Overall Score"]

    sns.barplot(x="index", y="Score", hue="Metric", data=df_melted, palette="viridis")
    plt.title("Detailed Performance Metrics Comparison", fontsize=14)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Method", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    detailed_fig_path = os.path.join(fig_dir, "detailed_performance_comparison.png")
    plt.savefig(detailed_fig_path, dpi=300)
    print(f"Saved detailed performance comparison to {detailed_fig_path}")


def compare_multiple_tickers(
    tickers, methods=None, fig_dir="figures", lookback_days=90
):
    """
    Generate support level visualizations for multiple tickers

    Parameters:
    -----------
    tickers : list
        List of ticker symbols to visualize
    methods : list
        List of prediction methods to include
    fig_dir : str
        Directory to save figures
    lookback_days : int
        Number of days to look back for the charts
    """

    fig_paths = []
    for ticker in tickers:
        try:
            fig_path = visualize_support_levels(
                ticker=ticker,
                methods=methods,
                fig_dir=fig_dir,
                lookback_days=lookback_days,
            )
            fig_paths.append(fig_path)
        except Exception as e:
            print(f"Error visualizing {ticker}: {e}")

    return fig_paths


def visualize_method_comparison(ticker="AAPL", fig_dir="figures"):
    """
    Generate a visualization comparing all methods for a single ticker

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol to visualize
    fig_dir : str
        Directory to save the figure
    """

    # Get all available methods
    methods = [
        "deepsupp",
        "hmm",
        "local_minima",
        "fractal",
        "moving_average",
        "fibonacci",
    ]

    # Load support levels for each method
    support_data = {}
    for method in methods:
        try:
            df = load_support_levels(method)
            if ticker in df["Ticker"].values:
                row = df[df["Ticker"] == ticker].iloc[0]
                levels = [
                    row[f"Support Level {i}"]
                    for i in range(1, 8)
                    if pd.notna(row[f"Support Level {i}"])
                ]
                support_data[method] = levels
        except Exception as e:
            print(f"Error loading {method} support levels: {e}")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the support levels for each method
    y_positions = list(range(len(methods)))
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))

    for i, method in enumerate(methods):
        if method in support_data:
            levels = support_data[method]
            ax.scatter(
                [level for level in levels],
                [i] * len(levels),
                label=method.capitalize(),
                color=colors[i],
                s=80,
                alpha=0.7,
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([m.capitalize() for m in methods])
    ax.set_title(f"Support Level Comparison for {ticker}", fontsize=14)
    ax.set_xlabel("Price Level ($)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, f"{ticker}_method_comparison.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Saved method comparison to {fig_path}")

    return fig_path


def create_research_figures():
    """Create all figures for the research paper"""

    print("Generating research paper figures...")

    # Create figures directory
    fig_dir = "figures"

    # Generate support level visualization
    visualize_support_levels(fig_dir=fig_dir)

    # Generate performance comparison charts
    # plot_performance_comparison(fig_dir=fig_dir)
