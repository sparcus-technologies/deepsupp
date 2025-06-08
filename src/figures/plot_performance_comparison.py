import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
