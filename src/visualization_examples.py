from figures import (
    compare_multiple_tickers,
    plot_performance_comparison,
    visualize_method_comparison,
    visualize_support_levels,
)


def main():
    # Create output directory for figures
    fig_dir = "c:\\projects\\deepsupp\\output\\figures"

    # Example 1: Visualize a single stock with all support methods
    print("Generating support level visualization for AAPL...")
    visualize_support_levels(ticker="AAPL", fig_dir=fig_dir)

    # Example 2: Compare performance of different methods
    print("Generating performance comparison chart...")
    plot_performance_comparison(fig_dir=fig_dir)

    # Example 3: Visualize multiple stocks
    print("Generating visualizations for multiple stocks...")
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
    compare_multiple_tickers(tickers=tickers, fig_dir=fig_dir)

    # Example 4: Compare all methods for a single stock
    print("Generating method comparison for MSFT...")
    visualize_method_comparison(ticker="MSFT", fig_dir=fig_dir)

    print("All visualizations complete. Check the output directory.")


if __name__ == "__main__":
    main()
