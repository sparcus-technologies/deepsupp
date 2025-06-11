import glob
import os

import numpy as np
import pandas as pd


class DataManager:
    """Handles data loading, creation and management for evaluation"""

    def __init__(self, datasets_dir="datasets", predictions_dir="predictions"):
        """Initialize DataManager with dataset directories"""
        self.datasets_dir = datasets_dir
        self.predictions_dir = predictions_dir
        self.historical_prices = None

    def find_price_file(self):
        """Find historical price file with flexible naming"""
        possible_names = ["sp500_filtered_prices.csv"]

        for name in possible_names:
            file_path = os.path.join(self.datasets_dir, name)
            if os.path.exists(file_path):
                print(f"✅ Found price file: {name}")
                return file_path

        # Look for any CSV with 'price' in name
        csv_files = glob.glob(os.path.join(self.datasets_dir, "*price*.csv"))
        if csv_files:
            print(f"✅ Found price file: {os.path.basename(csv_files[0])}")
            return csv_files[0]

        return None

    def load_historical_prices(self):
        """Load historical price data with flexible file detection"""
        price_file = self.find_price_file()

        if price_file is None:
            print("❌ No historical price file found!")
            self.historical_prices = self.create_sample_data()
            return

        try:
            self.historical_prices = pd.read_csv(
                price_file, index_col=0, parse_dates=True
            )
            print(
                f"✅ Loaded historical prices: {len(self.historical_prices.columns) -1} tickers"
            )

        except Exception as e:
            print(f"❌ Error loading {price_file}: {e}")
            print("Creating sample data instead...")
            self.historical_prices = self.create_sample_data()

    def load_support_method_results(self):
        """Load all support level detection results"""
        method_results = {}

        csv_files = glob.glob(
            os.path.join(self.predictions_dir, "*_support_levels.csv")
        )

        if not csv_files:
            print("❌ No support level CSV files found!")
            print("Creating sample support data...")
            self.create_sample_support_data()
            csv_files = glob.glob(
                os.path.join(self.predictions_dir, "*_support_levels.csv")
            )

        for csv_file in csv_files:
            method_name = os.path.basename(csv_file).replace("_support_levels.csv", "")
            try:
                df = pd.read_csv(csv_file)
                method_results[method_name] = df
                print(f"✅ Loaded {method_name}: {len(df)} tickers")
            except Exception as e:
                print(f"❌ Error loading {csv_file}: {e}")

        return method_results

    def create_sample_data(self):
        """Create sample data if no price file exists"""
        print("📥 No historical price file found. Creating sample data...")

        # Create sample S&P 500 data
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", end="2024-12-31", freq="D")
        tickers = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "JPM",
            "JNJ",
            "PG",
        ]

        price_data = {}
        for ticker in tickers:
            # Generate realistic price series
            base_price = np.random.uniform(50, 300)
            returns = np.random.normal(0.0005, 0.02, len(dates))

            # Add some trend and autocorrelation
            for i in range(1, len(returns)):
                returns[i] += 0.1 * returns[i - 1]

            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            price_data[ticker] = prices

        # Create DataFrame and save
        df = pd.DataFrame(price_data, index=dates)
        sample_file = os.path.join(self.datasets_dir, "sp500_historial_prices.csv")
        df.to_csv(sample_file)

        print(f"💾 Sample data created: {sample_file}")
        print(f"📊 Data: {len(tickers)} tickers, {len(dates)} days")

        return df

    def create_sample_support_data(self):
        """Create sample support level data if none exists"""
        print("📥 Creating sample support level data...")

        if self.historical_prices is None:
            return

        tickers = list(self.historical_prices.columns)
        methods = ["moving_average", "fibonacci", "local_minima"]

        for method in methods:
            support_data = []

            for ticker in tickers:
                prices = self.historical_prices[ticker].dropna()
                if len(prices) < 20:
                    continue

                # Generate realistic support levels
                support_levels = []
                percentiles = [5, 10, 15, 20, 25, 30, 35]

                for p in percentiles:
                    level = np.percentile(prices, p)
                    # Add method-specific variation
                    level *= np.random.uniform(0.98, 1.02)
                    support_levels.append(level)

                # Create row
                row_data = {"Ticker": ticker}
                for i, level in enumerate(support_levels, 1):
                    row_data[f"Support Level {i}"] = level

                support_data.append(row_data)

            # Save to CSV
            df = pd.DataFrame(support_data)
            filename = os.path.join(self.datasets_dir, f"{method}_support_levels.csv")
            df.to_csv(filename, index=False)
            print(f"💾 Created {method} support data: {filename}")
