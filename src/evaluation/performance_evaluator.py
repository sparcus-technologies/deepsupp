import glob
import os

import pandas as pd

from .data_manager import DataManager
from .metrics_calculator import MetricsCalculator
from .reporting import ReportGenerator


class SupportLevelEvaluator:
    """
    Evaluates the performance of different support level detection methods
    using six fundamental financial metrics as ground truth.
    """

    def __init__(self, reports_dir="reports"):
        self.reports_dir = reports_dir
        self.data_manager = DataManager()
        self.metrics_calculator = MetricsCalculator()
        self.report_generator = ReportGenerator()

        # Ensure datasets directory exists
        os.makedirs(self.reports_dir, exist_ok=True)

    def load_historical_prices(self):
        """Load historical price data with flexible file detection"""
        self.data_manager.load_historical_prices()

    def evaluate_all_methods(self) -> pd.DataFrame:
        """Evaluate all support level detection methods"""
        if self.data_manager.historical_prices is None:
            self.load_historical_prices()

        method_results = self.data_manager.load_support_method_results()

        if not method_results:
            raise ValueError("No support level results found")

        evaluation_results = []

        # Get common tickers
        historical_prices = self.data_manager.historical_prices
        common_tickers = set(historical_prices.columns)
        for method_df in method_results.values():
            common_tickers = common_tickers.intersection(set(method_df["Ticker"]))

        print(
            f"🔍 Evaluating {len(common_tickers)} tickers across {len(method_results)} methods"
        )

        for method_name, method_df in method_results.items():
            print(f"⚡ Evaluating {method_name}...")

            method_scores = self.metrics_calculator.evaluate_method(
                method_name, method_df, common_tickers, historical_prices
            )

            evaluation_results.append(method_scores)

        results_df = pd.DataFrame(evaluation_results)
        results_df = results_df.sort_values("avg_overall_score", ascending=False)

        return results_df

    def generate_performance_report(self, results_df: pd.DataFrame) -> str:
        """Generate detailed performance report"""
        return self.report_generator.generate_report(results_df)

    def save_results(self, results_df: pd.DataFrame, report: str):
        """Save evaluation results and report"""
        results_file = os.path.join(self.reports_dir, "support_methods_evaluation.csv")
        results_df.to_csv(results_file, index=False)
        print(f"💾 Results saved to: {results_file}")

        report_file = os.path.join(self.reports_dir, "performance_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"📝 Report saved to: {report_file}")


def run_evaluation():
    """Main function to run evaluation"""
    print("🚀 Starting Enhanced Support Level Detection Methods Evaluation...")
    print("Now with Volume Confirmation and Market Regime Sensitivity!")
    print("=" * 70)

    try:
        evaluator = SupportLevelEvaluator()
        results_df = evaluator.evaluate_all_methods()
        report = evaluator.generate_performance_report(results_df)

        print("\n" + report)
        evaluator.save_results(results_df, report)

        print("\n" + "=" * 70)
        print("✅ Enhanced evaluation completed successfully!")
        print("📁 Check the datasets folder for detailed results and report.")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
