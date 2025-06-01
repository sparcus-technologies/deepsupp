# performance_evaluator.py

import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class SupportLevelEvaluator:
    """
    Evaluates the performance of different support level detection methods
    using three fundamental financial metrics as ground truth.
    """
    
    def __init__(self, datasets_dir="datasets"):
        self.datasets_dir = datasets_dir
        self.historical_prices = None
        self.results = {}
        
    def load_historical_prices(self):
        """Load the historical price data"""
        price_file = os.path.join(self.datasets_dir, "sp500_historial_prices.csv")
        if os.path.exists(price_file):
            self.historical_prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
            print(f"Loaded historical prices for {len(self.historical_prices.columns)} tickers")
        else:
            raise FileNotFoundError(f"Historical prices file not found: {price_file}")
    
    def calculate_ground_truth_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """
        Calculate three fundamental financial metrics as ground truth:
        1. Support Accuracy: How often prices bounce off predicted support levels
        2. Price Proximity: How close support levels are to actual low prices
        3. Trend Stability: How well support levels align with market trends
        """
        metrics = {}
        
        # Clean price data
        clean_prices = prices.dropna()
        if len(clean_prices) < 20:
            return {"support_accuracy": 0, "price_proximity": 0, "trend_stability": 0}
        
        # 1. Support Accuracy - percentage of times price bounced off support
        # Find actual support events (local minima followed by upward movement)
        actual_supports = []
        for i in range(5, len(clean_prices) - 5):
            if (clean_prices.iloc[i] == min(clean_prices.iloc[i-5:i+6]) and 
                clean_prices.iloc[i+3] > clean_prices.iloc[i] * 1.01):  # 1% bounce
                actual_supports.append(clean_prices.iloc[i])
        
        # 2. Price Proximity - how close predicted levels are to actual price distribution
        price_percentiles = [np.percentile(clean_prices, p) for p in [5, 10, 15, 20, 25, 30, 35]]
        
        # 3. Trend Stability - volatility and consistency of the market
        returns = clean_prices.pct_change().dropna()
        volatility = returns.std()
        trend_consistency = abs(returns.mean()) / (volatility + 1e-8)
        
        metrics["actual_supports"] = actual_supports
        metrics["price_percentiles"] = price_percentiles
        metrics["volatility"] = volatility
        metrics["trend_consistency"] = trend_consistency
        
        return metrics
    
    def evaluate_support_method(self, predicted_levels: List[float], 
                              ground_truth: Dict[str, float], 
                              prices: pd.Series) -> Dict[str, float]:
        """
        Evaluate a support method against ground truth metrics
        """
        scores = {}
        clean_prices = prices.dropna()
        
        if not predicted_levels or len(clean_prices) < 20:
            return {"support_accuracy": 0, "price_proximity": 0, "trend_stability": 0, "overall_score": 0}
        
        # 1. Support Accuracy Score
        actual_supports = ground_truth.get("actual_supports", [])
        if actual_supports:
            accuracy_scores = []
            for actual_support in actual_supports:
                # Find closest predicted level
                closest_distance = min([abs(pred - actual_support) / actual_support 
                                      for pred in predicted_levels])
                # Score based on how close (within 5% is perfect, beyond 20% is zero)
                accuracy_score = max(0, 1 - (closest_distance / 0.05))
                accuracy_scores.append(accuracy_score)
            scores["support_accuracy"] = np.mean(accuracy_scores) if accuracy_scores else 0
        else:
            scores["support_accuracy"] = 0
        
        # 2. Price Proximity Score
        price_percentiles = ground_truth.get("price_percentiles", [])
        if price_percentiles:
            proximity_scores = []
            for i, predicted in enumerate(predicted_levels):
                if i < len(price_percentiles):
                    expected = price_percentiles[i]
                    relative_error = abs(predicted - expected) / expected
                    # Score decreases as error increases (within 10% is good)
                    proximity_score = max(0, 1 - (relative_error / 0.1))
                    proximity_scores.append(proximity_score)
            scores["price_proximity"] = np.mean(proximity_scores) if proximity_scores else 0
        else:
            scores["price_proximity"] = 0
        
        # 3. Trend Stability Score
        # Check if predicted levels are reasonably spaced and ordered
        if len(predicted_levels) > 1:
            level_spacing = np.diff(sorted(predicted_levels))
            avg_spacing = np.mean(level_spacing)
            spacing_consistency = 1 - (np.std(level_spacing) / (avg_spacing + 1e-8))
            spacing_consistency = max(0, min(1, spacing_consistency))
            
            # Check if levels are within reasonable range of price data
            price_range = clean_prices.max() - clean_prices.min()
            level_range = max(predicted_levels) - min(predicted_levels)
            range_ratio = min(1, level_range / (price_range + 1e-8))
            
            scores["trend_stability"] = (spacing_consistency + range_ratio) / 2
        else:
            scores["trend_stability"] = 0
        
        # 4. Overall Score (weighted average)
        weights = {"support_accuracy": 0.4, "price_proximity": 0.4, "trend_stability": 0.2}
        scores["overall_score"] = sum(scores[metric] * weights[metric] for metric in weights.keys())
        
        return scores
    
    def load_support_method_results(self) -> Dict[str, pd.DataFrame]:
        """Load all support level detection results"""
        method_results = {}
        
        # Find all CSV files with support levels
        csv_files = glob.glob(os.path.join(self.datasets_dir, "*_support_levels.csv"))
        
        for csv_file in csv_files:
            method_name = os.path.basename(csv_file).replace("_support_levels.csv", "")
            try:
                df = pd.read_csv(csv_file)
                method_results[method_name] = df
                print(f"Loaded {method_name} results: {len(df)} tickers")
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
        
        return method_results
    
    def evaluate_all_methods(self) -> pd.DataFrame:
        """
        Evaluate all support level detection methods and return comparison results
        """
        if self.historical_prices is None:
            self.load_historical_prices()
        
        method_results = self.load_support_method_results()
        
        if not method_results:
            raise ValueError("No support level results found in datasets directory")
        
        evaluation_results = []
        
        # Get common tickers across all methods
        common_tickers = set(self.historical_prices.columns)
        for method_df in method_results.values():
            common_tickers = common_tickers.intersection(set(method_df['Ticker']))
        
        print(f"Evaluating {len(common_tickers)} common tickers across {len(method_results)} methods")
        
        for method_name, method_df in method_results.items():
            print(f"\nEvaluating {method_name}...")
            
            method_scores = {
                "method": method_name,
                "support_accuracy": [],
                "price_proximity": [],
                "trend_stability": [],
                "overall_score": []
            }
            
            for ticker in common_tickers:
                # Get historical prices for this ticker
                if ticker not in self.historical_prices.columns:
                    continue
                    
                prices = self.historical_prices[ticker].dropna()
                if len(prices) < 20:
                    continue
                
                # Get predicted support levels for this ticker
                ticker_row = method_df[method_df['Ticker'] == ticker]
                if ticker_row.empty:
                    continue
                
                # Extract support levels (columns 1 onwards, excluding 'Ticker')
                support_cols = [col for col in method_df.columns if col.startswith('Support Level')]
                predicted_levels = []
                for col in support_cols:
                    level = ticker_row[col].iloc[0]
                    if pd.notna(level) and level > 0:
                        predicted_levels.append(float(level))
                
                if not predicted_levels:
                    continue
                
                # Calculate ground truth metrics
                ground_truth = self.calculate_ground_truth_metrics(prices)
                
                # Evaluate this method for this ticker
                scores = self.evaluate_support_method(predicted_levels, ground_truth, prices)
                
                # Accumulate scores
                for metric in ["support_accuracy", "price_proximity", "trend_stability", "overall_score"]:
                    method_scores[metric].append(scores[metric])
            
            # Calculate average scores for this method
            avg_scores = {
                "method": method_name,
                "tickers_evaluated": len(method_scores["overall_score"]),
                "avg_support_accuracy": np.mean(method_scores["support_accuracy"]) if method_scores["support_accuracy"] else 0,
                "avg_price_proximity": np.mean(method_scores["price_proximity"]) if method_scores["price_proximity"] else 0,
                "avg_trend_stability": np.mean(method_scores["trend_stability"]) if method_scores["trend_stability"] else 0,
                "avg_overall_score": np.mean(method_scores["overall_score"]) if method_scores["overall_score"] else 0,
                "std_overall_score": np.std(method_scores["overall_score"]) if method_scores["overall_score"] else 0
            }
            
            evaluation_results.append(avg_scores)
        
        # Create results DataFrame and sort by overall score
        results_df = pd.DataFrame(evaluation_results)
        results_df = results_df.sort_values("avg_overall_score", ascending=False)
        
        return results_df
    
    def generate_performance_report(self, results_df: pd.DataFrame) -> str:
        """Generate a detailed performance report"""
        report = []
        report.append("="*80)
        report.append("SUPPORT LEVEL DETECTION METHODS - PERFORMANCE EVALUATION")
        report.append("="*80)
        report.append("")
        
        report.append("METHODOLOGY:")
        report.append("Three fundamental financial metrics used as ground truth:")
        report.append("1. Support Accuracy: How often prices bounce off predicted support levels")
        report.append("2. Price Proximity: How close support levels are to actual price percentiles")
        report.append("3. Trend Stability: Consistency and spacing of predicted support levels")
        report.append("")
        
        report.append("RANKING (Best to Worst):")
        report.append("-" * 40)
        
        for i, row in results_df.iterrows():
            rank = len(results_df) - list(results_df.index).index(i)
            report.append(f"{rank}. {row['method'].upper()}")
            report.append(f"   Overall Score: {row['avg_overall_score']:.3f} (±{row['std_overall_score']:.3f})")
            report.append(f"   Support Accuracy: {row['avg_support_accuracy']:.3f}")
            report.append(f"   Price Proximity: {row['avg_price_proximity']:.3f}")
            report.append(f"   Trend Stability: {row['avg_trend_stability']:.3f}")
            report.append(f"   Tickers Evaluated: {row['tickers_evaluated']}")
            report.append("")
        
        # Performance insights
        report.append("PERFORMANCE INSIGHTS:")
        report.append("-" * 40)
        
        best_method = results_df.iloc[0]
        worst_method = results_df.iloc[-1]
        
        report.append(f"🏆 BEST PERFORMER: {best_method['method'].upper()}")
        report.append(f"   Excellence in: {self._get_strength_area(best_method)}")
        report.append("")
        
        report.append(f"📉 NEEDS IMPROVEMENT: {worst_method['method'].upper()}")
        report.append(f"   Weakness in: {self._get_weakness_area(worst_method)}")
        report.append("")
        
        # Method recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)
        top_3 = results_df.head(3)
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            report.append(f"{i}. Use {row['method'].upper()} for {self._get_use_case(row)}")
        
        report.append("")
        report.append("="*80)
        
        return "\n".join(report)
    
    def _get_strength_area(self, method_row) -> str:
        """Identify the strongest area for a method"""
        scores = {
            "Support Detection": method_row['avg_support_accuracy'],
            "Price Alignment": method_row['avg_price_proximity'], 
            "Trend Analysis": method_row['avg_trend_stability']
        }
        return max(scores, key=scores.get)
    
    def _get_weakness_area(self, method_row) -> str:
        """Identify the weakest area for a method"""
        scores = {
            "Support Detection": method_row['avg_support_accuracy'],
            "Price Alignment": method_row['avg_price_proximity'],
            "Trend Analysis": method_row['avg_trend_stability']
        }
        return min(scores, key=scores.get)
    
    def _get_use_case(self, method_row) -> str:
        """Suggest use case based on method strengths"""
        if method_row['avg_support_accuracy'] > 0.7:
            return "active trading and bounce predictions"
        elif method_row['avg_price_proximity'] > 0.7:
            return "price target setting and valuation"
        elif method_row['avg_trend_stability'] > 0.7:
            return "long-term trend analysis"
        else:
            return "general market analysis"
    
    def save_results(self, results_df: pd.DataFrame, report: str):
        """Save evaluation results and report"""
        # Save detailed results
        results_file = os.path.join(self.datasets_dir, "support_methods_evaluation.csv")
        results_df.to_csv(results_file, index=False)
        print(f"Detailed results saved to: {results_file}")
        
        # Save performance report
        report_file = os.path.join(self.datasets_dir, "performance_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"Performance report saved to: {report_file}")


def main():
    """
    Main function to run the comprehensive evaluation
    """
    print("Starting Support Level Detection Methods Evaluation...")
    print("=" * 60)
    
    try:
        # Initialize evaluator
        evaluator = SupportLevelEvaluator()
        
        # Run evaluation
        results_df = evaluator.evaluate_all_methods()
        
        # Generate report
        report = evaluator.generate_performance_report(results_df)
        
        # Display results
        print("\n" + report)
        
        # Save results
        evaluator.save_results(results_df, report)
        
        print("\n" + "=" * 60)
        print("Evaluation completed successfully!")
        print("Check the datasets folder for detailed results and report.")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()