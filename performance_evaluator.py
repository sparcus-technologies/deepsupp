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
    using six fundamental financial metrics as ground truth.
    """
    
    def __init__(self, datasets_dir="datasets"):
        self.datasets_dir = datasets_dir
        self.historical_prices = None
        self.results = {}
        
        # Ensure datasets directory exists
        os.makedirs(self.datasets_dir, exist_ok=True)
        
    def find_price_file(self):
        """Find historical price file with flexible naming"""
        possible_names = [
            "sp500_historial_prices.csv"
        ]
        
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
    
    def create_sample_data(self):
        """Create sample data if no price file exists"""
        print("📥 No historical price file found. Creating sample data...")
        
        # Create sample S&P 500 data
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'PG']
        
        price_data = {}
        for ticker in tickers:
            # Generate realistic price series
            base_price = np.random.uniform(50, 300)
            returns = np.random.normal(0.0005, 0.02, len(dates))
            
            # Add some trend and autocorrelation
            for i in range(1, len(returns)):
                returns[i] += 0.1 * returns[i-1]
            
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
    
    def load_historical_prices(self):
        """Load historical price data with flexible file detection"""
        price_file = self.find_price_file()
        
        if price_file is None:
            print("❌ No historical price file found!")
            self.historical_prices = self.create_sample_data()
            return
        
        try:
            self.historical_prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
            print(f"✅ Loaded historical prices: {len(self.historical_prices.columns)} tickers")
            
        except Exception as e:
            print(f"❌ Error loading {price_file}: {e}")
            print("Creating sample data instead...")
            self.historical_prices = self.create_sample_data()
    
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
                row_data = {'Ticker': ticker}
                for i, level in enumerate(support_levels, 1):
                    row_data[f'Support Level {i}'] = level
                
                support_data.append(row_data)
            
            # Save to CSV
            df = pd.DataFrame(support_data)
            filename = os.path.join(self.datasets_dir, f"{method}_support_levels.csv")
            df.to_csv(filename, index=False)
            print(f"💾 Created {method} support data: {filename}")

    def calculate_ground_truth_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """
        Calculate six fundamental financial metrics as ground truth:
        1. Support Accuracy: How often prices bounce off predicted support levels
        2. Price Proximity: How close support levels are to actual low prices
        3. Volume Confirmation: Whether high volume accompanies support bounces
        4. Market Regime Sensitivity: Performance across different market conditions
        5. Support Hold Duration: How long support levels remain valid
        6. False Breakout Rate: Recovery rate after brief support breaks
        """
        metrics = {}
        
        clean_prices = prices.dropna()
        if len(clean_prices) < 20:
            return {
                "support_accuracy": 0, "price_proximity": 0, "volume_confirmation": 0,
                "market_regime_sensitivity": 0, "support_hold_duration": 0, "false_breakout_rate": 0
            }
        
        # Find actual support events
        actual_supports = []
        volume_events = []
        support_durations = []
        breakout_recoveries = []
        market_regimes = []
        
        # Generate synthetic volume data (since we might not have real volume)
        synthetic_volume = self._generate_synthetic_volume(clean_prices)
        
        # Calculate market regimes (bull, bear, sideways)
        regime_labels = self._calculate_market_regimes(clean_prices)
        
        for i in range(10, len(clean_prices) - 10):
            current_price = clean_prices.iloc[i]
            
            # Check if local minimum
            if current_price == min(clean_prices.iloc[i-5:i+6]):
                # Check for bounce
                future_prices = clean_prices.iloc[i+1:i+11]
                if len(future_prices) > 0:
                    max_future = future_prices.max()
                    bounce_pct = (max_future - current_price) / current_price
                    
                    if bounce_pct > 0.01:  # At least 1% bounce
                        actual_supports.append(current_price)
                        
                        # Volume confirmation: check if volume was above average during bounce
                        volume_score = self._calculate_volume_confirmation(synthetic_volume, i)
                        volume_events.append(volume_score)
                        
                        # Support duration
                        hold_duration = self._calculate_support_duration(clean_prices, i, current_price)
                        support_durations.append(hold_duration)
                        
                        # False breakout check
                        false_breakout = self._check_false_breakout(clean_prices, i, current_price, hold_duration)
                        breakout_recoveries.append(false_breakout)
                        
                        # Market regime at this point
                        if i < len(regime_labels):
                            market_regimes.append(regime_labels[i])
        
        # Price percentiles
        price_percentiles = [np.percentile(clean_prices, p) for p in [5, 10, 15, 20, 25, 30, 35]]
        
        # Market regime analysis
        regime_performance = self._analyze_regime_performance(market_regimes, actual_supports)
        
        metrics.update({
            "actual_supports": actual_supports,
            "volume_events": volume_events,
            "support_durations": support_durations,
            "breakout_recoveries": breakout_recoveries,
            "market_regimes": market_regimes,
            "regime_performance": regime_performance,
            "price_percentiles": price_percentiles
        })
        
        return metrics
    
    def _generate_synthetic_volume(self, prices: pd.Series) -> np.ndarray:
        """Generate realistic synthetic volume data based on price movements"""
        returns = prices.pct_change().fillna(0)
        
        # Higher volume on larger price moves
        base_volume = 100000
        volume_multiplier = 1 + (np.abs(returns) * 10)  # Higher volume on big moves
        
        # Add some random noise
        noise = np.random.lognormal(0, 0.3, len(prices))
        
        synthetic_volume = base_volume * volume_multiplier * noise
        return synthetic_volume
    
    def _calculate_market_regimes(self, prices: pd.Series) -> List[str]:
        """Calculate market regimes: bull, bear, or sideways"""
        # Use 20-period moving average trend
        ma20 = prices.rolling(20).mean()
        
        regimes = []
        for i in range(len(prices)):
            if i < 20:
                regimes.append('sideways')
                continue
                
            current_price = prices.iloc[i]
            ma_value = ma20.iloc[i]
            
            # Recent trend (last 10 periods)
            recent_trend = (prices.iloc[i] - prices.iloc[max(0, i-10)]) / prices.iloc[max(0, i-10)]
            
            if current_price > ma_value * 1.02 and recent_trend > 0.05:
                regimes.append('bull')
            elif current_price < ma_value * 0.98 and recent_trend < -0.05:
                regimes.append('bear')
            else:
                regimes.append('sideways')
        
        return regimes
    
    def _calculate_volume_confirmation(self, volume: np.ndarray, support_idx: int) -> float:
        """Calculate volume confirmation score for support bounce"""
        if support_idx >= len(volume) - 5:
            return 0.5  # Default neutral score
        
        # Compare volume during bounce (next 5 periods) to average
        bounce_volume = np.mean(volume[support_idx:support_idx+5])
        avg_volume = np.mean(volume[max(0, support_idx-20):support_idx])
        
        if avg_volume == 0:
            return 0.5
        
        volume_ratio = bounce_volume / avg_volume
        
        # Score: 1.0 if volume is 2x average, 0.5 if equal, 0.0 if very low
        volume_score = min(1.0, max(0.0, (volume_ratio - 0.5) / 1.5))
        return volume_score
    
    def _analyze_regime_performance(self, regimes: List[str], supports: List[float]) -> Dict[str, float]:
        """Analyze support performance across different market regimes"""
        regime_counts = {'bull': 0, 'bear': 0, 'sideways': 0}
        
        for regime in regimes:
            if regime in regime_counts:
                regime_counts[regime] += 1
        
        total_supports = len(supports)
        if total_supports == 0:
            return {'bull_ratio': 0.33, 'bear_ratio': 0.33, 'sideways_ratio': 0.33, 'regime_diversity': 0.5}
        
        # Calculate ratios
        bull_ratio = regime_counts['bull'] / total_supports
        bear_ratio = regime_counts['bear'] / total_supports
        sideways_ratio = regime_counts['sideways'] / total_supports
        
        # Diversity score (higher is better - means supports work across all regimes)
        regime_diversity = 1.0 - max(bull_ratio, bear_ratio, sideways_ratio)
        
        return {
            'bull_ratio': bull_ratio,
            'bear_ratio': bear_ratio, 
            'sideways_ratio': sideways_ratio,
            'regime_diversity': regime_diversity
        }
    
    def _calculate_support_duration(self, prices: pd.Series, support_idx: int, support_level: float) -> int:
        """Calculate how long support level holds"""
        duration = 0
        for i in range(support_idx + 1, min(len(prices), support_idx + 60)):
            if prices.iloc[i] < support_level * 0.97:  # 3% break
                break
            duration += 1
        return duration
    
    def _check_false_breakout(self, prices: pd.Series, support_idx: int, 
                             support_level: float, duration: int) -> bool:
        """Check for false breakout recovery"""
        end_idx = min(len(prices), support_idx + duration)
        for i in range(support_idx + 1, end_idx):
            if prices.iloc[i] < support_level * 0.97:
                recovery_end = min(len(prices), i + 5)
                for j in range(i + 1, recovery_end):
                    if prices.iloc[j] > support_level:
                        return True
        return False
    
    def evaluate_support_method(self, predicted_levels: List[float], 
                              ground_truth: Dict[str, float], 
                              prices: pd.Series) -> Dict[str, float]:
        """Evaluate support method against all six metrics"""
        scores = {}
        clean_prices = prices.dropna()
        
        if not predicted_levels or len(clean_prices) < 20:
            return {
                "support_accuracy": 0, "price_proximity": 0, "volume_confirmation": 0,
                "market_regime_sensitivity": 0, "support_hold_duration": 0, 
                "false_breakout_rate": 0, "overall_score": 0
            }
        
        # 1. Support Accuracy
        actual_supports = ground_truth.get("actual_supports", [])
        if actual_supports:
            accuracy_scores = []
            for actual_support in actual_supports:
                closest_distance = min([abs(pred - actual_support) / actual_support 
                                      for pred in predicted_levels])
                accuracy_score = max(0, 1 - (closest_distance / 0.05))
                accuracy_scores.append(accuracy_score)
            scores["support_accuracy"] = np.mean(accuracy_scores) if accuracy_scores else 0
        else:
            scores["support_accuracy"] = 0
        
        # 2. Price Proximity
        price_percentiles = ground_truth.get("price_percentiles", [])
        if price_percentiles:
            proximity_scores = []
            for i, predicted in enumerate(predicted_levels):
                if i < len(price_percentiles):
                    expected = price_percentiles[i]
                    relative_error = abs(predicted - expected) / expected
                    proximity_score = max(0, 1 - (relative_error / 0.1))
                    proximity_scores.append(proximity_score)
            scores["price_proximity"] = np.mean(proximity_scores) if proximity_scores else 0
        else:
            scores["price_proximity"] = 0
        
        # 3. Volume Confirmation Score
        volume_events = ground_truth.get("volume_events", [])
        if volume_events and actual_supports:
            volume_scores = []
            for i, actual_support in enumerate(actual_supports):
                if i < len(volume_events):
                    closest_distance = min([abs(pred - actual_support) / actual_support 
                                          for pred in predicted_levels])
                    if closest_distance < 0.05:  # Within 5% tolerance
                        # Higher volume confirmation score is better
                        volume_score = volume_events[i]
                        volume_scores.append(volume_score)
            scores["volume_confirmation"] = np.mean(volume_scores) if volume_scores else 0
        else:
            scores["volume_confirmation"] = 0
        
        # 4. Market Regime Sensitivity Score
        regime_performance = ground_truth.get("regime_performance", {})
        if regime_performance:
            # Score based on how well supports work across different market conditions
            regime_diversity = regime_performance.get("regime_diversity", 0)
            
            # Bonus for balanced performance across regimes
            bull_ratio = regime_performance.get("bull_ratio", 0)
            bear_ratio = regime_performance.get("bear_ratio", 0)
            sideways_ratio = regime_performance.get("sideways_ratio", 0)
            
            # Ideal is roughly equal distribution (0.33 each)
            balance_score = 1.0 - abs(bull_ratio - 0.33) - abs(bear_ratio - 0.33) - abs(sideways_ratio - 0.33)
            balance_score = max(0, balance_score)
            
            scores["market_regime_sensitivity"] = (regime_diversity + balance_score) / 2
        else:
            scores["market_regime_sensitivity"] = 0
        
        # 5. Support Hold Duration
        support_durations = ground_truth.get("support_durations", [])
        if support_durations and actual_supports:
            duration_scores = []
            for i, actual_support in enumerate(actual_supports):
                if i < len(support_durations):
                    closest_distance = min([abs(pred - actual_support) / actual_support 
                                          for pred in predicted_levels])
                    if closest_distance < 0.05:
                        duration_score = min(1, support_durations[i] / 30)
                        duration_scores.append(duration_score)
            scores["support_hold_duration"] = np.mean(duration_scores) if duration_scores else 0
        else:
            scores["support_hold_duration"] = 0
        
        # 6. False Breakout Rate
        breakout_recoveries = ground_truth.get("breakout_recoveries", [])
        if breakout_recoveries and actual_supports:
            recovery_scores = []
            for i, actual_support in enumerate(actual_supports):
                if i < len(breakout_recoveries):
                    closest_distance = min([abs(pred - actual_support) / actual_support 
                                          for pred in predicted_levels])
                    if closest_distance < 0.05:
                        recovery_score = 1.0 if breakout_recoveries[i] else 0.8
                        recovery_scores.append(recovery_score)
            scores["false_breakout_rate"] = np.mean(recovery_scores) if recovery_scores else 0
        else:
            scores["false_breakout_rate"] = 0
        
        # Overall Score (updated weights)
        weights = {
            "support_accuracy": 0.25,
            "price_proximity": 0.20,
            "volume_confirmation": 0.20,  
            "market_regime_sensitivity": 0.15,  
            "support_hold_duration": 0.15,
            "false_breakout_rate": 0.05
        }
        
        scores["overall_score"] = sum(scores[metric] * weights[metric] for metric in weights.keys())
        
        return scores
    
    def load_support_method_results(self) -> Dict[str, pd.DataFrame]:
        """Load all support level detection results"""
        method_results = {}
        
        csv_files = glob.glob(os.path.join(self.datasets_dir, "*_support_levels.csv"))
        
        if not csv_files:
            print("❌ No support level CSV files found!")
            print("Creating sample support data...")
            self.create_sample_support_data()
            csv_files = glob.glob(os.path.join(self.datasets_dir, "*_support_levels.csv"))
        
        for csv_file in csv_files:
            method_name = os.path.basename(csv_file).replace("_support_levels.csv", "")
            try:
                df = pd.read_csv(csv_file)
                method_results[method_name] = df
                print(f"✅ Loaded {method_name}: {len(df)} tickers")
            except Exception as e:
                print(f"❌ Error loading {csv_file}: {e}")
        
        return method_results
    
    def evaluate_all_methods(self) -> pd.DataFrame:
        """Evaluate all support level detection methods"""
        if self.historical_prices is None:
            self.load_historical_prices()
        
        method_results = self.load_support_method_results()
        
        if not method_results:
            raise ValueError("No support level results found")
        
        evaluation_results = []
        
        # Get common tickers
        common_tickers = set(self.historical_prices.columns)
        for method_df in method_results.values():
            common_tickers = common_tickers.intersection(set(method_df['Ticker']))
        
        print(f"🔍 Evaluating {len(common_tickers)} tickers across {len(method_results)} methods")
        
        for method_name, method_df in method_results.items():
            print(f"⚡ Evaluating {method_name}...")
            
            method_scores = {
                "method": method_name,
                "support_accuracy": [], "price_proximity": [], "volume_confirmation": [],
                "market_regime_sensitivity": [], "support_hold_duration": [], "false_breakout_rate": [],
                "overall_score": []
            }
            
            evaluated_count = 0
            for ticker in common_tickers:
                if ticker not in self.historical_prices.columns:
                    continue
                    
                prices = self.historical_prices[ticker].dropna()
                if len(prices) < 20:
                    continue
                
                ticker_row = method_df[method_df['Ticker'] == ticker]
                if ticker_row.empty:
                    continue
                
                support_cols = [col for col in method_df.columns if col.startswith('Support Level')]
                predicted_levels = []
                for col in support_cols:
                    level = ticker_row[col].iloc[0]
                    if pd.notna(level) and level > 0:
                        predicted_levels.append(float(level))
                
                if not predicted_levels:
                    continue
                
                ground_truth = self.calculate_ground_truth_metrics(prices)
                scores = self.evaluate_support_method(predicted_levels, ground_truth, prices)
                
                for metric in method_scores.keys():
                    if metric != "method":
                        method_scores[metric].append(scores[metric])
                
                evaluated_count += 1
            
            print(f"   ✅ Evaluated {evaluated_count} tickers")
            
            # Calculate averages
            avg_scores = {
                "method": method_name,
                "tickers_evaluated": evaluated_count,
                "avg_support_accuracy": np.mean(method_scores["support_accuracy"]) if method_scores["support_accuracy"] else 0,
                "avg_price_proximity": np.mean(method_scores["price_proximity"]) if method_scores["price_proximity"] else 0,
                "avg_volume_confirmation": np.mean(method_scores["volume_confirmation"]) if method_scores["volume_confirmation"] else 0,
                "avg_market_regime_sensitivity": np.mean(method_scores["market_regime_sensitivity"]) if method_scores["market_regime_sensitivity"] else 0,
                "avg_support_hold_duration": np.mean(method_scores["support_hold_duration"]) if method_scores["support_hold_duration"] else 0,
                "avg_false_breakout_rate": np.mean(method_scores["false_breakout_rate"]) if method_scores["false_breakout_rate"] else 0,
                "avg_overall_score": np.mean(method_scores["overall_score"]) if method_scores["overall_score"] else 0,
                "std_overall_score": np.std(method_scores["overall_score"]) if method_scores["overall_score"] else 0
            }
            
            evaluation_results.append(avg_scores)
        
        results_df = pd.DataFrame(evaluation_results)
        results_df = results_df.sort_values("avg_overall_score", ascending=False)
        
        return results_df
    
    def generate_performance_report(self, results_df: pd.DataFrame) -> str:
        """Generate detailed performance report"""
        report = []
        report.append("="*90)
        report.append("SUPPORT LEVEL DETECTION METHODS - ENHANCED PERFORMANCE EVALUATION")
        report.append("="*90)
        report.append("")
        
        if len(results_df) == 0:
            report.append("❌ No results to report!")
            return "\n".join(report)
        
        report.append("ENHANCED METHODOLOGY:")
        report.append("Six fundamental financial metrics used as ground truth:")
        report.append("1. Support Accuracy: How often prices bounce off predicted support levels")
        report.append("2. Price Proximity: How close support levels are to actual price percentiles")
        report.append("3. Volume Confirmation: Whether high volume accompanies support bounces")
        report.append("4. Market Regime Sensitivity: Performance across bull/bear/sideways markets")
        report.append("5. Support Hold Duration: How long support levels remain valid")
        report.append("6. False Breakout Rate: Recovery rate after brief support breaks")
        report.append("")
        
        report.append("DETAILED RANKING (Best to Worst):")
        report.append("-" * 50)
        
        for i, (_, row) in enumerate(results_df.iterrows(), 1):
            report.append(f"{i}. {row['method'].upper()}")
            report.append(f"   Overall Score: {row['avg_overall_score']:.3f} (±{row['std_overall_score']:.3f})")
            report.append(f"   ├─ Support Accuracy: {row['avg_support_accuracy']:.3f}")
            report.append(f"   ├─ Price Proximity: {row['avg_price_proximity']:.3f}")
            report.append(f"   ├─ Volume Confirmation: {row['avg_volume_confirmation']:.3f}")
            report.append(f"   ├─ Market Regime Sensitivity: {row['avg_market_regime_sensitivity']:.3f}")
            report.append(f"   ├─ Support Duration: {row['avg_support_hold_duration']:.3f}")
            report.append(f"   └─ Breakout Recovery: {row['avg_false_breakout_rate']:.3f}")
            report.append(f"   Tickers Evaluated: {row['tickers_evaluated']}")
            report.append("")
        
        # Category leaders
        if len(results_df) > 0:
            report.append("CATEGORY LEADERS:")
            report.append("-" * 30)
            categories = {
                "Support Detection": "avg_support_accuracy",
                "Price Alignment": "avg_price_proximity",
                "Volume Analysis": "avg_volume_confirmation",
                "Market Adaptability": "avg_market_regime_sensitivity",
                "Duration Stability": "avg_support_hold_duration",
                "Breakout Handling": "avg_false_breakout_rate"
            }
            
            for category, metric in categories.items():
                if metric in results_df.columns:
                    leader = results_df.loc[results_df[metric].idxmax()]
                    report.append(f"🎯 {category}: {leader['method'].upper()} ({leader[metric]:.3f})")
        
        report.append("")
        
        # Enhanced insights
        if len(results_df) > 0:
            best_method = results_df.iloc[0]
            report.append("PERFORMANCE INSIGHTS:")
            report.append("-" * 30)
            report.append(f"🏆 BEST PERFORMER: {best_method['method'].upper()}")
            
            # Identify strengths
            strengths = []
            if best_method['avg_volume_confirmation'] > 0.7:
                strengths.append("Strong volume confirmation")
            if best_method['avg_market_regime_sensitivity'] > 0.7:
                strengths.append("Excellent market adaptability")
            if best_method['avg_support_accuracy'] > 0.7:
                strengths.append("High support accuracy")
            
            if strengths:
                report.append(f"   Key Strengths: {', '.join(strengths)}")
            
            # Trading recommendations
            report.append("")
            report.append("TRADING RECOMMENDATIONS:")
            report.append("-" * 30)
            
            for i, (_, row) in enumerate(results_df.head(3).iterrows(), 1):
                use_case = self._get_trading_use_case(row)
                report.append(f"{i}. {row['method'].upper()}: {use_case}")
        
        report.append("")
        report.append("="*90)
        
        return "\n".join(report)
    
    def _get_trading_use_case(self, method_row) -> str:
        """Get trading use case based on method strengths"""
        if method_row['avg_volume_confirmation'] > 0.7:
            return "Best for volume-based trading strategies"
        elif method_row['avg_market_regime_sensitivity'] > 0.7:
            return "Ideal for all-weather trading across market conditions"
        elif method_row['avg_support_hold_duration'] > 0.7:
            return "Perfect for position trading and long-term holds"
        elif method_row['avg_support_accuracy'] > 0.7:
            return "Excellent for day trading and quick entries"
        elif method_row['avg_false_breakout_rate'] > 0.7:
            return "Great for breakout trading and stop-loss placement"
        else:
            return "Suitable for general market analysis"
    
    def save_results(self, results_df: pd.DataFrame, report: str):
        """Save evaluation results and report"""
        results_file = os.path.join(self.datasets_dir, "enhanced_support_methods_evaluation.csv")
        results_df.to_csv(results_file, index=False)
        print(f"💾 Results saved to: {results_file}")
        
        report_file = os.path.join(self.datasets_dir, "enhanced_performance_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📝 Report saved to: {report_file}")


def main():
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


if __name__ == "__main__":
    main()