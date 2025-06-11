from typing import Dict, List, Set

import numpy as np
import pandas as pd

from .market_analysis import MarketAnalyzer


class MetricsCalculator:
    """Calculates evaluation metrics for support level detection methods"""

    def __init__(self):
        self.market_analyzer = MarketAnalyzer()

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
                "support_accuracy": 0,
                "price_proximity": 0,
                "volume_confirmation": 0,
                "market_regime_sensitivity": 0,
                "support_hold_duration": 0,
                "false_breakout_rate": 0,
            }

        # Find actual support events
        actual_supports = []
        volume_events = []
        support_durations = []
        breakout_recoveries = []
        market_regimes = []

        # Generate synthetic volume data (since we might not have real volume)
        synthetic_volume = self.market_analyzer.generate_synthetic_volume(clean_prices)

        # Calculate market regimes (bull, bear, sideways)
        regime_labels = self.market_analyzer.calculate_market_regimes(clean_prices)

        for i in range(10, len(clean_prices) - 10):
            current_price = clean_prices.iloc[i]

            # Check if local minimum
            if current_price == min(clean_prices.iloc[i - 5 : i + 6]):
                # Check for bounce
                future_prices = clean_prices.iloc[i + 1 : i + 11]
                if len(future_prices) > 0:
                    max_future = future_prices.max()
                    bounce_pct = (max_future - current_price) / current_price

                    if bounce_pct > 0.01:  # At least 1% bounce
                        actual_supports.append(current_price)

                        # Volume confirmation: check if volume was above average during bounce
                        volume_score = (
                            self.market_analyzer.calculate_volume_confirmation(
                                synthetic_volume, i
                            )
                        )
                        volume_events.append(volume_score)

                        # Support duration
                        hold_duration = self.market_analyzer.calculate_support_duration(
                            clean_prices, i, current_price
                        )
                        support_durations.append(hold_duration)

                        # False breakout check
                        false_breakout = self.market_analyzer.check_false_breakout(
                            clean_prices, i, current_price, hold_duration
                        )
                        breakout_recoveries.append(false_breakout)

                        # Market regime at this point
                        if i < len(regime_labels):
                            market_regimes.append(regime_labels[i])

        # Price percentiles
        price_percentiles = [
            np.percentile(clean_prices, p) for p in [5, 10, 15, 20, 25, 30, 35]
        ]

        # Market regime analysis
        regime_performance = self.market_analyzer.analyze_regime_performance(
            market_regimes, actual_supports
        )

        metrics.update(
            {
                "actual_supports": actual_supports,
                "volume_events": volume_events,
                "support_durations": support_durations,
                "breakout_recoveries": breakout_recoveries,
                "market_regimes": market_regimes,
                "regime_performance": regime_performance,
                "price_percentiles": price_percentiles,
            }
        )

        return metrics

    def evaluate_support_method(
        self,
        predicted_levels: List[float],
        ground_truth: Dict[str, float],
        prices: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate support method against all six metrics"""
        scores = {}
        clean_prices = prices.dropna()

        if not predicted_levels or len(clean_prices) < 20:
            return {
                "support_accuracy": 0,
                "price_proximity": 0,
                "volume_confirmation": 0,
                "market_regime_sensitivity": 0,
                "support_hold_duration": 0,
                "false_breakout_rate": 0,
                "overall_score": 0,
            }

        # 1. Support Accuracy
        actual_supports = ground_truth.get("actual_supports", [])
        if actual_supports:
            accuracy_scores = []
            for actual_support in actual_supports:
                closest_distance = min(
                    [
                        abs(pred - actual_support) / actual_support
                        for pred in predicted_levels
                    ]
                )
                accuracy_score = max(0, 1 - (closest_distance / 0.05))
                accuracy_scores.append(accuracy_score)
            scores["support_accuracy"] = (
                np.mean(accuracy_scores) if accuracy_scores else 0
            )
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
            scores["price_proximity"] = (
                np.mean(proximity_scores) if proximity_scores else 0
            )
        else:
            scores["price_proximity"] = 0

        # 3. Volume Confirmation Score
        volume_events = ground_truth.get("volume_events", [])
        if volume_events and actual_supports:
            volume_scores = []
            for i, actual_support in enumerate(actual_supports):
                if i < len(volume_events):
                    closest_distance = min(
                        [
                            abs(pred - actual_support) / actual_support
                            for pred in predicted_levels
                        ]
                    )
                    if closest_distance < 0.05:  # Within 5% tolerance
                        # Higher volume confirmation score is better
                        volume_score = volume_events[i]
                        volume_scores.append(volume_score)
            scores["volume_confirmation"] = (
                np.mean(volume_scores) if volume_scores else 0
            )
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
            balance_score = (
                1.0
                - abs(bull_ratio - 0.33)
                - abs(bear_ratio - 0.33)
                - abs(sideways_ratio - 0.33)
            )
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
                    closest_distance = min(
                        [
                            abs(pred - actual_support) / actual_support
                            for pred in predicted_levels
                        ]
                    )
                    if closest_distance < 0.05:
                        duration_score = min(1, support_durations[i] / 30)
                        duration_scores.append(duration_score)
            scores["support_hold_duration"] = (
                np.mean(duration_scores) if duration_scores else 0
            )
        else:
            scores["support_hold_duration"] = 0

        # 6. False Breakout Rate
        breakout_recoveries = ground_truth.get("breakout_recoveries", [])
        if breakout_recoveries and actual_supports:
            recovery_scores = []
            for i, actual_support in enumerate(actual_supports):
                if i < len(breakout_recoveries):
                    closest_distance = min(
                        [
                            abs(pred - actual_support) / actual_support
                            for pred in predicted_levels
                        ]
                    )
                    if closest_distance < 0.05:
                        recovery_score = 1.0 if breakout_recoveries[i] else 0.8
                        recovery_scores.append(recovery_score)
            scores["false_breakout_rate"] = (
                np.mean(recovery_scores) if recovery_scores else 0
            )
        else:
            scores["false_breakout_rate"] = 0

        # Overall Score (updated weights)
        weights = {
            "support_accuracy": 0.25,
            "price_proximity": 0.20,
            "volume_confirmation": 0.20,
            "market_regime_sensitivity": 0.15,
            "support_hold_duration": 0.15,
            "false_breakout_rate": 0.05,
        }

        scores["overall_score"] = sum(
            scores[metric] * weights[metric] for metric in weights.keys()
        )

        return scores

    def evaluate_method(
        self,
        method_name: str,
        method_df: pd.DataFrame,
        common_tickers: Set[str],
        historical_prices: pd.DataFrame,
    ) -> Dict[str, any]:
        """Evaluate a specific method across all tickers"""
        method_scores = {
            "method": method_name,
            "support_accuracy": [],
            "price_proximity": [],
            "volume_confirmation": [],
            "market_regime_sensitivity": [],
            "support_hold_duration": [],
            "false_breakout_rate": [],
            "overall_score": [],
        }

        evaluated_count = 0
        for ticker in common_tickers:
            if ticker not in historical_prices.columns:
                continue

            prices = historical_prices[ticker].dropna()
            if len(prices) < 20:
                continue

            ticker_row = method_df[method_df["Ticker"] == ticker]
            if ticker_row.empty:
                continue

            support_cols = [
                col for col in method_df.columns if col.startswith("Support Level")
            ]
            predicted_levels = []
            for col in support_cols:
                level = ticker_row[col].iloc[0]
                if pd.notna(level) and level > 0:
                    predicted_levels.append(float(level))

            if not predicted_levels:
                continue

            ground_truth = self.calculate_ground_truth_metrics(prices)
            scores = self.evaluate_support_method(
                predicted_levels, ground_truth, prices
            )

            for metric in method_scores.keys():
                if metric != "method":
                    method_scores[metric].append(scores[metric])

            evaluated_count += 1

        print(f"   ✅ Evaluated {evaluated_count} tickers")

        # Calculate averages
        avg_scores = {
            "method": method_name,
            "tickers_evaluated": evaluated_count,
            "avg_support_accuracy": (
                np.mean(method_scores["support_accuracy"])
                if method_scores["support_accuracy"]
                else 0
            ),
            "avg_price_proximity": (
                np.mean(method_scores["price_proximity"])
                if method_scores["price_proximity"]
                else 0
            ),
            "avg_volume_confirmation": (
                np.mean(method_scores["volume_confirmation"])
                if method_scores["volume_confirmation"]
                else 0
            ),
            "avg_market_regime_sensitivity": (
                np.mean(method_scores["market_regime_sensitivity"])
                if method_scores["market_regime_sensitivity"]
                else 0
            ),
            "avg_support_hold_duration": (
                np.mean(method_scores["support_hold_duration"])
                if method_scores["support_hold_duration"]
                else 0
            ),
            "avg_false_breakout_rate": (
                np.mean(method_scores["false_breakout_rate"])
                if method_scores["false_breakout_rate"]
                else 0
            ),
            "avg_overall_score": (
                np.mean(method_scores["overall_score"])
                if method_scores["overall_score"]
                else 0
            ),
            "std_overall_score": (
                np.std(method_scores["overall_score"])
                if method_scores["overall_score"]
                else 0
            ),
        }

        return avg_scores
