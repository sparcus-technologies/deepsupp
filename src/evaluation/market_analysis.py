from typing import Dict, List

import numpy as np
import pandas as pd


class MarketAnalyzer:
    """Analyzes market regimes and volume patterns"""

    @staticmethod
    def generate_synthetic_volume(prices: pd.Series) -> np.ndarray:
        """Generate realistic synthetic volume data based on price movements"""
        returns = prices.pct_change().fillna(0)

        # Higher volume on larger price moves
        base_volume = 100000
        volume_multiplier = 1 + (np.abs(returns) * 10)  # Higher volume on big moves

        # Add some random noise
        noise = np.random.lognormal(0, 0.3, len(prices))

        synthetic_volume = base_volume * volume_multiplier * noise
        return synthetic_volume

    @staticmethod
    def calculate_market_regimes(prices: pd.Series) -> List[str]:
        """Calculate market regimes: bull, bear, or sideways"""
        # Use 20-period moving average trend
        ma20 = prices.rolling(20).mean()

        regimes = []
        for i in range(len(prices)):
            if i < 20:
                regimes.append("sideways")
                continue

            current_price = prices.iloc[i]
            ma_value = ma20.iloc[i]

            # Recent trend (last 10 periods)
            recent_trend = (prices.iloc[i] - prices.iloc[max(0, i - 10)]) / prices.iloc[
                max(0, i - 10)
            ]

            if current_price > ma_value * 1.02 and recent_trend > 0.05:
                regimes.append("bull")
            elif current_price < ma_value * 0.98 and recent_trend < -0.05:
                regimes.append("bear")
            else:
                regimes.append("sideways")

        return regimes

    @staticmethod
    def calculate_volume_confirmation(volume: np.ndarray, support_idx: int) -> float:
        """Calculate volume confirmation score for support bounce"""
        if support_idx >= len(volume) - 5:
            return 0.5  # Default neutral score

        # Compare volume during bounce (next 5 periods) to average
        bounce_volume = np.mean(volume[support_idx : support_idx + 5])
        avg_volume = np.mean(volume[max(0, support_idx - 20) : support_idx])

        if avg_volume == 0:
            return 0.5

        volume_ratio = bounce_volume / avg_volume

        # Score: 1.0 if volume is 2x average, 0.5 if equal, 0.0 if very low
        volume_score = min(1.0, max(0.0, (volume_ratio - 0.5) / 1.5))
        return volume_score

    @staticmethod
    def analyze_regime_performance(
        regimes: List[str], supports: List[float]
    ) -> Dict[str, float]:
        """Analyze support performance across different market regimes"""
        regime_counts = {"bull": 0, "bear": 0, "sideways": 0}

        for regime in regimes:
            if regime in regime_counts:
                regime_counts[regime] += 1

        total_supports = len(supports)
        if total_supports == 0:
            return {
                "bull_ratio": 0.33,
                "bear_ratio": 0.33,
                "sideways_ratio": 0.33,
                "regime_diversity": 0.5,
            }

        # Calculate ratios
        bull_ratio = regime_counts["bull"] / total_supports
        bear_ratio = regime_counts["bear"] / total_supports
        sideways_ratio = regime_counts["sideways"] / total_supports

        # Diversity score (higher is better - means supports work across all regimes)
        regime_diversity = 1.0 - max(bull_ratio, bear_ratio, sideways_ratio)

        return {
            "bull_ratio": bull_ratio,
            "bear_ratio": bear_ratio,
            "sideways_ratio": sideways_ratio,
            "regime_diversity": regime_diversity,
        }

    @staticmethod
    def calculate_support_duration(
        prices: pd.Series, support_idx: int, support_level: float
    ) -> int:
        """Calculate how long support level holds"""
        duration = 0
        for i in range(support_idx + 1, min(len(prices), support_idx + 60)):
            if prices.iloc[i] < support_level * 0.97:  # 3% break
                break
            duration += 1
        return duration

    @staticmethod
    def check_false_breakout(
        prices: pd.Series, support_idx: int, support_level: float, duration: int
    ) -> bool:
        """Check for false breakout recovery"""
        end_idx = min(len(prices), support_idx + duration)
        for i in range(support_idx + 1, end_idx):
            if prices.iloc[i] < support_level * 0.97:
                recovery_end = min(len(prices), i + 5)
                for j in range(i + 1, recovery_end):
                    if prices.iloc[j] > support_level:
                        return True
        return False
