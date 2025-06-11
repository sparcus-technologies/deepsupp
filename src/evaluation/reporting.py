import pandas as pd


class ReportGenerator:
    """Generates performance reports for support level detection methods"""

    def generate_report(self, results_df: pd.DataFrame) -> str:
        """Generate detailed performance report"""
        report = []
        report.append("=" * 90)
        report.append(
            "SUPPORT LEVEL DETECTION METHODS - ENHANCED PERFORMANCE EVALUATION"
        )
        report.append("=" * 90)
        report.append("")

        if len(results_df) == 0:
            report.append("❌ No results to report!")
            return "\n".join(report)

        report.append("ENHANCED METHODOLOGY:")
        report.append("Six fundamental financial metrics used as ground truth:")
        report.append(
            "1. Support Accuracy: How often prices bounce off predicted support levels"
        )
        report.append(
            "2. Price Proximity: How close support levels are to actual price percentiles"
        )
        report.append(
            "3. Volume Confirmation: Whether high volume accompanies support bounces"
        )
        report.append(
            "4. Market Regime Sensitivity: Performance across bull/bear/sideways markets"
        )
        report.append("5. Support Hold Duration: How long support levels remain valid")
        report.append(
            "6. False Breakout Rate: Recovery rate after brief support breaks"
        )
        report.append("")

        report.append("DETAILED RANKING (Best to Worst):")
        report.append("-" * 50)

        for i, (_, row) in enumerate(results_df.iterrows(), 1):
            report.append(f"{i}. {row['method'].upper()}")
            report.append(
                f"   Overall Score: {row['avg_overall_score']:.3f} (±{row['std_overall_score']:.3f})"
            )
            report.append(f"   ├─ Support Accuracy: {row['avg_support_accuracy']:.3f}")
            report.append(f"   ├─ Price Proximity: {row['avg_price_proximity']:.3f}")
            report.append(
                f"   ├─ Volume Confirmation: {row['avg_volume_confirmation']:.3f}"
            )
            report.append(
                f"   ├─ Market Regime Sensitivity: {row['avg_market_regime_sensitivity']:.3f}"
            )
            report.append(
                f"   ├─ Support Duration: {row['avg_support_hold_duration']:.3f}"
            )
            report.append(
                f"   └─ Breakout Recovery: {row['avg_false_breakout_rate']:.3f}"
            )
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
                "Breakout Handling": "avg_false_breakout_rate",
            }

            for category, metric in categories.items():
                if metric in results_df.columns:
                    leader = results_df.loc[results_df[metric].idxmax()]
                    report.append(
                        f"🎯 {category}: {leader['method'].upper()} ({leader[metric]:.3f})"
                    )

        report.append("")

        # Enhanced insights
        if len(results_df) > 0:
            best_method = results_df.iloc[0]
            report.append("PERFORMANCE INSIGHTS:")
            report.append("-" * 30)
            report.append(f"🏆 BEST PERFORMER: {best_method['method'].upper()}")

            # Identify strengths
            strengths = []
            if best_method["avg_volume_confirmation"] > 0.7:
                strengths.append("Strong volume confirmation")
            if best_method["avg_market_regime_sensitivity"] > 0.7:
                strengths.append("Excellent market adaptability")
            if best_method["avg_support_accuracy"] > 0.7:
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
        report.append("=" * 90)

        return "\n".join(report)

    def _get_trading_use_case(self, method_row) -> str:
        """Get trading use case based on method strengths"""
        if method_row["avg_volume_confirmation"] > 0.7:
            return "Best for volume-based trading strategies"
        elif method_row["avg_market_regime_sensitivity"] > 0.7:
            return "Ideal for all-weather trading across market conditions"
        elif method_row["avg_support_hold_duration"] > 0.7:
            return "Perfect for position trading and long-term holds"
        elif method_row["avg_support_accuracy"] > 0.7:
            return "Excellent for day trading and quick entries"
        elif method_row["avg_false_breakout_rate"] > 0.7:
            return "Great for breakout trading and stop-loss placement"
        else:
            return "Suitable for general market analysis"
