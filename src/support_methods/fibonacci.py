from src.support_methods.percentile import percentile_support


def fibonacci_support(price_data, window=20, num_levels=7):
    """
    Identify support levels using Fibonacci retracement.
    """
    try:
        highest_high = float(price_data.max())
        lowest_low = float(price_data.min())
        price_range = highest_high - lowest_low

        if price_range == 0:
            return [round(lowest_low, 2)] * num_levels

        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 0.886, 0.925]
        fib_levels = [highest_high - (ratio * price_range) for ratio in fib_ratios]
        support_levels = fib_levels[:num_levels]

        return [round(level, 2) for level in support_levels]

    except Exception as e:
        print(f"Fibonacci failed, falling back to percentile method: {e}")
        return percentile_support(price_data, num_levels=num_levels)
