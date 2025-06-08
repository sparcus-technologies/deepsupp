import numpy as np


def percentile_support(price_data, window=None, num_levels=7):
    """Identify support levels based on percentiles"""
    try:
        lower_percentiles = np.linspace(5, 35, num_levels)
        return [
            round(float(np.percentile(price_data, p)), 2) for p in lower_percentiles
        ]
    except Exception as e:
        print(f"Percentile support failed: {e}")
        # Last resort - return evenly spaced levels around min price
        min_price = float(np.min(price_data))
        max_price = float(np.max(price_data))
        range_price = max_price - min_price
        return [
            round(min_price + i * range_price / (num_levels + 1), 2)
            for i in range(1, num_levels + 1)
        ]
