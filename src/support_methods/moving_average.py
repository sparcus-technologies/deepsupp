import numpy as np

from src.support_methods.percentile import percentile_support


def moving_average_support(price_data, window=20, num_levels=7):
    """
    Identify support levels based on moving averages
    """
    try:
        ma_periods = [20, 50, 100, 150, 200, 250, 300]
        levels = []

        for period in ma_periods[:num_levels]:
            if len(price_data) >= period:
                ma = price_data.rolling(window=period).mean().iloc[-1]
                if hasattr(ma, "iloc"):
                    levels.append(float(ma.iloc[0]))
                else:
                    levels.append(float(ma))

        # Fill remaining levels with percentiles
        while len(levels) < num_levels:
            percentile_val = np.percentile(price_data, 10 + len(levels) * 5)
            levels.append(float(percentile_val))

        return [round(level, 2) for level in levels[:num_levels]]

    except Exception as e:
        print(f"Moving average failed, falling back to percentile method: {e}")
        return percentile_support(price_data, num_levels=num_levels)
