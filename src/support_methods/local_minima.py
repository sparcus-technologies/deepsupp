import numpy as np

from src.support_methods.percentile import percentile_support


def local_minima_support(price_data, window=20, num_levels=7):
    """
    Identify meaningful support levels that are actually near the price data.
    """
    try:
        clean_data = price_data.dropna()
        local_mins = []
        prices = clean_data.values

        # Adjust window size if data is too small
        window = min(window, len(prices) // 4)
        if window < 2:
            return percentile_support(price_data, num_levels=num_levels)

        for i in range(window, len(prices) - window):
            if prices[i] == min(prices[i - window : i + window + 1]):
                local_mins.append(prices[i])

        # If we don't have enough local minima, add percentile-based levels
        if len(local_mins) < num_levels:
            percentiles = np.linspace(10, 40, num_levels - len(local_mins))
            for p in percentiles:
                percentile_value = np.percentile(prices, p)
                if percentile_value not in local_mins:
                    local_mins.append(percentile_value)

        # Group similar support levels
        grouped_levels = []
        for level in sorted(local_mins):
            found_group = False
            for i, group in enumerate(grouped_levels):
                if abs(level - np.mean(group)) / max(level, 1e-8) < 0.02:
                    grouped_levels[i].append(level)
                    found_group = True
                    break
            if not found_group:
                grouped_levels.append([level])

        support_levels = [np.mean(group) for group in grouped_levels]
        support_levels = sorted(support_levels)

        if len(support_levels) > num_levels:
            indices = np.linspace(0, len(support_levels) - 1, num_levels, dtype=int)
            support_levels = [support_levels[i] for i in indices]

        return [round(level, 2) for level in support_levels[:num_levels]]

    except Exception as e:
        print(f"Local minima failed, falling back to percentile method: {e}")
        return percentile_support(price_data, num_levels=num_levels)
