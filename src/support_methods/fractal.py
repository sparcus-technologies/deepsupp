import numpy as np

from src.support_methods.percentile import percentile_support


def fractal_support(price_data, window=2, num_levels=7):
    """
    Identify support levels based on Bill Williams' fractal indicator concept.
    """
    try:
        clean_data = price_data.dropna()
        prices = clean_data.values
        fractals = []

        # Adjust window if data is too small
        window = min(window, len(prices) // 6)
        if window < 1:
            return percentile_support(price_data, num_levels=num_levels)

        for i in range(window * 2, len(prices) - window * 2):
            left_side = all(prices[i] < prices[i - j] for j in range(1, window + 1))
            right_side = all(prices[i] < prices[i + j] for j in range(1, window + 1))

            if left_side and right_side:
                fractals.append(prices[i])

        # Fill with percentiles if needed
        if len(fractals) < num_levels:
            percentiles = np.linspace(5, 35, num_levels - len(fractals))
            for p in percentiles:
                percentile_value = np.percentile(prices, p)
                if percentile_value not in fractals:
                    fractals.append(percentile_value)

        # Group similar levels
        grouped_levels = []
        for level in sorted(fractals):
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
        print(f"Fractal failed, falling back to percentile method: {e}")
        return percentile_support(price_data, num_levels=num_levels)
