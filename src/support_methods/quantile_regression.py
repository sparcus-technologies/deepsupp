import warnings

import numpy as np
import statsmodels.api as sm

from src.support_methods.percentile import percentile_support


def quantile_regression_support(price_data, window=20, num_levels=7):
    """
    Identify support levels using Quantile Regression method with improved stability.
    """
    try:
        # Drop NaN values
        clean_data = price_data.dropna()
        prices = clean_data.values

        if len(prices) < 20:
            return percentile_support(price_data, num_levels=num_levels)

        # Prepare data for regression
        X = np.arange(len(prices)).reshape(-1, 1)
        X = sm.add_constant(X)

        # Define quantiles for support levels
        quantiles = np.linspace(0.05, 0.35, num_levels)
        support_levels = []

        for q in quantiles:
            try:
                # Fit quantile regression model with better parameters
                model = sm.QuantReg(prices, X)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = model.fit(
                        q=q,
                        max_iter=500,  # Reduce max iterations
                        p_tol=1e-3,  # Less strict tolerance
                    )

                support_level = result.params[0]
                if np.isfinite(support_level):
                    support_levels.append(support_level)

            except Exception as e:
                # If regression fails, use percentile
                percentile_val = np.percentile(prices, q * 100)
                support_levels.append(percentile_val)

        # Ensure we have enough levels
        while len(support_levels) < num_levels:
            percentile_val = np.percentile(prices, 5 + len(support_levels) * 5)
            support_levels.append(percentile_val)

        support_levels = sorted(support_levels)[:num_levels]
        return [round(level, 2) for level in support_levels]

    except Exception as e:
        print(f"Quantile regression failed, falling back to percentile method: {e}")
        return percentile_support(price_data, num_levels=num_levels)
