import os
import warnings

import numpy as np
import statsmodels.api as sm
from hmmlearn import hmm

from src.support_methods.percentile import percentile_support


def hmm_support_levels(price_data, window=20, num_levels=7):
    """
    Identify support levels using Hidden Markov Models.
    """
    try:
        # Drop NaN values
        clean_data = price_data.dropna()
        prices = clean_data.values

        if len(prices) < 50:  # Need sufficient data for HMM
            return percentile_support(price_data, num_levels=num_levels)

        # Prepare data for HMM - use price returns with better preprocessing
        returns = np.diff(prices) / prices[:-1]

        # Remove outliers and infinite values
        returns = returns[np.isfinite(returns)]
        if len(returns) == 0:
            return percentile_support(price_data, num_levels=num_levels)

        # Normalize returns to improve numerical stability
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        returns = returns.reshape(-1, 1)

        # Use fewer states for better stability
        n_states = min(3, max(2, num_levels // 2))

        # Fit HMM model with better parameters
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="spherical",  # More stable than "full"
            n_iter=100,  # Reduce iterations
            random_state=42,
            tol=1e-2,  # Less strict tolerance
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(returns)

        # Predict hidden states
        hidden_states = model.predict(returns)

        # Calculate support levels from states
        state_prices = {}
        for state in range(n_states):
            state_indices = np.where(hidden_states == state)[0]
            if len(state_indices) > 0:
                state_price_points = prices[state_indices + 1]
                state_prices[state] = np.percentile(state_price_points, 20)

        support_levels = sorted(state_prices.values())

        # Fill remaining levels with percentiles
        if len(support_levels) < num_levels:
            percentiles = np.linspace(5, 35, num_levels - len(support_levels))
            for p in percentiles:
                support_levels.append(np.percentile(prices, p))

        support_levels = sorted(support_levels)[:num_levels]
        return [round(level, 2) for level in support_levels]

    except Exception as e:
        print(f"HMM failed, falling back to percentile method: {e}")
        return percentile_support(price_data, num_levels=num_levels)
