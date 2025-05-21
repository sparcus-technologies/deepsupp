import numpy as np
import statsmodels.api as sm
from hmmlearn import hmm


def hmm_support_levels(price_data, window=20, num_levels=7):
    """
    Identify support levels using Hidden Markov Models.

    This method uses HMMs to identify latent regimes in price data, and extracts
    support levels based on the lower bounds of these regimes.

    Args:
        price_data: Series of price data
        window: Window size for local analysis
        num_levels: Number of support levels to return

    Returns:
        List of support level prices identified by HMM
    """
    # Drop NaN values
    clean_data = price_data.dropna()
    prices = clean_data.values

    # Prepare data for HMM - use price returns
    returns = np.diff(prices) / prices[:-1]
    returns = returns.reshape(-1, 1)  # Reshape for HMM

    # Define number of hidden states
    n_states = min(5, num_levels + 2)  # Use more states than needed levels

    # Fit HMM model
    model = hmm.GaussianHMM(
        n_components=n_states, covariance_type="full", n_iter=1000, random_state=42
    )
    model.fit(returns)

    # Predict hidden states
    hidden_states = model.predict(returns)

    # Calculate mean price level for each state
    state_prices = {}
    for state in range(n_states):
        state_indices = np.where(hidden_states == state)[0]
        if len(state_indices) > 0:
            # Add +1 to indices to account for the diff operation
            state_price_points = prices[state_indices + 1]
            # Use lower percentile of each state's price distribution as a support level
            state_prices[state] = np.percentile(state_price_points, 10)

    # Sort the support levels and select the lowest ones
    support_levels = sorted(state_prices.values())[:num_levels]

    # If we don't have enough levels, add some based on quantiles
    if len(support_levels) < num_levels:
        quantile_levels = np.percentile(
            prices, [5, 10, 15, 20][: num_levels - len(support_levels)]
        )
        support_levels.extend(quantile_levels)
        support_levels = sorted(support_levels)[:num_levels]

    return [round(level, 2) for level in support_levels]


def quantile_regression_support(price_data, window=20, num_levels=7):
    """
    Identify support levels using Quantile Regression method.

    This method identifies support levels by fitting quantile regression models
    at low quantiles (0.05, 0.10, 0.15) to capture price floors.

    Args:
        price_data: Series of price data
        window: Window size for local analysis
        num_levels: Number of support levels to return

    Returns:
        List of support level prices identified by quantile regression
    """

    # Drop NaN values
    clean_data = price_data.dropna()
    prices = clean_data.values

    # Prepare data for regression
    X = np.arange(len(prices)).reshape(-1, 1)
    X = sm.add_constant(X)  # Add constant term

    # Define quantiles for support levels (lower quantiles)
    quantiles = [0.05, 0.10, 0.15][:num_levels]
    support_levels = []

    for q in quantiles:
        # Fit quantile regression model
        model = sm.QuantReg(prices, X)
        result = model.fit(q=q)

        # Extract the constant term as the support level estimate
        # (intercept of the quantile regression line)
        support_level = result.params[0]
        support_levels.append(support_level)

    # If we don't have enough levels, use additional quantiles
    if len(support_levels) < num_levels:
        additional_quantiles = [0.20, 0.25, 0.30][: (num_levels - len(support_levels))]
        for q in additional_quantiles:
            model = sm.QuantReg(prices, X)
            result = model.fit(q=q)
            support_level = result.params[0]
            support_levels.append(support_level)

    # Make sure we have exactly num_levels support levels
    support_levels = sorted(support_levels)[:num_levels]

    return [round(level, 2) for level in support_levels]
