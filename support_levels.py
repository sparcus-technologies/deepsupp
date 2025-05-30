import numpy as np
import statsmodels.api as sm
from hmmlearn import hmm
import warnings
import os

# Must be set BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress all warnings
warnings.filterwarnings('ignore')


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
            tol=1e-2  # Less strict tolerance
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
                        p_tol=1e-3     # Less strict tolerance
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


def percentile_support(price_data, window=None, num_levels=7):
    """Identify support levels based on percentiles"""
    try:
        lower_percentiles = np.linspace(5, 35, num_levels)
        return [round(float(np.percentile(price_data, p)), 2) for p in lower_percentiles]
    except Exception as e:
        print(f"Percentile support failed: {e}")
        # Last resort - return evenly spaced levels around min price
        min_price = float(np.min(price_data))
        max_price = float(np.max(price_data))
        range_price = max_price - min_price
        return [round(min_price + i * range_price / (num_levels + 1), 2) for i in range(1, num_levels + 1)]


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
    

def deepsupp(price_data, window=20, num_levels=7):
    """
    Identify support levels using deep learning autoencoder with correlation analysis.
    """
    try:
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler
        from scipy.stats import spearmanr
        from sklearn.cluster import DBSCAN
        from tensorflow.keras.layers import Input, MultiHeadAttention, Dense, GlobalAveragePooling1D
        from tensorflow.keras.models import Model
        
        # Convert price_data to DataFrame format expected by the algorithm
        if isinstance(price_data, pd.Series):
            data = pd.DataFrame({'Close': price_data})
        else:
            data = price_data.copy()
            
        # Calculate required features exactly as in original implementation
        if 'Volume' not in data.columns:
            data['Volume'] = 100000  # Default volume if not available
            
        data['VWAP'] = (data['Close']*data['Volume']).cumsum()/data['Volume'].cumsum()
        data['PriceChangeVolume'] = data['Close'].pct_change().fillna(0)*data['Volume']
        data['VolumeRatio'] = data['Volume']/data['Volume'].rolling(20).mean()
        data.dropna(inplace=True)
        
        features = ['Close', 'VWAP', 'Volume', 'PriceChangeVolume', 'VolumeRatio']
        
        # Create and fit scaler with the data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data[features])
        
        # Compute correlation matrix (Spearman) for the data
        corr_matrix, _ = spearmanr(data_scaled, axis=1)
        
        # Prepare data sequences
        sequence_length = 30
        X = []
        for i in range(len(corr_matrix)-sequence_length):
            X.append(corr_matrix[i:i+sequence_length, i:i+sequence_length])
        
        X = np.array(X)
        
        # Define proper autoencoder model (input-output shapes match)
        inp = Input(shape=(sequence_length, sequence_length))
        x = MultiHeadAttention(num_heads=4, key_dim=sequence_length)(inp, inp)
        x = Dense(64, activation='relu')(x)
        x = Dense(sequence_length, activation='linear')(x)  # decoder layer to match input dimension
        
        autoencoder = Model(inp, x)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train Autoencoder
        autoencoder.fit(X, X, epochs=30, batch_size=16, verbose=0)
        
        # Extract embeddings from trained encoder
        encoder = Model(inp, GlobalAveragePooling1D()(x))
        embeddings = encoder.predict(X)
        
        # Clustering embeddings (unsupervised support line detection)
        clustering = DBSCAN(eps=0.08, min_samples=3).fit(embeddings)
        
        # Count the number of clusters found (excluding noise)
        unique_labels = np.unique(clustering.labels_)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # For each cluster, calculate the median price level as support line
        support_levels = []
        for cluster_label in np.unique(clustering.labels_):
            if cluster_label == -1: continue  # Skip outliers
            
            # Get points in this cluster
            cluster_indices = np.where(clustering.labels_ == cluster_label)[0]
            cluster_points = data.index[sequence_length:][cluster_indices]
            cluster_prices = data['Close'].loc[cluster_points]
            
            # Calculate median price for this cluster to use as support line
            support_level = np.median(cluster_prices)
            support_levels.append(support_level)
        
        # Add nearest support level info if we have support levels
        if support_levels:
            support_levels = sorted(support_levels)
            # Ensure we have enough levels
            while len(support_levels) < num_levels:
                # Add levels based on percentiles if needed
                percentile_val = np.percentile(data['Close'], 5 + len(support_levels) * 5)
                support_levels.append(percentile_val)
        else:
            # If no clusters found, fall back to percentile method
            return percentile_support(price_data, num_levels=num_levels)
        
        return [round(level, 2) for level in support_levels[:num_levels]]
        
    except Exception as e:
        print(f"Deep support method failed, falling back to percentile method: {e}")
        return percentile_support(price_data, num_levels=num_levels)