# DeepSupp: A Comparative Analysis of Support Level Detection Methods with Deep Learning Integration

## Abstract

This paper presents DeepSupp, a novel approach to financial support level detection that employs deep learning techniques alongside traditional methods. We compare eight different support level detection algorithms: quantile regression, hidden Markov models (HMM), local minima detection, moving averages, percentile-based approaches, Fibonacci retracements, fractal analysis, and our proposed DeepSupp method that incorporates multi-head attention mechanisms. Using a comprehensive evaluation framework with six fundamental financial metrics—support accuracy, price proximity, volume confirmation, market regime sensitivity, support hold duration, and false breakout rate—we assess the performance of each method across multiple stocks. Our findings indicate that DeepSupp outperforms traditional methods in overall effectiveness, particularly in market adaptability and volume-based confirmation scenarios, offering traders more reliable support level identification across varying market conditions.

## 1. Introduction

Support levels represent price points where a security has historically found buying pressure sufficient to prevent further decline. Accurate identification of these levels is crucial for traders and investors making entry and exit decisions. Despite their importance, current approaches to support level detection remain heavily subjective or rely on simplified statistical methods that fail to capture complex market dynamics.

Traditional methodologies such as trend line analysis, moving averages, and Fibonacci retracements have long dominated practice, but they lack the adaptability required in today's volatile markets. Recent advancements in computational methods have introduced more sophisticated approaches like quantile regression and hidden Markov models, yet these still struggle with the non-stationary and multi-dimensional nature of financial time series.

This research makes three key contributions:

1. We provide a standardized comparison framework for evaluating support level detection methods using six fundamental financial metrics
2. We introduce DeepSupp, a novel deep learning approach utilizing multi-head attention mechanisms to identify meaningful support levels
3. We demonstrate empirically that deep learning techniques can capture subtle price-volume relationships that traditional methods miss

## 2. Literature Review

### 2.1 Traditional Support Level Detection Methods

Early work in support level detection relied primarily on visual identification (Edwards & Magee, 1948) and simple moving averages (Murphy, 1999). Fibonacci retracements gained popularity through the work of Fischer (1993), who connected these mathematical ratios to market psychology. Williams (1995) introduced fractal theory to technical analysis, proposing that markets exhibit self-similar patterns across different timeframes.

Statistical approaches emerged with the work of Lo et al. (2000), who applied kernel regression to identify significant support and resistance levels. Quantile regression methods were later proposed by Koenker and Hallock (2001) as a way to model the lower bounds of price movements. More recently, hidden Markov models have been applied to financial time series by Hassan and Nath (2005) to capture regime-switching behavior in markets.

### 2.2 Machine Learning in Technical Analysis

The application of machine learning to technical analysis has grown significantly in recent years. Sezer et al. (2017) provided a comprehensive survey of deep learning models in financial time series forecasting, while Tsantekidis et al. (2017) demonstrated the power of convolutional neural networks in capturing price patterns.

Despite these advancements, the specific application of deep learning to support level detection remains underexplored. Most existing research focuses on price prediction rather than identifying structural levels in the market. This research gap motivates our development of DeepSupp, which specifically targets support level identification through attention-based mechanisms.

## 3. Methodology

### 3.1 Support Level Detection Methods

We implemented and compared eight distinct methods for support level detection:

#### 3.1.1 Quantile Regression

This method fits regression models at specific quantiles (5th to 35th percentiles) of the price distribution, identifying price levels that act as lower bounds for market movements.

#### 3.1.2 Hidden Markov Models (HMM)

Our implementation uses HMM to identify distinct states in price data, with lower states corresponding to support levels. We employ a spherical covariance structure for improved numerical stability.

#### 3.1.3 Local Minima Detection

This approach identifies price points that are local minimums within specified windows, representing potential areas where buying pressure emerged.

#### 3.1.4 Moving Average Support

Support levels are derived from key moving averages (20, 50, 100, 150, 200, 250, and 300-day), which often serve as dynamic support in trending markets.

#### 3.1.5 Percentile-Based Support

This method uses lower percentiles of the price distribution (5th to 35th) as potential support levels, providing a simple statistical baseline.

#### 3.1.6 Fibonacci Support

Based on Fibonacci retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%, 88.6%, 92.5%) applied to the range between the highest high and lowest low.

#### 3.1.7 Fractal Support

Implements Bill Williams' fractal concept, identifying points where a low is surrounded by higher lows on both sides within a specified window.

#### 3.1.8 DeepSupp (Proposed Method)

Our novel approach utilizes a lightweight PyTorch-based multi-head attention autoencoder to analyze correlation matrices of price-volume relationships, followed by DBSCAN clustering to identify meaningful support levels.

The DeepSupp method specifically:

1. Processes price-volume relationships through correlation analysis
2. Applies a multi-head attention mechanism to capture temporal dependencies
3. Uses an autoencoder architecture to create compressed representations
4. Employs density-based clustering to identify support level zones

### 3.2 Data Collection and Processing

We collected daily price and volume data for S&P 500 companies over a two-year period using the Yahoo Finance API. The data was preprocessed to handle missing values and normalized for each detection method. For each stock, we calculated derived features including VWAP (Volume-Weighted Average Price), price-volume change relationships, and volume ratios.

### 3.3 Evaluation Framework

Our evaluation framework assesses each method using six fundamental financial metrics:

1. **Support Accuracy**: Measures how closely predicted support levels align with actual price bounces
2. **Price Proximity**: Evaluates how well predicted levels match statistical price percentiles
3. **Volume Confirmation**: Assesses whether support levels coincide with volume spikes that confirm buyer presence
4. **Market Regime Sensitivity**: Measures performance across bull, bear, and sideways markets
5. **Support Hold Duration**: Evaluates how long identified support levels remain valid before breaking
6. **False Breakout Rate**: Measures the frequency of price recovery after brief support breaks

These metrics are combined into an overall score using weighted averaging, with higher weights assigned to accuracy (25%), price proximity (20%), and volume confirmation (20%).

## 4. Results and Discussion

### 4.1 Comparative Performance

Our evaluation across the S&P 500 dataset revealed notable differences between the eight support level detection methods. Table 1 presents the average scores for each metric.

**Table 1: Performance Comparison of Support Level Detection Methods**

| Method              | Support Accuracy | Price Proximity | Volume Confirmation | Market Sensitivity | Support Duration | Breakout Recovery | Overall Score |
| ------------------- | ---------------- | --------------- | ------------------- | ------------------ | ---------------- | ----------------- | ------------- |
| DeepSupp            | 0.782            | 0.754           | 0.812               | 0.776              | 0.685            | 0.721             | 0.763         |
| HMM                 | 0.753            | 0.731           | 0.692               | 0.704              | 0.644            | 0.698             | 0.720         |
| Quantile Regression | 0.744            | 0.769           | 0.642               | 0.663              | 0.718            | 0.685             | 0.712         |
| Local Minima        | 0.726            | 0.685           | 0.689               | 0.642              | 0.691            | 0.702             | 0.689         |
| Fractal             | 0.698            | 0.673           | 0.672               | 0.635              | 0.643            | 0.714             | 0.671         |
| Moving Average      | 0.682            | 0.654           | 0.683               | 0.712              | 0.587            | 0.675             | 0.665         |
| Fibonacci           | 0.643            | 0.712           | 0.605               | 0.625              | 0.615            | 0.687             | 0.647         |
| Percentile          | 0.624            | 0.748           | 0.512               | 0.583              | 0.568            | 0.654             | 0.615         |

The DeepSupp method demonstrated superior performance in overall score, with particularly strong results in volume confirmation and market regime sensitivity. This suggests that the deep learning approach captures important price-volume relationships that traditional methods miss.

### 4.2 Analysis of DeepSupp's Effectiveness

DeepSupp's effectiveness can be attributed to several factors:

1. **Attention to Volume-Price Dynamics**: The multi-head attention mechanism effectively captures the relationship between price movements and volume, which is critical for identifying genuine support levels versus random fluctuations.

2. **Adaptability Across Market Regimes**: Unlike traditional methods that often perform well only in specific market conditions, DeepSupp maintained consistent performance across bull, bear, and sideways markets (regime diversity score of 0.71).

3. **Effective Compression of Time-Series Information**: The autoencoder architecture successfully compresses temporal information while preserving essential structures that indicate support levels.

4. **Cluster-Based Level Identification**: The DBSCAN clustering step effectively groups similar price behaviors, reducing noise and identifying meaningful support zones rather than precise but brittle price points.

### 4.3 Method-Specific Strengths

Each method demonstrated unique strengths that make them suitable for different trading scenarios:

- **DeepSupp**: Best for volume-based trading strategies and adaptable across market conditions
- **Quantile Regression**: Excellent for position trading with longer holding periods
- **HMM**: Strong in capturing regime changes, useful for transitional markets
- **Local Minima**: Effective for day trading and short-term entries
- **Fractal**: Superior in detecting false breakouts
- **Moving Average**: Best for trend-following strategies in strongly trending markets
- **Fibonacci**: Performs well in markets with clear swing highs and lows
- **Percentile**: Provides reliable baseline in highly volatile markets

### 4.4 Limitations

While DeepSupp demonstrates superior overall performance, we identified several limitations:

1. Computational overhead is higher than traditional methods, though our lightweight implementation mitigates this concern
2. Performance depends on sufficient historical data (minimum 30 trading days required)
3. Requires both price and volume data, unlike some methods that can operate on price alone

## 5. Conclusion

This study demonstrates the efficacy of deep learning approaches for financial support level detection. The proposed DeepSupp method consistently outperforms traditional approaches across multiple evaluation metrics, with particular strength in volume confirmation and market regime sensitivity.

The superior performance of DeepSupp suggests that attention-based deep learning models can capture complex market dynamics that elude traditional statistical methods. By examining correlations between price and volume data through the lens of multi-head attention mechanisms, DeepSupp identifies support levels that more closely match actual market behavior.

Future work should explore the integration of additional market data sources, such as order book information and sentiment analysis, to further enhance the accuracy of support level detection. Additionally, the application of reinforcement learning to dynamically adjust support level identification based on market feedback represents a promising direction for research.

## References

1. Edwards, R. D., & Magee, J. (1948). Technical Analysis of Stock Trends.
2. Murphy, J. J. (1999). Technical Analysis of the Financial Markets.
3. Fischer, R. (1993). Fibonacci Applications and Strategies for Traders.
4. Williams, B. (1995). Trading Chaos: Applying Expert Techniques to Maximize Your Profits.
5. Lo, A. W., Mamaysky, H., & Wang, J. (2000). Foundations of Technical Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation. The Journal of Finance, 55(4), 1705-1765.
6. Koenker, R., & Hallock, K. F. (2001). Quantile Regression. Journal of Economic Perspectives, 15(4), 143-156.
7. Hassan, M. R., & Nath, B. (2005). Stock Market Forecasting Using Hidden Markov Model: A New Approach. In Proceedings of the 5th International Conference on Intelligent Systems Design and Applications.
8. Sezer, O. B., Gudelek, M. U., & Ozbayoglu, A. M. (2017). Financial Time Series Forecasting with Deep Learning: A Systematic Literature Review. ArXiv Preprint ArXiv:1911.13288.
9. Tsantekidis, A., Passalis, N., Tefas, A., Kanniainen, J., Gabbouj, M., & Iosifidis, A. (2017). Forecasting Stock Prices from the Limit Order Book Using Convolutional Neural Networks. In IEEE 19th Conference on Business Informatics (CBI).
