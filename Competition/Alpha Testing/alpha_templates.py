"""
ALPHA CREATION TEMPLATE
Copy and modify these templates to create your own alphas
"""

import pandas as pd
import numpy as np
from alpha_testing_framework import AlphaFactor, AlphaTester

# ============================================================================
# TEMPLATE 1: SIMPLE INDICATOR ALPHA
# ============================================================================

class SimpleIndicatorAlpha(AlphaFactor):
    """
    Template for alphas based on a single technical indicator
    
    HOW TO USE:
    1. Replace "SimpleIndicatorAlpha" with your alpha name
    2. Modify the calculate() method with your logic
    3. Return signals as [-1, 0, 1]
    """
    
    def __init__(self, param1=20, param2=50):
        super().__init__(f"SimpleIndicator_{param1}_{param2}")
        self.param1 = param1
        self.param2 = param2
    
    def calculate(self, df):
        """Calculate alpha signals"""
        close = df['Close']
        
        # MODIFY THIS SECTION:
        # Calculate your indicator
        indicator = close.rolling(window=self.param1).mean()  # Example: moving average
        
        # Generate signals based on indicator
        signals = np.where(close > indicator, 1, -1)  # Buy if price > MA
        
        return pd.Series(signals, index=df.index)


# ============================================================================
# TEMPLATE 2: MULTI-FACTOR ALPHA (Combining multiple signals)
# ============================================================================

class MultiFactorAlpha(AlphaFactor):
    """
    Template for alphas combining multiple indicators
    
    Example: Buy only when BOTH momentum AND volume are positive
    """
    
    def __init__(self):
        super().__init__("MultiFactor_v1")
    
    def calculate(self, df):
        """Combine multiple signals"""
        close = df['Close']
        volume = df['Volume']
        high = df['High']
        low = df['Low']
        
        # FACTOR 1: Momentum
        returns = close.pct_change(5)
        momentum_signal = np.where(returns > 0, 1, -1)
        
        # FACTOR 2: Volume
        avg_volume = volume.rolling(20).mean()
        volume_signal = np.where(volume > avg_volume, 1, -1)
        
        # FACTOR 3: Price strength
        range_pct = (high - low) / close * 100
        strength_signal = np.where(range_pct > 2, 1, -1)
        
        # COMBINE: Sum all signals
        combined = momentum_signal + volume_signal + strength_signal
        
        # Signal rules:
        # If 2+ factors are positive -> BUY
        # If 2+ factors are negative -> SELL
        # Otherwise -> NEUTRAL
        signals = np.where(combined >= 2, 1, np.where(combined <= -2, -1, 0))
        
        return pd.Series(signals, index=df.index)


# ============================================================================
# TEMPLATE 3: MEAN REVERSION ALPHA
# ============================================================================

class MeanReversionTemplateAlpha(AlphaFactor):
    """
    Mean reversion assumes prices that move too far will revert back
    """
    
    def __init__(self, zscore_threshold=2.0):
        super().__init__(f"MeanReversion_ZScore_{zscore_threshold}")
        self.zscore_threshold = zscore_threshold
    
    def calculate(self, df):
        """Calculate mean reversion signals"""
        close = df['Close']
        
        # Calculate z-score (how many standard deviations from mean)
        rolling_mean = close.rolling(window=20).mean()
        rolling_std = close.rolling(window=20).std()
        zscore = (close - rolling_mean) / rolling_std
        
        # Mean reversion logic:
        # - If price is too low (z-score < -threshold), expect it to go up -> BUY
        # - If price is too high (z-score > threshold), expect it to go down -> SELL
        signals = np.where(
            zscore < -self.zscore_threshold,
            1,  # Buy (oversold)
            np.where(
                zscore > self.zscore_threshold,
                -1,  # Sell (overbought)
                0   # Hold (neutral)
            )
        )
        
        return pd.Series(signals, index=df.index)


# ============================================================================
# TEMPLATE 4: TREND-FOLLOWING ALPHA
# ============================================================================

class TrendFollowingAlpha(AlphaFactor):
    """
    Trend following assumes price momentum continues
    """
    
    def __init__(self, fast_window=10, slow_window=30):
        super().__init__(f"TrendFollowing_{fast_window}_{slow_window}")
        self.fast_window = fast_window
        self.slow_window = slow_window
    
    def calculate(self, df):
        """Calculate trend following signals"""
        close = df['Close']
        
        # Calculate moving averages
        fast_ma = close.rolling(window=self.fast_window).mean()
        slow_ma = close.rolling(window=self.slow_window).mean()
        
        # Trend following:
        # - If fast MA > slow MA -> uptrend -> BUY
        # - If fast MA < slow MA -> downtrend -> SELL
        signals = np.where(fast_ma > slow_ma, 1, -1)
        
        # Optional: Add additional confirmation
        # Only trade if trend is strong enough
        trend_strength = (fast_ma - slow_ma) / slow_ma * 100  # % difference
        min_strength = 0.5  # 0.5% difference
        
        signals = np.where(
            abs(trend_strength) > min_strength,
            signals,
            0  # No signal if trend too weak
        )
        
        return pd.Series(signals, index=df.index)


# ============================================================================
# TEMPLATE 5: STATISTICAL ARBITRAGE ALPHA
# ============================================================================

class StatArbAlpha(AlphaFactor):
    """
    Statistical arbitrage based on historical relationships
    """
    
    def __init__(self, percentile_threshold=0.25):
        super().__init__(f"StatArb_Percentile_{percentile_threshold}")
        self.percentile_threshold = percentile_threshold
    
    def calculate(self, df):
        """Calculate stat arb signals"""
        close = df['Close']
        returns = close.pct_change()
        
        # Calculate rolling percentile of returns
        rolling_min = returns.rolling(window=30).quantile(self.percentile_threshold)
        rolling_max = returns.rolling(window=30).quantile(1 - self.percentile_threshold)
        
        # Signal when current return is at extremes
        signals = np.where(
            returns < rolling_min,
            1,  # Buy if return unusually low (mean reversion)
            np.where(
                returns > rolling_max,
                -1,  # Sell if return unusually high
                0   # Hold otherwise
            )
        )
        
        return pd.Series(signals, index=df.index)


# ============================================================================
# TEMPLATE 6: MACHINE LEARNING INSPIRED ALPHA
# ============================================================================

class FeatureScoreAlpha(AlphaFactor):
    """
    Calculate a score from multiple features, use as signal
    """
    
    def __init__(self):
        super().__init__("FeatureScore_v1")
    
    def calculate(self, df):
        """Calculate composite feature score"""
        close = df['Close']
        volume = df['Volume']
        high = df['High']
        low = df['Low']
        
        # Create normalized scores (-1 to 1) for each feature
        
        # Feature 1: Momentum score
        returns_5d = close.pct_change(5)
        momentum_score = np.where(returns_5d > 0, 1, -1) * returns_5d.abs().clip(0, 0.05) / 0.05
        
        # Feature 2: Volume score
        volume_ratio = volume / volume.rolling(20).mean()
        volume_score = np.where(volume_ratio > 1, 1, -1) * (volume_ratio - 1).clip(0, 1)
        
        # Feature 3: Price position in range
        high_20 = high.rolling(20).max()
        low_20 = low.rolling(20).min()
        price_position = (close - low_20) / (high_20 - low_20)  # 0 (low) to 1 (high)
        price_score = np.where(price_position > 0.5, 1, -1) * (2 * price_position - 1).abs()
        
        # Combine all scores
        total_score = momentum_score + volume_score + price_score
        
        # Convert score to signal
        signals = np.where(
            total_score > 1.0,
            1,   # Buy if positive score
            np.where(
                total_score < -1.0,
                -1,  # Sell if negative score
                0   # Hold if neutral
            )
        )
        
        return pd.Series(signals, index=df.index)


# ============================================================================
# TEMPLATE 7: CONTRARIAN ALPHA
# ============================================================================

class ContrarianAlpha(AlphaFactor):
    """
    Contrarian: Do opposite of crowd
    Buy when most people are selling, sell when most are buying
    """
    
    def __init__(self):
        super().__init__("Contrarian_v1")
    
    def calculate(self, df):
        """Calculate contrarian signals"""
        close = df['Close']
        
        # Calculate what most traders would do
        # (assume most traders follow momentum)
        returns = close.pct_change(10)
        crowd_signal = np.where(returns > 0, 1, -1)  # Crowd buys on up, sells on down
        
        # Do opposite (contrarian)
        contrarian_signal = -crowd_signal
        
        # Add confidence filter: only trade when conviction is high
        returns_std = returns.rolling(20).std()
        confidence = returns.abs() / returns_std.rolling(20).mean()
        
        # Trade when crowd is very certain (we bet against them)
        signals = np.where(confidence > 1.5, contrarian_signal, 0)
        
        return pd.Series(signals, index=df.index)


# ============================================================================
# TEMPLATE 8: VOLATILITY ALPHA
# ============================================================================

class VolatilityAlpha(AlphaFactor):
    """
    Trade based on volatility changes
    High volatility -> mean reversion
    Low volatility -> momentum
    """
    
    def __init__(self):
        super().__init__("Volatility_v1")
    
    def calculate(self, df):
        """Calculate volatility-based signals"""
        close = df['Close']
        returns = close.pct_change()
        
        # Calculate volatility
        current_vol = returns.rolling(20).std()
        avg_vol = returns.rolling(60).std()
        vol_ratio = current_vol / avg_vol
        
        # Get trend
        trend = np.where(close > close.rolling(20).mean(), 1, -1)
        
        # Signal logic:
        # High volatility -> mean reversion (opposite of trend)
        # Low volatility -> momentum (follow trend)
        signals = np.where(
            vol_ratio > 1.2,
            -trend,  # Mean reversion when volatile
            trend    # Momentum when calm
        )
        
        return pd.Series(signals, index=df.index)


# ============================================================================
# EXAMPLE: HOW TO USE THESE TEMPLATES
# ============================================================================

if __name__ == "__main__":
    
    # 1. Load data
    df = pd.read_csv('stock_data.csv', index_col='Date', parse_dates=True)
    
    # 2. Create your alpha instances
    alpha1 = SimpleIndicatorAlpha(param1=20, param2=50)
    alpha2 = MultiFactorAlpha()
    alpha3 = MeanReversionTemplateAlpha(zscore_threshold=2.0)
    alpha4 = TrendFollowingAlpha(fast_window=10, slow_window=30)
    alpha5 = VolatilityAlpha()
    
    # 3. Create tester
    tester = AlphaTester(df, initial_capital=100000, transaction_cost=0.001)
    
    # 4. Test all alphas
    alphas_to_test = [alpha1, alpha2, alpha3, alpha4, alpha5]
    comparison = tester.compare_alphas(alphas_to_test)
    
    # 5. Export results
    comparison.to_csv('/mnt/user-data/outputs/template_comparison.csv')
    
    print("\nResults saved!")
    print(comparison[['Alpha_Name', 'Excess_Return_%', 'Sharpe_Ratio', 'Max_Drawdown_%']])
