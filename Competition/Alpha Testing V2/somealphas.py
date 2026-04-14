
# ============================================================================
# EXAMPLE ALPHAS - MODIFY THESE OR CREATE YOUR OWN
# ============================================================================
def momentum_alpha(df, lookback=20):
    """
    Momentum: Buy stocks with positive recent returns
    `
    Args:
        df: DataFrame with OHLCV
        lookback: days to look back for returns
    
    Returns:
        Series with signals [-1, 1]
    """
    returns = df['Close'].pct_change(lookback)
    signals = np.where(returns > 0, 1, -1)
    signals = np.where(np.isnan(returns), 0, signals)
    return pd.Series(signals, index=df.index)


def mean_reversion_alpha(df, window=20, num_std=2):
    """
    Mean Reversion: Buy oversold (price < lower band), sell overbought (price > upper band)
    Uses Bollinger Bands
    
    Args:
        df: DataFrame with OHLCV
        window: rolling window for bands
        num_std: number of standard deviations
    
    Returns:
        Series with signals [-1, 0, 1]
    """
    close = df['Close']
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    
    signals = np.zeros(len(df))
    signals[close < lower] = 1   # Oversold - buy
    signals[close > upper] = -1  # Overbought - sell
    
    return pd.Series(signals, index=df.index)


def macd_alpha(df):
    """
    MACD: Buy when MACD > signal line, sell when MACD < signal line
    
    Args:
        df: DataFrame with OHLCV
    
    Returns:
        Series with signals [-1, 1]
    """
    close = df['Close']
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    
    signals = np.where(macd > signal_line, 1, -1)
    return pd.Series(signals, index=df.index)


def rsi_alpha(df, threshold_low=30, threshold_high=70):
    """
    RSI: Buy when oversold (RSI < 30), sell when overbought (RSI > 70)
    
    Args:
        df: DataFrame with OHLCV
        threshold_low: RSI level for oversold (buy)
        threshold_high: RSI level for overbought (sell)
    
    Returns:
        Series with signals [-1, 0, 1]
    """
    close = df['Close']
    delta = close.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(len(df))
    signals[rsi < threshold_low] = 1   # Oversold - buy
    signals[rsi > threshold_high] = -1 # Overbought - sell
    
    return pd.Series(signals, index=df.index)


def sma_crossover_alpha(df, fast=20, slow=50):
    """
    SMA Crossover: Buy when fast MA > slow MA, sell when fast MA < slow MA
    
    Args:
        df: DataFrame with OHLCV
        fast: fast moving average period
        slow: slow moving average period
    
    Returns:
        Series with signals [-1, 1]
    """
    close = df['Close']
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    
    signals = np.where(ma_fast > ma_slow, 1, -1)
    return pd.Series(signals, index=df.index)


def volume_alpha(df, window=20, volume_multiplier=1.5):
    """
    Volume: Trade larger moves when volume is high
    
    Args:
        df: DataFrame with OHLCV
        window: rolling window for average volume
        volume_multiplier: how much above average to trigger signal
    
    Returns:
        Series with signals [-1, 0, 1]
    """
    close = df['Close']
    volume = df['Volume']
    
    price_change = close.pct_change()
    avg_volume = volume.rolling(window).mean()
    volume_ratio = volume / avg_volume
    
    signals = np.zeros(len(df))
    signals[(price_change > 0) & (volume_ratio > volume_multiplier)] = 1   # Strong up
    signals[(price_change < 0) & (volume_ratio > volume_multiplier)] = -1  # Strong down
    
    return pd.Series(signals, index=df.index)


# ============================================================================
# COMPARISON FUNCTION
# ============================================================================

def compare_alphas(df, alpha_functions_dict, initial_capital=100000):
    """
    Test multiple alphas and compare them
    
    Args:
        df: DataFrame with OHLCV
        alpha_functions_dict: dict like {'Alpha_Name': alpha_function}
                             where alpha_function(df) returns signals
        initial_capital: starting capital
    
    Returns:
        DataFrame with comparison results
    """
    
    results_list = []
    
    for alpha_name, alpha_func in alpha_functions_dict.items():
        print(f"\nTesting {alpha_name}...")
        
        # Generate signals
        signals = alpha_func(df)
        
        # Backtest
        result = backtest_alpha(df, signals, alpha_name, initial_capital)
        metrics = result['metrics']
        
        # Print results
        print_results(metrics)
        
        results_list.append(metrics)
    
    # Create comparison table
    comparison_df = pd.DataFrame(results_list)
    
    # Print summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY - Ranked by Excess Return")
    print("="*70)
    comparison_df_sorted = comparison_df.sort_values('Excess_Return_%', ascending=False)
    
    cols = ['Alpha_Name', 'Excess_Return_%', 'Sharpe_Ratio', 'Max_Drawdown_%', 'Win_Rate_%', 'Num_Trades']
    print(comparison_df_sorted[cols].to_string(index=False))
    
    return comparison_df
