"""
SIMPLE ALPHA TESTING FRAMEWORK
================================
No fancy classes or interfaces. Just:
1. Load your CSV data
2. Define your alpha function
3. Run backtest
4. See results

For Quant Invitational - quick and dirty testing
"""

import pandas as pd
import numpy as np
from datetime import datetime


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def backtest_alpha(df, signals, alpha_name="Alpha", initial_capital=100000, transaction_cost=0.001):
    """
    Backtest an alpha strategy
    
    Args:
        df: DataFrame with OHLCV data (must have 'Close' column)
        signals: pandas Series with same index as df, values in [-1, 0, 1]
                 1 = long, -1 = short, 0 = no position
        alpha_name: name of your alpha (for printing)
        initial_capital: starting portfolio value
        transaction_cost: % cost per trade (e.g., 0.001 = 0.1%)
    
    Returns:
        Dictionary with results and metrics
    """
    
    bt = df.copy()
    bt['Signal'] = signals
    bt['Daily_Return'] = bt['Close'].pct_change()
    
    # Hold position until signal changes
    bt['Position'] = bt['Signal'].fillna(method='ffill').fillna(0)
    
    # Transaction cost when position changes
    bt['Position_Change'] = bt['Position'].diff().abs()
    bt['Trans_Cost'] = bt['Position_Change'] * transaction_cost
    
    # Strategy return = signal * daily_return - transaction costs
    bt['Strat_Return'] = bt['Position'].shift(1) * bt['Daily_Return'] - bt['Trans_Cost']
    
    # Cumulative return
    bt['Cumul_Return'] = (1 + bt['Strat_Return']).cumprod() - 1
    bt['Portfolio_Value'] = initial_capital * (1 + bt['Cumul_Return'])
    
    # Buy & Hold comparison
    bt['BH_Return'] = bt['Daily_Return']
    bt['BH_Cumul'] = (1 + bt['BH_Return']).cumprod() - 1
    bt['BH_Portfolio'] = initial_capital * (1 + bt['BH_Cumul'])
    
    # Calculate metrics
    metrics = _calculate_metrics(bt, alpha_name, initial_capital)
    
    return {
        'backtest_df': bt,
        'metrics': metrics
    }


def _calculate_metrics(bt, alpha_name, initial_capital):
    """Calculate performance metrics"""
    
    strat_ret = bt['Strat_Return'].dropna()
    bh_ret = bt['BH_Return'].dropna()
    
    # Returns
    total_return_pct = (bt.iloc[-1]['Portfolio_Value'] / initial_capital - 1) * 100
    bh_total_pct = (bt.iloc[-1]['BH_Portfolio'] / initial_capital - 1) * 100
    excess_return_pct = total_return_pct - bh_total_pct
    
    # Volatility (annualized)
    annual_vol = strat_ret.std() * np.sqrt(252) * 100
    bh_vol = bh_ret.std() * np.sqrt(252) * 100
    
    # Sharpe Ratio (annualized, risk-free rate = 0)
    sharpe = (strat_ret.mean() * 252) / (strat_ret.std() * np.sqrt(252)) if strat_ret.std() > 0 else 0
    bh_sharpe = (bh_ret.mean() * 252) / (bh_ret.std() * np.sqrt(252)) if bh_ret.std() > 0 else 0
    
    # Max Drawdown
    cummax = bt['Portfolio_Value'].cummax()
    drawdown = (bt['Portfolio_Value'] - cummax) / cummax
    max_dd = drawdown.min() * 100
    
    # Win Rate
    winning = (strat_ret > 0).sum()
    losing = (strat_ret < 0).sum()
    win_rate = (winning / (winning + losing) * 100) if (winning + losing) > 0 else 0
    
    # Number of trades
    num_trades = int(bt['Position_Change'].sum())
    
    return {
        'Alpha_Name': alpha_name,
        'Total_Return_%': total_return_pct,
        'BH_Return_%': bh_total_pct,
        'Excess_Return_%': excess_return_pct,
        'Annual_Volatility_%': annual_vol,
        'Sharpe_Ratio': sharpe,
        'BH_Sharpe': bh_sharpe,
        'Max_Drawdown_%': max_dd,
        'Win_Rate_%': win_rate,
        'Num_Trades': num_trades,
        'Final_Value': bt.iloc[-1]['Portfolio_Value'],
        'BH_Final_Value': bt.iloc[-1]['BH_Portfolio']
    }


def print_results(metrics):
    """Pretty print backtest results"""
    
    print("\n" + "="*70)
    print(f"ALPHA: {metrics['Alpha_Name']}")
    print("="*70)
    print(f"{'Metric':<35} {'Strategy':>15} {'Buy&Hold':>15}")
    print("-"*70)
    print(f"{'Total Return':<35} {metrics['Total_Return_%']:>14.2f}% {metrics['BH_Return_%']:>14.2f}%")
    print(f"{'Excess Return':<35} {metrics['Excess_Return_%']:>14.2f}% {'-':>15}")
    print(f"{'Annual Volatility':<35} {metrics['Annual_Volatility_%']:>14.2f}% {'-':>15}")
    print(f"{'Sharpe Ratio':<35} {metrics['Sharpe_Ratio']:>14.2f} {metrics['BH_Sharpe']:>14.2f}")
    print(f"{'Max Drawdown':<35} {metrics['Max_Drawdown_%']:>14.2f}% {'-':>15}")
    print(f"{'Win Rate':<35} {metrics['Win_Rate_%']:>14.2f}% {'-':>15}")
    print(f"{'Number of Trades':<35} {metrics['Num_Trades']:>14.0f} {'-':>15}")
    print(f"{'Final Portfolio Value':<35} ${metrics['Final_Value']:>13,.0f} ${metrics['BH_Final_Value']:>13,.0f}")
    print("="*70)


# ============================================================================
# EXAMPLE ALPHAS - MODIFY THESE OR CREATE YOUR OWN
# ============================================================================

def momentum_alpha(df, lookback=20):
    """
    Momentum: Buy stocks with positive recent returns
    
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

# ============================================================================
# MAIN - EASY TO MODIFY AND TEST NEW ALPHAS
# ============================================================================

def my_alpha(df):
    close = df['Close']
    
    # Your logic here
    signals = np.where(close>1, 1, -1)
    
    return pd.Series(signals, index=df.index)

if __name__ == "__main__":
    
    # ===== STEP 1: LOAD YOUR DATA =====
    print("Loading data...")
    
    csv_file = "stock_data.csv"  # CHANGE THIS TO YOUR FILE
    
    try:
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df = df.set_index('Date')
        
        # Check required columns
        required = ['Close', 'Volume', 'High', 'Low', 'Open']
        assert all(col in df.columns for col in required), f"Missing columns. Need: {required}"
        
        print(f"✓ Loaded {len(df)} rows")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Columns: {list(df.columns)}")
    
    except FileNotFoundError:
        print(f"Creating sample data (file {csv_file} not found)...")
        dates = pd.date_range('2020-01-01', periods=1000)
        np.random.seed(42)
        price = 100 + np.cumsum(np.random.randn(1000) * 2)
        
        df = pd.DataFrame({
            'Close': price,
            'Open': price + np.random.randn(1000),
            'High': price + abs(np.random.randn(1000) * 2),
            'Low': price - abs(np.random.randn(1000) * 2),
            'Volume': np.random.randint(1000000, 10000000, 1000)
        }, index=dates)
    
    # ===== STEP 2: TEST INDIVIDUAL ALPHAS =====
    print("\n" + "="*70)
    print("TESTING INDIVIDUAL ALPHAS")
    print("="*70)
    
    """
    # Test one alpha
    print("\n--- Testing Momentum Alpha ---")
    signals = momentum_alpha(df, lookback=20)
    result = backtest_alpha(df, signals, "Momentum_20d")
    print_results(result['metrics'])"""

    signals = my_alpha(df)
    result = backtest_alpha(df, signals, "my_alpha")
    print_results(result['metrics'])
    
    # ===== STEP 3: COMPARE MULTIPLE ALPHAS =====
    print("\n" + "="*70)
    print("COMPARING MULTIPLE ALPHAS")
    print("="*70)
    
    alphas_to_test = {
        'Momentum_20d': lambda df: momentum_alpha(df, lookback=20),
        'Momentum_10d': lambda df: momentum_alpha(df, lookback=10),
        'MeanReversion_20': lambda df: mean_reversion_alpha(df, window=20),
        'MACD': lambda df: macd_alpha(df),
        'RSI': lambda df: rsi_alpha(df),
        'SMA_20_50': lambda df: sma_crossover_alpha(df, fast=20, slow=50),
        'my_alpha': lambda df: my_alpha(df),
    }
    
    comparison = compare_alphas(df, alphas_to_test)
    
    # Save comparison
    comparison.to_csv('/mnt/user-data/outputs/alpha_comparison.csv', index=False)
    print("\n✓ Comparison saved to: alpha_comparison.csv")