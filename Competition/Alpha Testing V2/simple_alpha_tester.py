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
        signals: pandas Series with same index as df, values in [-1,1]
                 1 = long, -1 = short, and by how much
        alpha_name: name of your alpha (for printing)
        initial_capital: starting portfolio value
        transaction_cost: % cost per trade (e.g., 0.001 = 0.1%)
    
    Returns:
        Dictionary with results and metrics
    """
    
    bt = df.copy()
    bt['Signal'] = signals #Adds singal column
    bt['Daily_Return'] = bt['Close'].pct_change()#caluclate close percent change
    
    # Hold position until signal changes (fixed for pandas 3.0+)
    bt['Position'] = bt['Signal'].ffill().fillna(0)#forward fills
    
    # Transaction cost when position changes
    bt['Position_Change'] = bt['Position'].diff().abs()#calculate the position change, how much we hold has changed
    bt['Trans_Cost'] = bt['Position_Change'] * transaction_cost#the absolute difference is our transaction cost
    
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
    
    # Sortino Ratio (uses only downside volatility)
    # Downside deviation: standard deviation of negative returns only
    negative_returns = strat_ret[strat_ret < 0]
    downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.0001
    
    if downside_deviation > 0:
        sortino = (strat_ret.mean() * 252) / downside_deviation
    else:
        sortino = 0
    
    # Sortino for Buy & Hold
    bh_negative_returns = bh_ret[bh_ret < 0]
    bh_downside_deviation = bh_negative_returns.std() * np.sqrt(252) if len(bh_negative_returns) > 0 else 0.0001
    
    if bh_downside_deviation > 0:
        bh_sortino = (bh_ret.mean() * 252) / bh_downside_deviation
    else:
        bh_sortino = 0

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
        'Sortino_Ratio': sortino,         
        'BH_Sortino': bh_sortino,            
        'Max_Drawdown_%': max_dd,
        'Win_Rate_%': win_rate,
        'Num_Trades': num_trades,
        'Final_Value': bt.iloc[-1]['Portfolio_Value'],
        'BH_Final_Value': bt.iloc[-1]['BH_Portfolio']
    }

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
    print(f"{'Sortino Ratio':<35} {metrics['Sortino_Ratio']:>14.2f} {metrics['BH_Sortino']:>14.2f}")
    print(f"{'Max Drawdown':<35} {metrics['Max_Drawdown_%']:>14.2f}% {'-':>15}")
    print(f"{'Win Rate':<35} {metrics['Win_Rate_%']:>14.2f}% {'-':>15}")
    print(f"{'Number of Trades':<35} {metrics['Num_Trades']:>14.0f} {'-':>15}")
    print(f"{'Final Portfolio Value':<35} ${metrics['Final_Value']:>13,.0f} ${metrics['BH_Final_Value']:>13,.0f}")
    print("="*70)



# ============================================================================
# MAIN - EASY TO MODIFY AND TEST NEW ALPHAS
# ============================================================================
def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr

def calculate_alpha(df):
    """
    Multi-factor alpha combining:
    1. Volume-weighted momentum
    2. Volatility-adjusted mean reversion
    3. Intraday strength (Close vs Open)
    4. Volume confirmation
    """
    
    # Make a copy to avoid warnings
    data = df.copy()
    
    # For each stock, calculate rolling metrics
    stocks = data['Stock'].unique()
    alphas = []
    
    for stock in stocks:
        stock_data = data[data['Stock'] == stock].sort_values('Date').copy()
        
        # Initialize with NaN
        stock_data['alpha'] = np.nan
        
        # Need minimum 20-30 days of data for reliable calculations
        if len(stock_data) < 30:
            print(f"Warning: {stock} has only {len(stock_data)} days, skipping")
            alphas.append(stock_data[['Date', 'Stock', 'alpha']])
            continue
        
        # === Factor 1: Volume-Weighted Momentum (20-day) ===
        stock_data['returns'] = stock_data['Close'].pct_change()
        stock_data['volume_ma'] = stock_data['Volume'].rolling(20, min_periods=1).mean()
        stock_data['volume_ratio'] = stock_data['Volume'] / stock_data['volume_ma']
        stock_data['vw_momentum'] = stock_data['returns'].rolling(5, min_periods=1).mean() * stock_data['volume_ratio']
        
        # === Factor 2: Volatility-Adjusted Mean Reversion ===
        stock_data['atr'] = calculate_atr(stock_data)
        stock_data['ma_10'] = stock_data['Close'].rolling(10, min_periods=1).mean()
        stock_data['std_10'] = stock_data['Close'].rolling(10, min_periods=1).std()
        stock_data['price_vs_ma'] = (stock_data['Close'] - stock_data['ma_10']) / (stock_data['std_10'] + 1e-8)
        stock_data['mean_reversion'] = -stock_data['price_vs_ma'] / (stock_data['atr'] + 1e-8)
        
        # === Factor 3: Intraday Strength ===
        stock_data['intraday_range'] = stock_data['High'] - stock_data['Low']
        stock_data['intraday_strength'] = (stock_data['Close'] - stock_data['Open']) / (stock_data['intraday_range'] + 1e-8)
        stock_data['intraday_signal'] = stock_data['intraday_strength'].rolling(3, min_periods=1).mean()
        
        # === Factor 4: Volume Breakout ===
        stock_data['volume_ma_10'] = stock_data['Volume'].rolling(10, min_periods=1).mean()
        stock_data['volume_surge'] = (stock_data['Volume'] > stock_data['volume_ma_10'] * 1.5).astype(int)
        stock_data['high_10_max'] = stock_data['High'].rolling(10, min_periods=1).max().shift(1)
        stock_data['price_breakout'] = (stock_data['Close'] > stock_data['high_10_max']).astype(int)
        stock_data['breakout_signal'] = (stock_data['volume_surge'] + stock_data['price_breakout']) / 2
        
        # === Factor 5: Trend Following (EMA crossover) ===
        stock_data['ema_fast'] = stock_data['Close'].ewm(span=5, adjust=False, min_periods=1).mean()
        stock_data['ema_slow'] = stock_data['Close'].ewm(span=15, adjust=False, min_periods=1).mean()
        stock_data['trend_signal'] = (stock_data['ema_fast'] - stock_data['ema_slow']) / (stock_data['Close'] + 1e-8)
        
        # === Combine factors with optimized weights ===
        stock_data['alpha_raw'] = (
            0.35 * stock_data['vw_momentum'].fillna(0) +
            0.25 * stock_data['mean_reversion'].fillna(0) +
            0.20 * stock_data['intraday_signal'].fillna(0) +
            0.10 * stock_data['breakout_signal'].fillna(0) +
            0.10 * stock_data['trend_signal'].fillna(0)
        )
        
        # Normalize alpha within each stock (z-score) - use expanding window to avoid lookahead
        alpha_mean = stock_data['alpha_raw'].expanding(min_periods=1).mean()
        alpha_std = stock_data['alpha_raw'].expanding(min_periods=1).std()
        stock_data['alpha'] = (stock_data['alpha_raw'] - alpha_mean) / (alpha_std + 1e-8)
        
        # Clip extreme values
        stock_data['alpha'] = stock_data['alpha'].clip(-3, 3)
        
        # Keep only rows with valid alpha (after at least 20 days of data)
        stock_data.loc[stock_data.index[:20], 'alpha'] = np.nan
        
        alphas.append(stock_data[['Date', 'Stock', 'alpha']])
    
    # Combine all stocks
    result = pd.concat(alphas, ignore_index=True)
    
    return result

if __name__ == "__main__":
    
    # ===== STEP 1: LOAD YOUR DATA =====
    print("Loading data...")
    
    csv_file = "stock_data.csv"  # CHANGE THIS TO YOUR FILE
    
    try:
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Check required columns
        required = ['Close', 'Volume', 'High', 'Low', 'Open','Stock','Date']
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
    
    # Test one alpha
    print("\n--- Testing Momentum Alpha ---")
    signals = calculate_alpha(df)

    new_df=df[df['Stock']=='GAMMA'].sort_values('Date').copy()
    sd = signals[signals['Stock'] == 'GAMMA'].sort_values('Date').copy()
    sd=sd["alpha"].copy()

    result = backtest_alpha(new_df, sd, "calculate_alpha")

    print_results(result['metrics'])
    
    # ===== STEP 3: COMPARE MULTIPLE ALPHAS =====
    print("\n" + "="*70)
    print("COMPARING MULTIPLE ALPHAS")
    print("="*70)
    """
    alphas_to_test = {
        'Momentum_20d': lambda df: momentum_alpha(df, lookback=20),
        'Momentum_10d': lambda df: momentum_alpha(df, lookback=10),
        'MeanReversion_20': lambda df: mean_reversion_alpha(df, window=20),
        'MACD': lambda df: macd_alpha(df),
        'RSI': lambda df: rsi_alpha(df),
        'SMA_20_50': lambda df: sma_crossover_alpha(df, fast=20, slow=50),
        'Volume': lambda df: volume_alpha(df),
    }"""
    alphas_to_test={"calculate_alpha":lambda df: calculate_alpha(df)}
    
    comparison = compare_alphas(df, alphas_to_test)
    
    # Save comparison
    # Just save in your current folder
    comparison.to_csv('alpha_comparison.csv', index=False)
    
    print("\n✓ Comparison saved to: alpha_comparison.csv")