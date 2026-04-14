"""
Alpha Factor Testing Framework - Quant Invitational
====================================================
Test any alpha factor on historical stock data
Easily modify and insert new alphas for testing

Author: Quant Team
Purpose: Systematic backtesting of alpha factors
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# BASE ALPHA CLASS - INHERIT FROM THIS TO CREATE YOUR ALPHAS
# ============================================================================

class AlphaFactor(ABC):
    """
    Base class for all alpha factors.
    
    To create your own alpha:
    1. Inherit from AlphaFactor
    2. Implement calculate() method
    3. Return pandas Series with values [-1, 0, 1] or continuous values
    
    Returns:
        -1 or negative: Short signal (expect price to go down)
         0: Neutral
         1 or positive: Long signal (expect price to go up)
    """
    
    def __init__(self, name):
        self.name = name
        self.df = None
        self.signals = None
    
    @abstractmethod
    def calculate(self, df):
        """
        Calculate the alpha signal
        
        Args:
            df: DataFrame with OHLCV data
               Required columns: Close, Volume, High, Low, Open
               Optional: any custom columns you add
        
        Returns:
            pandas Series with same length as df containing signals
        """
        pass
    
    def fit(self, df):
        """Fit the alpha to data and generate signals"""
        self.df = df.copy()
        self.signals = self.calculate(df)
        return self.signals


# ============================================================================
# EXAMPLE ALPHAS - MODIFY THESE OR CREATE YOUR OWN
# ============================================================================

class MomentumAlpha(AlphaFactor):
    """
    Simple momentum alpha: Buy stocks with positive recent returns
    """
    
    def __init__(self, lookback_period=20):
        super().__init__(f"Momentum_{lookback_period}d")
        self.lookback_period = lookback_period
    
    def calculate(self, df):
        # Calculate returns over lookback period
        returns = df['Close'].pct_change(self.lookback_period)
        
        # Convert to signals: positive return = buy, negative = sell
        signals = np.where(returns > 0, 1, -1)
        signals = np.where(np.isnan(returns), 0, signals)
        
        return pd.Series(signals, index=df.index)


class MeanReversionAlpha(AlphaFactor):
    """
    Mean reversion alpha: Buy stocks that are oversold, sell overbought
    Uses Bollinger Bands
    """
    
    def __init__(self, window=20):
        super().__init__(f"MeanReversion_{window}d")
        self.window = window
    
    def calculate(self, df):
        close = df['Close']
        
        # Calculate Bollinger Bands
        sma = close.rolling(window=self.window).mean()
        std = close.rolling(window=self.window).std()
        
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        # Signal: price below lower band = oversold = buy
        #         price above upper band = overbought = sell
        signals = np.zeros(len(df))
        signals[close < lower_band] = 1   # Oversold - buy
        signals[close > upper_band] = -1  # Overbought - sell
        
        return pd.Series(signals, index=df.index)


class MacdAlpha(AlphaFactor):
    """
    MACD (Moving Average Convergence Divergence) alpha
    Buy when MACD crosses above signal line
    """
    
    def __init__(self):
        super().__init__("MACD")
    
    def calculate(self, df):
        close = df['Close']
        
        # Calculate MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        # Signal: MACD > signal line = buy, MACD < signal line = sell
        signals = np.where(macd_line > signal_line, 1, -1)
        
        return pd.Series(signals, index=df.index)


class VolumeAlpha(AlphaFactor):
    """
    Volume-based alpha: High volume moves are more significant
    Buy when price up with high volume, sell when price down with high volume
    """
    
    def __init__(self, window=20):
        super().__init__(f"Volume_{window}d")
        self.window = window
    
    def calculate(self, df):
        close = df['Close']
        volume = df['Volume']
        
        # Price change
        price_change = close.pct_change()
        
        # Volume compared to average
        avg_volume = volume.rolling(window=self.window).mean()
        volume_ratio = volume / avg_volume
        
        # Signal: if price up AND high volume = strong buy
        #         if price down AND high volume = strong sell
        signals = np.zeros(len(df))
        signals[(price_change > 0) & (volume_ratio > 1.5)] = 1
        signals[(price_change < 0) & (volume_ratio > 1.5)] = -1
        
        return pd.Series(signals, index=df.index)


class RSIAlpha(AlphaFactor):
    """
    RSI (Relative Strength Index) alpha
    Buy when oversold (RSI < 30), sell when overbought (RSI > 70)
    """
    
    def __init__(self):
        super().__init__("RSI")
    
    def calculate(self, df):
        close = df['Close']
        delta = close.diff()
        
        # Calculate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Average gains and losses
        avg_gains = gains.rolling(window=14).mean()
        avg_losses = losses.rolling(window=14).mean()
        
        # RSI calculation
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Signals
        signals = np.zeros(len(df))
        signals[rsi < 30] = 1   # Oversold - buy
        signals[rsi > 70] = -1  # Overbought - sell
        
        return pd.Series(signals, index=df.index)


class CustomAlpha(AlphaFactor):
    """
    TEMPLATE: Create your own alpha here
    
    Instructions:
    1. Replace "CustomAlpha" with your alpha name
    2. Implement your logic in calculate()
    3. Return signals as pandas Series with values in [-1, 0, 1] range
    """
    
    def __init__(self):
        super().__init__("Custom_Alpha_v1")
    
    def calculate(self, df):
        """
        Example: Buy when closing price > 50-day moving average
                Sell when closing price < 50-day moving average
        
        MODIFY THIS LOGIC FOR YOUR OWN ALPHA:
        """
        close = df['Close']
        
        # Your alpha logic here
        ma_50 = close.rolling(window=50).mean()
        
        signals = np.where(close > ma_50, 1, -1)
        signals = np.where(np.isnan(signals), 0, signals)
        
        return pd.Series(signals, index=df.index)


# ============================================================================
# ALPHA TESTING ENGINE
# ============================================================================

class AlphaTester:
    """
    Backtesting engine for alpha factors
    Tests factor performance, generates signals, calculates P&L
    """
    
    def __init__(self, df, initial_capital=100000, transaction_cost=0.001):
        """
        Args:
            df: DataFrame with Close, Volume, High, Low, Open prices
            initial_capital: Starting portfolio value
            transaction_cost: Percentage cost per trade (0.001 = 0.1%)
        """
        self.df = df.copy()
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.results = {}
    
    def test_alpha(self, alpha_factor, rebalance_frequency='daily'):
        """
        Backtest an alpha factor
        
        Args:
            alpha_factor: AlphaFactor instance
            rebalance_frequency: 'daily', 'weekly', 'monthly'
        
        Returns:
            Dictionary with backtest results and metrics
        """
        
        print(f"\n{'='*70}")
        print(f"Testing Alpha: {alpha_factor.name}")
        print(f"{'='*70}")
        
        # Generate signals
        signals = alpha_factor.fit(self.df)
        
        # Calculate returns
        backtest_df = self.df.copy()
        backtest_df['Signal'] = signals
        backtest_df['Daily_Return'] = backtest_df['Close'].pct_change()
        
        # Strategy return: multiply daily return by signal (holding signal)
        # This is crucial: you hold the signal until it changes
        backtest_df['Position'] = signals.ffill().fillna(0)
        
        # Transaction costs when position changes
        backtest_df['Position_Change'] = backtest_df['Position'].diff().abs()
        backtest_df['Transaction_Cost'] = backtest_df['Position_Change'] * self.transaction_cost
        
        # Strategy return = signal * daily return - transaction costs
        backtest_df['Strategy_Return'] = (
            backtest_df['Position'].shift(1) * backtest_df['Daily_Return'] - 
            backtest_df['Transaction_Cost']
        )
        
        # Cumulative wealth
        backtest_df['Cumulative_Return'] = (1 + backtest_df['Strategy_Return']).cumprod() - 1
        backtest_df['Portfolio_Value'] = self.initial_capital * (1 + backtest_df['Cumulative_Return'])
        
        # Buy and hold comparison
        backtest_df['BH_Return'] = backtest_df['Daily_Return']
        backtest_df['BH_Cumulative'] = (1 + backtest_df['BH_Return']).cumprod() - 1
        backtest_df['BH_Portfolio_Value'] = self.initial_capital * (1 + backtest_df['BH_Cumulative'])
        
        # Calculate metrics
        metrics = self._calculate_metrics(backtest_df, alpha_factor.name)
        
        # Store results
        self.results[alpha_factor.name] = {
            'backtest_df': backtest_df,
            'metrics': metrics,
            'alpha_factor': alpha_factor
        }
        
        # Print summary
        self._print_metrics(metrics)
        
        return backtest_df, metrics
    
    def _calculate_metrics(self, backtest_df, alpha_name):
        """Calculate performance metrics"""
        
        strategy_returns = backtest_df['Strategy_Return'].dropna()
        bh_returns = backtest_df['BH_Return'].dropna()
        
        # Basic returns
        total_return = (backtest_df.iloc[-1]['Portfolio_Value'] / self.initial_capital - 1) * 100
        bh_total_return = (backtest_df.iloc[-1]['BH_Portfolio_Value'] / self.initial_capital - 1) * 100
        excess_return = total_return - bh_total_return
        
        # Annualized metrics (assuming 252 trading days per year)
        annual_return = total_return * (252 / len(strategy_returns))
        annual_volatility = strategy_returns.std() * np.sqrt(252) * 100
        annual_bh_volatility = bh_returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (strategy_returns.mean() * 252) / (strategy_returns.std() * np.sqrt(252)) if strategy_returns.std() > 0 else 0
        bh_sharpe = (bh_returns.mean() * 252) / (bh_returns.std() * np.sqrt(252)) if bh_returns.std() > 0 else 0
        
        # Sortino ratio (only downside volatility)
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino = (strategy_returns.mean() * 252) / downside_volatility if downside_volatility > 0 else 0
        
        # Maximum drawdown
        cummax = backtest_df['Portfolio_Value'].cummax()
        drawdown = (backtest_df['Portfolio_Value'] - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # Win rate
        winning_days = (strategy_returns > 0).sum()
        losing_days = (strategy_returns < 0).sum()
        win_rate = (winning_days / (winning_days + losing_days) * 100) if (winning_days + losing_days) > 0 else 0
        
        # Number of trades
        num_trades = backtest_df['Position_Change'].sum()
        
        # Calmar ratio (return / max drawdown)
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            'Alpha_Name': alpha_name,
            'Total_Return_%': total_return,
            'BH_Total_Return_%': bh_total_return,
            'Excess_Return_%': excess_return,
            'Annual_Return_%': annual_return,
            'Annual_Volatility_%': annual_volatility,
            'BH_Volatility_%': annual_bh_volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'BH_Sharpe': bh_sharpe,
            'Sortino_Ratio': sortino,
            'Max_Drawdown_%': max_drawdown,
            'Win_Rate_%': win_rate,
            'Num_Trades': int(num_trades),
            'Calmar_Ratio': calmar,
            'Final_Portfolio_Value': backtest_df.iloc[-1]['Portfolio_Value'],
            'Final_BH_Value': backtest_df.iloc[-1]['BH_Portfolio_Value']
        }
        
        return metrics
    
    def _print_metrics(self, metrics):
        """Pretty print metrics"""
        
        print(f"\n{'Metric':<30} {'Strategy':>15} {'Buy&Hold':>15} {'Difference':>15}")
        print("-" * 75)
        print(f"{'Total Return':<30} {metrics['Total_Return_%']:>14.2f}% {metrics['BH_Total_Return_%']:>14.2f}% {metrics['Excess_Return_%']:>14.2f}%")
        print(f"{'Annual Return':<30} {metrics['Annual_Return_%']:>14.2f}% {'-':>15} {'-':>15}")
        print(f"{'Volatility (Annual)':<30} {metrics['Annual_Volatility_%']:>14.2f}% {metrics['BH_Volatility_%']:>14.2f}% {'-':>15}")
        print(f"{'Sharpe Ratio':<30} {metrics['Sharpe_Ratio']:>14.2f} {metrics['BH_Sharpe']:>14.2f} {'-':>15}")
        print(f"{'Sortino Ratio':<30} {metrics['Sortino_Ratio']:>14.2f} {'-':>15} {'-':>15}")
        print(f"{'Max Drawdown':<30} {metrics['Max_Drawdown_%']:>14.2f}% {'-':>15} {'-':>15}")
        print(f"{'Calmar Ratio':<30} {metrics['Calmar_Ratio']:>14.2f} {'-':>15} {'-':>15}")
        print(f"{'Win Rate':<30} {metrics['Win_Rate_%']:>14.2f}% {'-':>15} {'-':>15}")
        print(f"{'Number of Trades':<30} {metrics['Num_Trades']:>14.0f} {'-':>15} {'-':>15}")
        print(f"{'Final Portfolio Value':<30} ${metrics['Final_Portfolio_Value']:>14,.2f} ${metrics['Final_BH_Value']:>14,.2f} {'-':>15}")
    
    def compare_alphas(self, alpha_list):
        """
        Test multiple alphas and create comparison table
        
        Args:
            alpha_list: List of AlphaFactor instances
        
        Returns:
            DataFrame with all alphas ranked by metrics
        """
        
        print(f"\n{'='*70}")
        print("TESTING MULTIPLE ALPHAS")
        print(f"{'='*70}")
        
        for alpha in alpha_list:
            self.test_alpha(alpha)
        
        # Create comparison table
        comparison_data = []
        for alpha_name, result in self.results.items():
            metrics = result['metrics'].copy()
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank by excess return
        comparison_df['Rank_by_Return'] = comparison_df['Excess_Return_%'].rank(ascending=False)
        comparison_df['Rank_by_Sharpe'] = comparison_df['Sharpe_Ratio'].rank(ascending=False)
        comparison_df['Rank_by_Sortino'] = comparison_df['Sortino_Ratio'].rank(ascending=False)
        
        print(f"\n{'='*70}")
        print("ALPHA COMPARISON SUMMARY")
        print(f"{'='*70}\n")
        
        # Display key columns
        display_cols = [
            'Alpha_Name',
            'Excess_Return_%',
            'Annual_Return_%',
            'Sharpe_Ratio',
            'Max_Drawdown_%',
            'Win_Rate_%',
            'Num_Trades'
        ]
        
        print(comparison_df[display_cols].to_string(index=False))
        
        return comparison_df
    
    def export_results(self, alpha_name, output_dir='/mnt/user-data/outputs'):
        """Export backtest results to CSV"""
        
        if alpha_name not in self.results:
            print(f"Alpha {alpha_name} not found in results")
            return
        
        backtest_df = self.results[alpha_name]['backtest_df']
        backtest_df.to_csv(f'{output_dir}/backtest_{alpha_name}.csv')
        
        print(f"\nResults exported to: backtest_{alpha_name}.csv")


# ============================================================================
# MAIN USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("ALPHA TESTING FRAMEWORK")
    print("="*70)
    
    # ===== STEP 1: LOAD YOUR DATA =====
    print("\nStep 1: Loading data...")
    
    # Option A: Load from CSV
    csv_file = "your_stock_data.csv"  # Replace with your CSV path
    
    try:
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df = df.set_index('Date')
        
        # Ensure required columns exist
        required_cols = ['Close', 'Volume', 'High', 'Low', 'Open']
        if not all(col in df.columns for col in required_cols):
            print(f"CSV must contain columns: {required_cols}")
            raise ValueError("Missing required columns")
        
        print(f"Loaded {len(df)} rows of data")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    except FileNotFoundError:
        print(f"File {csv_file} not found. Using sample data instead...")
        
        # Generate sample data for demonstration
        dates = pd.date_range('2020-01-01', periods=1000)
        np.random.seed(42)
        
        # Create realistic stock price movement
        price = 100 + np.cumsum(np.random.randn(1000) * 2)
        volume = np.random.randint(1000000, 10000000, 1000)
        
        df = pd.DataFrame({
            'Close': price,
            'Open': price + np.random.randn(1000) * 1,
            'High': price + abs(np.random.randn(1000) * 2),
            'Low': price - abs(np.random.randn(1000) * 2),
            'Volume': volume
        }, index=dates)
    
    # ===== STEP 2: CREATE YOUR ALPHAS =====
    print("\nStep 2: Creating alpha factors...")
    
    alphas_to_test = [
        MomentumAlpha(lookback_period=20),
        MeanReversionAlpha(window=20),
        MacdAlpha(),
        VolumeAlpha(window=20),
        RSIAlpha(),
        # CustomAlpha(),  # Uncomment to test your custom alpha
    ]
    
    print(f"Created {len(alphas_to_test)} alphas to test:")
    for alpha in alphas_to_test:
        print(f"  - {alpha.name}")
    
    # ===== STEP 3: TEST ALPHAS =====
    print("\nStep 3: Testing alphas...")
    
    tester = AlphaTester(df, initial_capital=100000, transaction_cost=0.001)
    
    # Test all alphas and get comparison
    comparison_df = tester.compare_alphas(alphas_to_test)
    
    # ===== STEP 4: DETAILED ANALYSIS OF BEST ALPHA =====
    best_alpha_idx = comparison_df['Excess_Return_%'].idxmax()
    best_alpha_name = comparison_df.loc[best_alpha_idx, 'Alpha_Name']
    
    print(f"\n{'='*70}")
    print(f"BEST PERFORMING ALPHA: {best_alpha_name}")
    print(f"{'='*70}")
    print(comparison_df.loc[best_alpha_idx].to_string())
    
    # ===== STEP 5: EXPORT RESULTS =====
    print("\nStep 5: Exporting results...")
    
    for alpha in alphas_to_test:
        tester.export_results(alpha.name)
    
    # Export comparison
    comparison_df.to_csv('/mnt/user-data/outputs/alpha_comparison.csv', index=False)
    print("Comparison table exported to: alpha_comparison.csv")
    
    print("\n" + "="*70)
    print("ALPHA TESTING COMPLETE")
    print("="*70)
