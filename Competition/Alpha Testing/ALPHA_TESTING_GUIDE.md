# Alpha Testing Framework - Complete Guide

## Overview

This framework allows you to:
1. **Load historical stock data** from CSV
2. **Define custom alpha factors** with your own trading logic
3. **Backtest alphas** on historical data
4. **Compare performance** across multiple alphas
5. **Analyze detailed metrics** to understand what works

---

## File Overview

### 1. `alpha_testing_framework.py`
**The main workhorse** - Python script for backtesting alphas
- Load CSV data
- Define alpha factors by inheriting `AlphaFactor` class
- Run backtests with `AlphaTester`
- Compare alphas automatically
- Calculate detailed performance metrics

### 2. `alpha_testing_dashboard.jsx`
**Interactive UI** - React dashboard for testing alphas
- Upload CSV files
- Manage alpha factors
- Run backtests visually
- Compare alphas in charts
- View detailed metrics

---

## Quick Start (5 minutes)

### Step 1: Prepare Your Data
Create a CSV file with these columns:
```
Date,Open,High,Low,Close,Volume
2023-01-01,100.00,101.50,99.50,100.50,1000000
2023-01-02,100.50,102.00,100.00,101.00,1200000
...
```

**Required columns:** `Date`, `Close`, `Open`, `High`, `Low`, `Volume`

### Step 2: Run the Framework
```bash
python alpha_testing_framework.py
```

The script will:
1. Load your CSV data
2. Test built-in alphas (Momentum, Mean Reversion, MACD, RSI, Volume)
3. Compare performance
4. Export results to CSV

### Step 3: View Results
Check these output files:
- `alpha_comparison.csv` - Summary table
- `backtest_AlphaName.csv` - Detailed results for each alpha

---

## Creating Custom Alphas

### Understanding the AlphaFactor Class

All alphas inherit from `AlphaFactor`. You only need to implement one method:

```python
from alpha_testing_framework import AlphaFactor
import pandas as pd
import numpy as np

class MyCustomAlpha(AlphaFactor):
    def __init__(self):
        super().__init__("My_Alpha_Name")
    
    def calculate(self, df):
        """
        Args:
            df: DataFrame with OHLCV data
               Columns: Close, Open, High, Low, Volume
        
        Returns:
            pandas Series with signals:
            - 1 or positive: Buy signal (expect price up)
            - 0 or neutral: No signal
            - -1 or negative: Sell signal (expect price down)
        """
        
        # Your logic here
        close = df['Close']
        volume = df['Volume']
        
        # Example: Buy when volume is high
        avg_volume = volume.rolling(20).mean()
        signals = np.where(volume > avg_volume * 1.5, 1, -1)
        
        return pd.Series(signals, index=df.index)
```

### Alpha Signal Interpretation

**What your alpha should return:**
- **+1 or positive values** = "BUY" signal (stock will go up)
- **0 or neutral** = "HOLD" signal (no opinion)
- **-1 or negative values** = "SELL" signal (stock will go down)

```python
# Example: All 1's means always be long
signals = np.ones(len(df))

# Example: Buy/Sell based on condition
signals = np.where(condition, 1, -1)

# Example: More sophisticated scoring
signals = np.where(score > 0.5, 1, np.where(score < -0.5, -1, 0))
```

### Real Alpha Examples

#### Example 1: Simple Moving Average Crossover
```python
class MASignalAlpha(AlphaFactor):
    def __init__(self, fast=20, slow=50):
        super().__init__(f"MA_{fast}_{slow}")
        self.fast = fast
        self.slow = slow
    
    def calculate(self, df):
        close = df['Close']
        ma_fast = close.rolling(self.fast).mean()
        ma_slow = close.rolling(self.slow).mean()
        
        # Buy when fast MA > slow MA
        signals = np.where(ma_fast > ma_slow, 1, -1)
        return pd.Series(signals, index=df.index)
```

#### Example 2: Volatility Mean Reversion
```python
class VolatilityAlpha(AlphaFactor):
    def __init__(self):
        super().__init__("Volatility_MR")
    
    def calculate(self, df):
        returns = df['Close'].pct_change()
        volatility = returns.rolling(20).std()
        avg_vol = volatility.rolling(60).mean()
        
        # Buy when volatility is low (mean reversion expected)
        # Sell when volatility is high (expect reversion)
        signals = np.where(volatility < avg_vol * 0.8, 1, -1)
        return pd.Series(signals, index=df.index)
```

#### Example 3: Combining Multiple Signals (Ensemble)
```python
class EnsembleAlpha(AlphaFactor):
    def __init__(self):
        super().__init__("Ensemble")
    
    def calculate(self, df):
        close = df['Close']
        volume = df['Volume']
        
        # Signal 1: Price trend
        ma_20 = close.rolling(20).mean()
        trend_signal = np.where(close > ma_20, 1, -1)
        
        # Signal 2: Volume strength
        avg_vol = volume.rolling(20).mean()
        volume_signal = np.where(volume > avg_vol, 1, -1)
        
        # Signal 3: Momentum
        returns = close.pct_change(5)
        momentum_signal = np.where(returns > 0, 1, -1)
        
        # Combine: require agreement from 2 out of 3 signals
        combined = trend_signal + volume_signal + momentum_signal
        signals = np.where(combined >= 2, 1, np.where(combined <= -2, -1, 0))
        
        return pd.Series(signals, index=df.index)
```

---

## How Testing Works

### The Backtest Process

```
1. Load historical data
2. For each alpha:
   a) Calculate signals for each day (1, 0, or -1)
   b) Calculate daily returns
   c) Apply strategy returns = signal * daily_return
   d) Subtract transaction costs
   e) Calculate cumulative portfolio value
3. Calculate performance metrics
4. Compare across all alphas
```

### Key Metrics Explained

| Metric | What It Means | Why It Matters |
|--------|---------------|----------------|
| **Total Return %** | Overall profit from start to end | Bottom line performance |
| **Annual Return %** | Annualized return (extrapolated) | Standardized comparison |
| **Volatility %** | Standard deviation of returns | Risk/variability |
| **Sharpe Ratio** | Return per unit of risk | Best risk-adjusted metric |
| **Sortino Ratio** | Return per unit of downside risk | Ignores upside volatility |
| **Max Drawdown %** | Largest peak-to-trough loss | Worst-case scenario |
| **Calmar Ratio** | Annual return / max drawdown | Return vs worst loss |
| **Win Rate %** | % of days with positive return | Consistency |
| **Num Trades** | How many buy/sell signals | Trading frequency |

### Interpreting Results

**Good Alpha Properties:**
```
✓ High Sharpe Ratio (>1.0 is good, >2.0 is excellent)
✓ Low Max Drawdown (less than 20% is ideal)
✓ Positive excess return vs Buy & Hold
✓ High win rate (>50%)
✓ Low transaction costs (reasonable # of trades)
✓ Consistent returns (low volatility)
```

**Red Flags:**
```
✗ Sharpe Ratio < 0 (worse than cash)
✗ Max Drawdown > 50% (too risky)
✗ Negative returns in backtest (loss of money)
✗ Thousands of trades (overfitting to noise)
✗ Works only in specific market regimes (not robust)
```

---

## Using the Dashboard

### 1. Upload Data
- Click "Upload CSV" and select your stock data file
- Ensure columns: Date, Open, High, Low, Close, Volume

### 2. Manage Alphas
- **Toggle** alphas on/off with checkboxes
- **Delete** alphas with ✕ button
- **Add custom alphas** using the text input

### 3. Run Backtest
- Click "Run Backtest" button
- Wait for results to appear (may take a moment)

### 4. View Results
- **Compare tab**: Side-by-side table of all metrics
- **Detailed tab**: Deep dive into selected alpha
- **Performance tab**: Charts comparing all alphas

---

## Advanced Usage

### Testing in Python Script

```python
from alpha_testing_framework import (
    AlphaTester, MomentumAlpha, CustomAlpha
)
import pandas as pd

# Load your data
df = pd.read_csv('stock_data.csv', index_col='Date', parse_dates=True)

# Create tester
tester = AlphaTester(
    df, 
    initial_capital=100000,
    transaction_cost=0.001  # 0.1% per trade
)

# Test multiple alphas
alphas = [
    MomentumAlpha(lookback_period=20),
    CustomAlpha()
]

comparison = tester.compare_alphas(alphas)

# Get detailed results
backtest_df, metrics = tester.test_alpha(MomentumAlpha(30))

# Access specific metrics
print(f"Sharpe Ratio: {metrics['Sharpe_Ratio']}")
print(f"Max Drawdown: {metrics['Max_Drawdown_%']}%")

# Export
tester.export_results('Momentum_20d')
```

### Parameter Optimization

Test alphas with different parameters:

```python
from alpha_testing_framework import AlphaTester, MomentumAlpha

df = pd.read_csv('stock_data.csv', index_col='Date', parse_dates=True)
tester = AlphaTester(df)

# Test different lookback periods
results = {}
for lookback in [5, 10, 15, 20, 30, 50]:
    alpha = MomentumAlpha(lookback_period=lookback)
    _, metrics = tester.test_alpha(alpha)
    results[lookback] = metrics['Sharpe_Ratio']

# Find best parameter
best_lookback = max(results, key=results.get)
print(f"Best lookback: {best_lookback} days")
```

### Walk-Forward Analysis (Advanced)

Test alpha on different time periods:

```python
# Split data into train/test periods
train_df = df['2020':'2022']
test_df = df['2023':]

# Train on one period, test on another
tester_train = AlphaTester(train_df)
tester_test = AlphaTester(test_df)

alpha = MyAlpha()
tester_train.test_alpha(alpha)  # Learn on historical data
tester_test.test_alpha(alpha)   # Verify on fresh data
```

---

## Common Mistakes

### ❌ Data Snooping (Overfitting)
```python
# BAD: Tuning alpha specifically to test data
# This will fail in real trading!
alpha = TunedAlpha(parameter=123)  # Optimized to backtest data
```

**Solution:** 
- Use train/test split
- Test on out-of-sample data
- Keep alphas simple and robust

### ❌ Ignoring Transaction Costs
```python
# BAD: Trading thousands of times per day
if close[today] != close[yesterday]:
    trade()  # Too many trades!
```

**Solution:**
- Set realistic transaction costs (0.1% - 0.5%)
- Filter out noise signals
- Add minimum holding periods

### ❌ Look-Ahead Bias
```python
# BAD: Using future data to make today's decision
signal = close[tomorrow] > close[today]  # CHEATING!
```

**Solution:**
- Only use data up to current time
- Framework handles this automatically

### ❌ Unrealistic Assumptions
```python
# BAD: Assuming perfect fills at exact prices
# Real trading has slippage!
```

**Solution:**
- Add slippage estimation (0.1% - 0.5%)
- Use conservative projections
- Paper trade first

---

## Testing Checklist

Before deploying an alpha, verify:

- [ ] Backtest period >= 2 years of data
- [ ] Sharpe Ratio > 1.0
- [ ] Max Drawdown < 30%
- [ ] Win Rate > 50%
- [ ] Works across different market periods (bull/bear)
- [ ] Reasonable number of trades
- [ ] Transaction costs subtracted from returns
- [ ] Out-of-sample tested (not just backtested)
- [ ] No look-ahead bias
- [ ] Economically rational explanation

---

## Troubleshooting

### "Missing required columns"
**Fix:** Ensure CSV has: Date, Open, High, Low, Close, Volume

### "All signals are 0"
**Fix:** Your alpha logic is returning 0 for all rows
```python
# Debug: print intermediate values
print(df.head())
print(signals.unique())  # Should show mix of -1, 0, 1
```

### "Negative Sharpe Ratio"
**Fix:** Alpha is worse than just holding cash
- Reconsider the alpha logic
- Check for look-ahead bias
- Increase signal frequency

### "Too many/few trades"
**Fix:** Adjust signal frequency
```python
# Too many trades? Use longer lookback
alpha = MomentumAlpha(lookback_period=60)  # Instead of 5

# Too few trades? Use shorter lookback
alpha = MomentumAlpha(lookback_period=5)  # Instead of 60
```

---

## Next Steps

1. **Create your first alpha** - Copy template and modify
2. **Backtest on your data** - Run framework on CSV
3. **Analyze results** - Check metrics and compare
4. **Iterate** - Refine based on results
5. **Paper trade** - Test in real-time with fake money
6. **Deploy** - Once confident in live trading

Good luck! 🚀
