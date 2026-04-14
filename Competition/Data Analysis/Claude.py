"""
Stock Market Data Analysis Guide - Quant Invitational
======================================================
Comprehensive examples using pandas, numpy, and common quant analysis techniques
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 1. LOADING AND EXPLORING DATA
# ============================================================================

def load_and_explore(csv_path):
    """Load CSV and get initial insights"""
    
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Basic exploration
    print("Dataset Shape:", df.shape)  # rows, columns
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nBasic statistics:")
    print(df.describe())
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    return df


# ============================================================================
# 2. DATA CLEANING & PREPARATION
# ============================================================================

def clean_stock_data(df):
    """Prepare data for analysis"""
    
    df = df.copy()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    # Option 1: Drop rows with any missing values
    # df = df.dropna()
    
    # Option 2: Fill missing values (forward fill for time series)
    # df = df.fillna(method='ffill')
    
    # Option 3: Fill with specific value
    # df['Volume'] = df['Volume'].fillna(0)
    
    # Convert date columns to datetime if present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Convert price/volume columns to numeric
    price_cols = ['Open', 'High', 'Low', 'Close', 'Price']
    for col in price_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by date if available
    if 'Date' in df.columns:
        df = df.sort_values('Date')
    
    return df


# ============================================================================
# 3. FUNDAMENTAL ANALYSIS
# ============================================================================

def calculate_fundamental_metrics(df):
    """Calculate key fundamental metrics"""
    
    df = df.copy()
    
    # Price-to-Earnings Ratio (if P/E data available)
    if 'Price' in df.columns and 'EPS' in df.columns:
        df['PE_Ratio'] = df['Price'] / df['EPS']
    
    # Price-to-Book Ratio
    if 'Price' in df.columns and 'Book_Value'] in df.columns:
        df['PB_Ratio'] = df['Price'] / df['Book_Value']
    
    # Dividend Yield
    if 'Annual_Dividend' in df.columns and 'Price' in df.columns:
        df['Dividend_Yield'] = (df['Annual_Dividend'] / df['Price']) * 100
    
    # Market Cap (if shares outstanding available)
    if 'Price' in df.columns and 'Shares_Outstanding' in df.columns:
        df['Market_Cap'] = df['Price'] * df['Shares_Outstanding']
    
    return df


# ============================================================================
# 4. TECHNICAL ANALYSIS
# ============================================================================

def calculate_technical_indicators(df):
    """Calculate common technical indicators"""
    
    df = df.copy()
    
    # Moving Averages
    if 'Close' in df.columns:
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # Relative Strength Index (RSI)
    if 'Close' in df.columns:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    if 'Close' in df.columns:
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    if 'Close' in df.columns:
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (std * 2)
    
    # Volume-weighted moving average
    if 'Close' in df.columns and 'Volume' in df.columns:
        df['VWAP'] = (df['Close'] * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()
    
    return df


# ============================================================================
# 5. RETURNS & VOLATILITY ANALYSIS
# ============================================================================

def calculate_returns_volatility(df):
    """Calculate returns and risk metrics"""
    
    df = df.copy()
    
    if 'Close' in df.columns:
        # Daily Returns (percentage)
        df['Daily_Return'] = df['Close'].pct_change() * 100
        
        # Cumulative Returns
        df['Cumulative_Return'] = (1 + df['Daily_Return']/100).cumprod() - 1
        
        # Rolling Volatility (standard deviation of returns)
        df['Volatility_20d'] = df['Daily_Return'].rolling(window=20).std()
        df['Volatility_60d'] = df['Daily_Return'].rolling(window=60).std()
        
        # Sharpe Ratio (annualized, assuming 0% risk-free rate)
        risk_free_rate = 0
        df['Sharpe_Ratio'] = (df['Daily_Return'] - risk_free_rate) / df['Volatility_20d']
        
    return df


# ============================================================================
# 6. COMPARATIVE ANALYSIS & GROUPING
# ============================================================================

def compare_stocks(df):
    """Compare performance across multiple stocks"""
    
    if 'Ticker' in df.columns or 'Symbol' in df.columns:
        ticker_col = 'Ticker' if 'Ticker' in df.columns else 'Symbol'
        
        # Group by stock
        grouped = df.groupby(ticker_col)
        
        # Summary statistics by stock
        summary = grouped['Close'].agg([
            ('Latest_Price', 'last'),
            ('Avg_Price', 'mean'),
            ('Max_Price', 'max'),
            ('Min_Price', 'min'),
            ('Std_Dev', 'std')
        ])
        
        print("Stock Comparison:")
        print(summary)
        
        return summary


# ============================================================================
# 7. SCREENING & FILTERING
# ============================================================================

def screen_stocks(df, criteria):
    """Filter stocks based on investment criteria"""
    
    screened = df.copy()
    
    # Example: Value stocks (low P/E, low P/B)
    if 'PE_Ratio' in screened.columns and 'PB_Ratio' in screened.columns:
        value_stocks = screened[
            (screened['PE_Ratio'] < 15) & 
            (screened['PB_Ratio'] < 1.5)
        ]
        print(f"Value stocks found: {len(value_stocks)}")
    
    # Example: Momentum stocks (positive recent returns, rising MA)
    if 'Daily_Return' in screened.columns and 'MA_20' in screened.columns:
        momentum_stocks = screened[
            (screened['Daily_Return'] > 1) & 
            (screened['Close'] > screened['MA_20'])
        ]
        print(f"Momentum stocks found: {len(momentum_stocks)}")
    
    # Example: Oversold stocks (RSI < 30)
    if 'RSI' in screened.columns:
        oversold = screened[screened['RSI'] < 30]
        print(f"Oversold stocks: {len(oversold)}")
    
    return screened


# ============================================================================
# 8. CORRELATION & RELATIONSHIP ANALYSIS
# ============================================================================

def analyze_correlations(df):
    """Analyze relationships between variables"""
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Correlation matrix
    corr_matrix = numeric_df.corr()
    
    print("Correlation Matrix:")
    print(corr_matrix)
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    print("\nHighly correlated pairs (|r| > 0.7):")
    for pair in high_corr_pairs:
        print(f"{pair['var1']} <-> {pair['var2']}: {pair['correlation']:.3f}")
    
    return corr_matrix


# ============================================================================
# 9. RANKING & SCORING
# ============================================================================

def score_stocks(df):
    """Create composite scores for ranking"""
    
    df = df.copy()
    
    # Normalize metrics to 0-100 scale
    def normalize(series, ascending=True):
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series([50] * len(series), index=series.index)
        normalized = ((series - min_val) / (max_val - min_val)) * 100
        return normalized if ascending else 100 - normalized
    
    # Create component scores (customize based on your metrics)
    if 'PE_Ratio' in df.columns:
        df['Value_Score'] = normalize(df['PE_Ratio'], ascending=False)  # Lower is better
    
    if 'Daily_Return' in df.columns:
        df['Momentum_Score'] = normalize(df['Daily_Return'], ascending=True)
    
    if 'Volatility_20d' in df.columns:
        df['Stability_Score'] = normalize(df['Volatility_20d'], ascending=False)  # Lower volatility is better
    
    # Composite score (equal weighting - adjust as needed)
    score_cols = [col for col in df.columns if 'Score' in col]
    if score_cols:
        df['Composite_Score'] = df[score_cols].mean(axis=1)
        
        # Rank stocks
        df['Rank'] = df['Composite_Score'].rank(ascending=False)
        
        print("\nTop 10 ranked stocks:")
        print(df[score_cols + ['Composite_Score', 'Rank']].nlargest(10, 'Composite_Score'))
    
    return df


# ============================================================================
# 10. VISUALIZATION
# ============================================================================

def create_visualizations(df):
    """Create useful charts for analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Price trend
    if 'Close' in df.columns:
        axes[0, 0].plot(df.index, df['Close'], label='Close Price')
        if 'MA_20' in df.columns:
            axes[0, 0].plot(df.index, df['MA_20'], label='MA 20', alpha=0.7)
        axes[0, 0].set_title('Price Trend')
        axes[0, 0].legend()
    
    # Returns distribution
    if 'Daily_Return' in df.columns:
        axes[0, 1].hist(df['Daily_Return'].dropna(), bins=50, edgecolor='black')
        axes[0, 1].set_title('Daily Returns Distribution')
        axes[0, 1].set_xlabel('Return %')
    
    # Volume
    if 'Volume' in df.columns:
        axes[1, 0].bar(df.index, df['Volume'], alpha=0.7)
        axes[1, 0].set_title('Trading Volume')
    
    # RSI
    if 'RSI' in df.columns:
        axes[1, 1].plot(df.index, df['RSI'], label='RSI', color='purple')
        axes[1, 1].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        axes[1, 1].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
        axes[1, 1].set_title('RSI Indicator')
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/stock_analysis_charts.png', dpi=300, bbox_inches='tight')
    print("Charts saved to stock_analysis_charts.png")
    plt.close()


# ============================================================================
# MAIN WORKFLOW - Example Usage
# ============================================================================

if __name__ == "__main__":
    # Load your CSV file
    csv_file = "your_stock_data.csv"  # Replace with your actual file
    
    # 1. Load and explore
    df = load_and_explore(csv_file)
    
    # 2. Clean data
    df = clean_stock_data(df)
    
    # 3. Calculate metrics
    df = calculate_fundamental_metrics(df)
    df = calculate_technical_indicators(df)
    df = calculate_returns_volatility(df)
    
    # 4. Analysis
    compare_stocks(df)
    analyze_correlations(df)
    screen_stocks(df, {})
    df = score_stocks(df)
    
    # 5. Visualize
    create_visualizations(df)
    
    # 6. Export results
    df.to_csv('/mnt/user-data/outputs/analyzed_stock_data.csv', index=False)
    print("\nAnalyzed data saved to analyzed_stock_data.csv")