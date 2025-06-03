import os
import pandas as pd
import numpy as np
from textblob import TextBlob
from scipy.stats import pearsonr

def load_news_data(news_path):
    """Load and preprocess news data."""
    news_df = pd.read_csv(news_path)

    # Convert 'date' column to datetime, coercing errors to NaT (missing)
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')

    # Drop rows where 'date' couldn't be parsed (NaT values)
    news_df = news_df.dropna(subset=['date'])

    # Convert datetime to date (removing time part)
    news_df['date'] = news_df['date'].dt.date

    return news_df


def load_stock_data(stock_path):
    """Load and preprocess individual stock data."""
    df = pd.read_csv(stock_path)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df['Daily_Return'] = df['Close'].pct_change() * 100
    return df

def perform_sentiment_analysis(news_df):
    """Perform sentiment analysis on news headlines."""
    news_df['sentiment'] = news_df['headline'].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
    return news_df

def correlation_analysis(news_df, stock_df, stock_symbol):
    """Calculate correlation between sentiment and stock returns."""
    # Filter news for current stock
    news_for_stock = news_df[news_df['stock'] == stock_symbol]
    
    # Aggregate sentiment by date
    daily_sentiment = news_for_stock.groupby('date')['sentiment'].mean().reset_index()

    # Merge with stock data
    merged_df = pd.merge(daily_sentiment, stock_df, left_on='date', right_on='Date')

    if len(merged_df) < 2:
        print(f"⚠️ Not enough data to compute correlation for {stock_symbol}")
        return None, None, None

    # Correlation
    correlation, p_value = pearsonr(merged_df['sentiment'], merged_df['Daily_Return'].fillna(0))

    print(f"📊 {stock_symbol}: Pearson Correlation = {correlation:.4f}, P-value = {p_value:.4f}")
    return merged_df, correlation, p_value

if __name__ == "__main__":
    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    news_path = os.path.join(script_dir, 'data', 'raw_analyst_ratings.csv')
    stock_folder = os.path.join(script_dir, 'yfinance_data')

    news_df = load_news_data(news_path)
    news_df = perform_sentiment_analysis(news_df)

    # Loop through each stock file
    for filename in os.listdir(stock_folder):
        if filename.endswith('.csv'):
            stock_symbol = filename.replace('.csv', '')
            stock_path = os.path.join(stock_folder, filename)

            try:
                stock_df = load_stock_data(stock_path)
                merged_df, corr, pval = correlation_analysis(news_df, stock_df, stock_symbol)
            except Exception as e:
                print(f"❌ Error processing {stock_symbol}: {str(e)}")
