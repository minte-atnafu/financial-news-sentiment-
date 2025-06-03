import os
import pandas as pd
from finta import TA
import matplotlib.pyplot as plt

def load_stock_data(file_path):
    """Load stock price data from CSV."""
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    return df

def calculate_technical_indicators(df):
    """Calculate technical indicators using FinTA."""
    df['SMA_20'] = TA.SMA(df, period=20)
    df['EMA_20'] = TA.EMA(df, period=20)
    df['RSI'] = TA.RSI(df, period=14)
    
    macd = TA.MACD(df)
    df['MACD'] = macd['MACD']
    df['MACD_Signal'] = macd['SIGNAL']
    df['MACD_Hist'] = macd['MACD'] - macd['SIGNAL']
    
    return df

def visualize_indicators(df, stock_symbol):
    """Visualize technical indicators and save plot as PNG."""
    plt.figure(figsize=(14, 10))

    # Price and MAs
    plt.subplot(3, 1, 1)
    plt.plot(df['Date'], df['Close'], label='Close Price')
    plt.plot(df['Date'], df['SMA_20'], label='SMA 20')
    plt.plot(df['Date'], df['EMA_20'], label='EMA 20')
    plt.title(f'{stock_symbol} Stock Price and Moving Averages')
    plt.legend()

    # RSI
    plt.subplot(3, 1, 2)
    plt.plot(df['Date'], df['RSI'], label='RSI')
    plt.axhline(70, linestyle='--', color='r', alpha=0.5)
    plt.axhline(30, linestyle='--', color='g', alpha=0.5)
    plt.title('Relative Strength Index (RSI)')
    plt.legend()

    # MACD
    plt.subplot(3, 1, 3)
    plt.plot(df['Date'], df['MACD'], label='MACD')
    plt.plot(df['Date'], df['MACD_Signal'], label='Signal Line')
    plt.bar(df['Date'], df['MACD_Hist'], label='MACD Histogram', alpha=0.5)
    plt.title('MACD')
    plt.legend()

    plt.tight_layout()
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{stock_symbol}_indicators.png')
    plt.close()

if __name__ == "__main__":
    # Get the script's directory (where the .py file is)
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    
    # Look for data in a sibling folder (../yfinance_data from script location)
    data_folder = r'C:\Users\mintesinot\financial-news-analysis\yfinance_data'  # Normalize path
    
    if not os.path.exists(data_folder):
        print(f"❌ Data folder not found at: {data_folder}")
        print("Please ensure:")
        print("1. The 'yfinance_data' folder exists")
        print("2. It contains your stock CSV files")
        print("3. It's in the correct location relative to your script")
    else:
        print(f"✅ Found data folder at: {data_folder}")
        for filename in os.listdir(data_folder):
            if filename.endswith('.csv'):
                file_path = os.path.join(data_folder, filename)
                stock_symbol = filename.replace('.csv', '')
                try:
                    print(f"📈 Processing {stock_symbol}...")
                    df = load_stock_data(file_path)
                    df = calculate_technical_indicators(df)
                    visualize_indicators(df, stock_symbol)
                    print(f"✅ Done: outputs/{stock_symbol}_indicators.png")
                except Exception as e:
                    print(f"❌ Error processing {stock_symbol}: {str(e)}")
