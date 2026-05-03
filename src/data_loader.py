import yfinance as yf
import pandas as pd

def download_data(ticker="AAPL", period="10y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    
    return df