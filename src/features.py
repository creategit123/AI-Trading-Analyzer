import pandas as pd

def add_features(df):
    df = df.copy()
    df["return_1d"] = df["Close"].pct_change()
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["volatility_5"] = df["return_1d"].rolling(5).std()
    df["momentum_5"] = df["Close"] - df["Close"].shift(5)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return df.dropna()