"""
Main script for AI Trading Analyzer
"""

from src.data_loader import download_data
from src.features import add_features
from src.model import train_model, evaluate_model
from src.backtest import simple_backtest, performance_metrics
from src.visualization import plot_strategy_vs_buy_hold


print("AI Trading Analyzer setup is working.")


def run_pipeline(ticker):
    print(f"\n================ {ticker} ================")

    # 1. Load and prepare data
    df = download_data(ticker)
    df = add_features(df)

    # 2. Save CSV
    csv_path = f"data/raw/{ticker.lower()}.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")

    # 3. Quick dataset checks
    print(df.head())
    print(df.columns)

    print("\nTarget Counts:")
    print(df["target"].value_counts())

    print("\nTarget Percentages:")
    print(df["target"].value_counts(normalize=True) * 100)

    # 4. Select features
    features = ["return_1d", "ma_5", "ma_20", "volatility_5", "momentum_5"]

    X = df[features]
    y = df["target"]

    # 5. Time-based train/test split
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # 6. Train and evaluate model
    model = train_model(X_train, y_train)
    predictions, accuracy, probabilities = evaluate_model(model, X_test, y_test)

    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("First 10 predictions:", predictions[:10])
    print("First 10 actuals:   ", y_test.iloc[:10].values)

    # 7. Backtest
    test_df = df.iloc[split_index:].copy()
    backtest_results = simple_backtest(test_df, predictions, probabilities)

    # 8. Display results in percentage format
    display_results = backtest_results.copy()

    display_results["return_1d"] = display_results["return_1d"] * 100
    display_results["strategy_return"] = display_results["strategy_return"] * 100
    display_results["strategy_growth"] = display_results["strategy_growth"] * 100
    display_results["buy_hold_growth"] = display_results["buy_hold_growth"] * 100
    if "confidence" in display_results.columns:
        display_results["confidence"] = display_results["confidence"] * 100

    print("\nBacktest Preview:")
    print(display_results[
        [
            "Date",
            "Close",
            "return_1d",
            "prediction",
            "confidence",
            "strategy_return",
            "strategy_growth",
            "buy_hold_growth"
        ]
    ].head(10))

    # 9. Performance metrics
    metrics = performance_metrics(backtest_results)

    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value * 100:.2f}%")

    # 10. Visualization
    plot_strategy_vs_buy_hold(backtest_results, ticker=ticker)

    return {
    "Ticker": ticker,
    "Accuracy": accuracy,
    "Strategy Return": metrics["strategy_total_return"],
    "Buy Hold Return": metrics["buy_hold_total_return"],
    "Trade Win Rate": metrics["trade_win_rate"],
    "Max Drawdown": metrics["max_drawdown"]
    }

# 11. Multi stock function
tickers = ["AAPL", "AMZN", "TGT", "TSLA", "SPY"]

import pandas as pd

all_results = []

for ticker in tickers:
    result = run_pipeline(ticker)
    all_results.append(result)

summary_df = pd.DataFrame(all_results)

summary_display = summary_df.copy()

for col in ["Accuracy", "Strategy Return", "Buy Hold Return", "Trade Win Rate", "Max Drawdown"]:
    summary_display[col] = summary_display[col] * 100

print("\n================ SUMMARY TABLE ================\n")
print(summary_display.round(2))

summary_display.to_csv("data/raw/summary_results.csv", index=False)

# 12. Show mutiple charts on screen
import matplotlib.pyplot as plt

plt.show()