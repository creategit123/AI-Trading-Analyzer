"""
Streamlit dashboard for AI Trading Analyzer
------------------------------------------
This app allows users to:
1. Select a stock and time period
2. Train a machine learning model
3. Backtest a trading strategy
4. Visualize performance
5. Compare multiple stocks
"""

import sys
import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Allow imports from src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import project modules
from src.data_loader import download_data
from src.features import add_features
from src.model import train_model, evaluate_model
from src.backtest import simple_backtest, performance_metrics

# Page Configuration
st.set_page_config(page_title="AI Trading Analyzer", layout="wide")

st.title("AI Trading Strategy Analyzer")

st.write(
    "This dashboard uses historical market data, feature engineering, "
    "machine learning, and backtesting to evaluate a simple trading strategy."
)

# Sidebar Controls
st.sidebar.header("Settings")

ticker = st.sidebar.selectbox(
    "Choose a stock",
    ["AAPL", "AMZN", "TGT", "TSLA", "SPY"]
)

period = st.sidebar.selectbox(
    "Choose historical period",
    ["1y", "2y", "5y", "10y"]
)

model_choice = st.sidebar.selectbox(
    "Choose ML model",
    ["logistic", "xgboost"]
)

threshold = st.sidebar.slider(
    "Trade confidence threshold",
    min_value=0.50,
    max_value=0.80,
    value=0.60,
    step=0.01
)

run_button = st.sidebar.button("Run Analysis")
run_all = st.sidebar.button("Run All Stocks Comparison")

# Core Pipeline Function
def run_analysis(selected_ticker, selected_period, selected_model, threshold):

    # 1. Load and prepare data
    df = download_data(selected_ticker, period=selected_period)
    df = add_features(df)

    # 2. Select features for ML model
    features = ["return_1d", "ma_5", "ma_20", "volatility_5", "momentum_5"]

    X = df[features]
    y = df["target"]

    # 3. Time-based split (NO random shuffling)
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # 4. Train model
    model = train_model(X_train, y_train, model_type = selected_model)

    # 5. Evaluate model (predictions + accuracy)
    predictions, accuracy, probabilities = evaluate_model(model, X_test, y_test)

    if probabilities is not None:
        predictions = (probabilities >= threshold).astype(int)
    

    # 6. Backtest strategy using predictions
    test_df = df.iloc[split_index:].copy()
    backtest_results = simple_backtest(test_df, predictions, probabilities)

    # 7. Calculate performance metrics
    metrics = performance_metrics(backtest_results)

    return df, backtest_results, metrics, predictions, accuracy, probabilities

# Single Stock Analysis
if run_button:

    df, backtest_results, metrics, predictions, accuracy, probabilities = run_analysis(
    ticker,
    period,
    model_choice,
    threshold
)

    st.subheader(f"Results for {ticker}")

    # Display key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
    col2.metric("Strategy Return", f"{metrics['strategy_total_return'] * 100:.2f}%")
    col3.metric("Buy & Hold Return", f"{metrics['buy_hold_total_return'] * 100:.2f}%")
    col4.metric("Trade Win Rate", f"{metrics['trade_win_rate'] * 100:.2f}%")
    col5.metric("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")

    # Strategy vs Buy & Hold Chart
    st.subheader("ML Strategy vs Buy and Hold")

    display_results = backtest_results.copy()

    # Convert to percentage format
    display_results["strategy_growth"] *= 100
    display_results["buy_hold_growth"] *= 100

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=display_results["Date"],
            y=display_results["strategy_growth"],
            mode="lines",
            name="ML Strategy"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=display_results["Date"],
            y=display_results["buy_hold_growth"],
            mode="lines",
            name="Buy and Hold"
        )
    )

    fig.update_layout(
        title=f"{ticker}: ML Strategy vs Buy and Hold",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (%)",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Backtest Table
    st.subheader("Backtest Preview")

    table = backtest_results.copy()

    # Convert values to percentage format
    table["return_1d"] *= 100
    table["strategy_return"] *= 100
    table["strategy_growth"] *= 100
    table["buy_hold_growth"] *= 100

    st.dataframe(
        table[
            [
                "Date",
                "Close",
                "return_1d",
                "prediction",
                "strategy_return",
                "strategy_growth",
                "buy_hold_growth"
            ]
        ].head(20),
        use_container_width=True
    )

    # Prediction Distribution
    st.subheader("Prediction Distribution")

    pred_counts = pd.Series(predictions).value_counts().rename(
        index={0: "Stay Out", 1: "Enter Trade"}
    )

    st.bar_chart(pred_counts)

# Multi-Stock Comparison
elif run_all:

    st.subheader("Multi-Stock Comparison")

    tickers = ["AAPL", "AMZN", "TGT", "TSLA", "SPY"]
    results_list = []

    for current_ticker in tickers:

        df, backtest_results, metrics, predictions, accuracy, probabilities = run_analysis(
            current_ticker,
            period,
            model_choice,
            threshold
)

        # Store results for comparison
        results_list.append({
            "Ticker": current_ticker,
            "Accuracy (%)": accuracy * 100,
            "Strategy Return (%)": metrics["strategy_total_return"] * 100,
            "Buy & Hold (%)": metrics["buy_hold_total_return"] * 100,
            "Trade Win Rate (%)": metrics["trade_win_rate"] * 100,
            "Max Drawdown (%)": metrics["max_drawdown"] * 100
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    # Sort best to worst (using strategy return)
    results_df = results_df.sort_values(
        by="Strategy Return (%)",
        ascending=False
    )

    # Fix index issue (numbered rankings properly)
    results_df = results_df.reset_index(drop=True)

    st.dataframe(results_df.round(2), use_container_width=True)

    # Ranking Chart (display chart)
    st.subheader("Strategy Return Ranking")

    ranking_fig = go.Figure()

    ranking_fig.add_trace(
        go.Bar(
            x=results_df["Ticker"],
            y=results_df["Strategy Return (%)"],
            name="Strategy Return (%)"
        )
    )

    ranking_fig.update_layout(
        title="Strategy Return by Stock",
        xaxis_title="Ticker",
        yaxis_title="Strategy Return (%)",
        height=450
    )

    st.plotly_chart(ranking_fig, use_container_width=True)

# Default message
else:
    st.info("Choose a stock and click 'Run Analysis', or run the multi-stock comparison.")