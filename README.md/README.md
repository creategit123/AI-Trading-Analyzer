# AI Trading Strategy Analyzer

## Overview

This project is an AI-driven trading strategy analyzer that uses machine learning and historical market data to evaluate stock trading strategies.

It compares different models, applies confidence-based decision making, and evaluates performance using backtesting techniques.

## Features

* Historical stock data pipeline (yfinance)
* Feature engineering (returns, moving averages, volatility, momentum)
* Machine learning models:

  * Logistic Regression (baseline)
  * XGBoost (advanced)
* Confidence threshold filtering for trade decisions
* Backtesting engine (strategy vs buy-and-hold)
* Multi-stock comparison (AAPL, AMZN, TGT, TSLA, SPY)
* Interactive Streamlit dashboard
* Performance metrics:

  * Strategy return
  * Buy & hold return
  * Trade win rate
  * Max drawdown

## Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Plotly

## How It Works

1. Download historical stock data
2. Engineer features
3. Train ML model
4. Predict next-day direction
5. Apply confidence threshold
6. Backtest trading strategy
7. Visualize results in dashboard

## Key Insights

* Advanced models (XGBoost) may improve performance on certain assets
* Results vary across stocks due to market complexity
* Confidence thresholds help filter weaker trades
* ML does not guarantee profitable trading outcomes

## How to Run

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ai-trading-analyzer.git
cd ai-trading-analyzer
```

Create virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run dashboard:

```bash
python -m streamlit run app/streamlit_app.py
```

## Screenshots

![Dashboard](images/Screenshot 2026-05-02 225452.png)

## Future Improvements

* More technical indicators (RSI, MACD)
* Model comparison expansion
* Real-time data integration
* Trade signal generation
* Paper trading simulation

## Disclaimer

This project is for educational purposes only. It is not financial advice.
