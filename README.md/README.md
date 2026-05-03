# AI Trading Strategy Analyzer

## Overview

The **AI Trading Strategy Analyzer** is a machine learning–driven system designed to evaluate stock trading strategies using historical market data, predictive modeling, and backtesting.

This project focuses on analyzing **market direction (up/down movement)** rather than predicting exact prices. It integrates data engineering, machine learning, and financial evaluation techniques into an interactive dashboard.

The goal is to explore how machine learning models can assist in **decision-making for trading strategies**, while emphasizing realistic limitations and risk awareness.

---

## Key Features

### Data Pipeline

* Retrieves historical stock data using `yfinance`
* Supports multiple timeframes (1y, 2y, 5y, 10y)
* Handles multiple tickers:

  * AAPL, AMZN, TGT, TSLA, SPY

### Feature Engineering

* Daily returns (`return_1d`)
* Moving averages (`ma_5`, `ma_20`)
* Volatility (`volatility_5`)
* Momentum (`momentum_5`)

### Machine Learning Models

* **Logistic Regression (Baseline Model)**

  * Simple, interpretable
  * Used for comparison and benchmarking

* **XGBoost (Advanced Model)**

  * Gradient boosting algorithm
  * Captures nonlinear relationships
  * Often improves predictive performance on structured data

### Strategy Logic

* Binary prediction:

  * `1` → Enter trade (expect price increase)
  * `0` → Stay out
* Confidence-based filtering:

  * Trades only executed when model confidence exceeds threshold (e.g., 60%)

### Backtesting Engine

* Simulates trading decisions over historical data
* Compares:

  * ML-driven strategy
  * Buy-and-hold baseline
* Calculates cumulative returns over time

### Performance Metrics

* Strategy total return
* Buy-and-hold return
* Trade win rate
* Overall accuracy
* Maximum drawdown

### Visualization Dashboard

* Built with **Streamlit**
* Interactive features:

  * Model selection (Logistic vs XGBoost)
  * Time period selection
  * Confidence threshold adjustment
* Charts powered by **Plotly**

  * Strategy vs Buy-and-Hold comparison
  * Multi-stock ranking visualization

---

## Project Architecture

```
AI-Trading-Analyzer/
│
├── app/
│   └── streamlit_app.py      # Dashboard interface
│
├── src/
│   ├── data_loader.py        # Data retrieval
│   ├── features.py           # Feature engineering
│   ├── model.py              # ML models
│   └── backtest.py           # Strategy evaluation
│
├── data/
│   └── raw/                  # Stored datasets
│
├── images/                   # Screenshots
├── main.py                   # Testing pipeline
├── requirements.txt
└── README.md
```

---

## How It Works

1. **Data Collection**

   * Historical stock data is downloaded via `yfinance`

2. **Feature Engineering**

   * Raw price data is transformed into meaningful indicators

3. **Model Training**

   * Data is split into time-based training and testing sets
   * Selected model (Logistic or XGBoost) is trained

4. **Prediction**

   * Model predicts probability of next-day price increase

5. **Confidence Filtering**

   * Trades are only executed if probability exceeds threshold

6. **Backtesting**

   * Strategy performance is simulated on historical data

7. **Visualization**

   * Results are displayed in an interactive dashboard

---

## Key Insights

* Machine learning can identify patterns, but **does not guarantee profitability**
* Performance varies significantly across different stocks
* XGBoost may outperform simpler models in certain conditions
* Confidence thresholds improve trade quality by filtering weak signals
* Market behavior is complex and difficult to generalize

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/creategit123/AI-Trading-Analyzer.git
cd AI-Trading-Analyzer
```

### 2. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
python -m streamlit run app/streamlit_app.py
```

---

## Screenshots

### Dashboard Overview

Images section

---

## Future Improvements

* Additional technical indicators (RSI, MACD, Bollinger Bands)
* Expanded model comparison (Random Forest, LightGBM)
* Real-time data integration
* Probability calibration improvements
* Trade signal system (research-based alerts)
* Paper trading simulation
* Portfolio-level strategy optimization

---

## Limitations

* Uses only historical price data (no news, sentiment, or macro factors)
* Assumes perfect trade execution (no slippage or fees)
* Models may overfit or fail in changing market conditions
* Not suitable for real-world trading without further validation

---

## Tech Stack

* **Programming Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost
* **Visualization:** Plotly
* **Dashboard:** Streamlit
* **Data Source:** yfinance

---

## Resume Summary

Developed a machine learning–based trading strategy analyzer using Python, XGBoost, and Streamlit to evaluate stock market direction through feature engineering, confidence filtering, and backtesting with interactive visualization.

---

## Disclaimer

This project is for **educational and research purposes only**.

It does not constitute financial advice, and it should not be used to make real trading decisions without proper validation and risk management.

