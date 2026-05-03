import numpy as np

def simple_backtest(test_df, predictions, probabilities=None):
    results = test_df.copy()

    results["prediction"] = predictions

    if probabilities is not None:
        results["confidence"] = probabilities
    

    # Strategy return:
    # If prediction is 1, take the next day's return.
    # If prediction is 0, stay out and earn 0.
    results["strategy_return"] = results["prediction"] * results["return_1d"]

    # Buy-and-hold return:
    # This means staying invested every day.
    results["buy_hold_return"] = results["return_1d"]

    # Cumulative returns
    results["strategy_growth"] = (1 + results["strategy_return"]).cumprod()
    results["buy_hold_growth"] = (1 + results["buy_hold_return"]).cumprod()

    return results

def performance_metrics(results):
    
    # Total return
    strategy_total = results["strategy_growth"].iloc[-1] - 1
    buy_hold_total = results["buy_hold_growth"].iloc[-1] - 1

    # Win rate (only when in trade)
    trades = results[results["prediction"] == 1]

    if len(trades) > 0:
        trade_win_rate = (trades["strategy_return"] > 0).mean()
    else:
        trade_win_rate = 0

    # Max drawdown
    cumulative = results["strategy_growth"]
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    return {
    "strategy_total_return": strategy_total,
    "buy_hold_total_return": buy_hold_total,
    "trade_win_rate": trade_win_rate,
    "max_drawdown": max_drawdown
}
