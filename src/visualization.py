import matplotlib.pyplot as plt


def plot_strategy_vs_buy_hold(results, ticker="AAPL"):
    display_results = results.copy()

    display_results["strategy_growth"] = display_results["strategy_growth"] * 100
    display_results["buy_hold_growth"] = display_results["buy_hold_growth"] * 100

    plt.figure(figsize=(10, 6))

    plt.plot(display_results["Date"], display_results["strategy_growth"], label="ML Strategy")
    plt.plot(display_results["Date"], display_results["buy_hold_growth"], label="Buy and Hold")

    plt.title(f"{ticker}: ML Strategy vs Buy and Hold")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (%)")
    plt.legend()
    plt.grid(True)

    plt.savefig(f"data/raw/{ticker.lower()}_chart.png")
