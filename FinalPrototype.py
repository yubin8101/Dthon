#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings
from joblib import Parallel, delayed
import datetime as dt
import requests
import time
import random

warnings.filterwarnings("ignore")

try:
    plt.style.use("seaborn-v0_8-darkgrid")
except Exception:
    try:
        plt.style.use("seaborn-darkgrid")
    except Exception:
        plt.style.use("ggplot")


def calculate_metrics(df, use_net=True):
    value_col = "Net_Value" if use_net else "Strategy_Value"
    return_col = "Net_Return" if use_net else "Daily_Return"
    total_return = df[value_col].iloc[-1] - 1.0
    daily_returns = df[return_col].dropna()
    if daily_returns.std() != 0 and len(daily_returns) > 1:
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0
    cumulative = df[value_col].copy()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    trades = df["Signal"].diff().fillna(0)
    num_trades = trades.abs().sum() / 2
    trade_frequency = num_trades / len(df) * 100
    df["Trade_Return"] = df[return_col] * (df["Position"] != 0)
    wins = df[df["Trade_Return"] > 0]["Trade_Return"].count()
    total_trades_for_wr = df[df["Position"] != 0]["Trade_Return"].count()
    win_rate = wins / total_trades_for_wr if total_trades_for_wr > 0 else 0
    profits = df[df["Trade_Return"] > 0]["Trade_Return"].sum()
    losses = abs(df[df["Trade_Return"] < 0]["Trade_Return"].sum())
    profit_factor = profits / losses if losses != 0 else float('inf')
    avg_profit_per_trade = df["Trade_Return"].sum() / num_trades if num_trades > 0 else 0
    meets_trade_frequency = trade_frequency >= 3.0
    metrics = {
        "total_return": total_return,
        "annualized_return": (1 + total_return) ** (252 / len(df)) - 1,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": num_trades,
        "trade_frequency": trade_frequency,
        "avg_profit_per_trade": avg_profit_per_trade,
        "meets_trade_frequency": meets_trade_frequency,
    }
    return metrics


def run_strategy(df, n_components=2, short_ma=10, long_ma=30, volatility_window=10,
                 regime_filter_threshold=0.2, fee_rate=0.0006, random_state=42):
    df = df.copy()
    df["LogRet"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility"] = df["LogRet"].rolling(volatility_window).std()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    df[f"SMA_{short_ma}"] = df["Close"].rolling(short_ma).mean()
    df[f"SMA_{long_ma}"] = df["Close"].rolling(long_ma).mean()
    df["MA_Ratio"] = df[f"SMA_{short_ma}"] / df[f"SMA_{long_ma}"] - 1
    df["Price_Momentum"] = df["Close"] / df["Close"].shift(10) - 1
    df["Trend_Strength"] = df["MA_Ratio"].rolling(10).mean()
    df.dropna(inplace=True)
    features = [
        "Open", "High", "Low", "Close", "Volume",
        "Open_src2", "High_src2", "Low_src2", "Close_src2", "Volume_src2",
        "LogRet", "Volatility", "MA_Ratio", "RSI", "MACD_Hist"
    ]
    for col in ["Open_src2", "High_src2", "Low_src2", "Close_src2", "Volume_src2"]:
        if col not in df.columns:
            features.remove(col)
    if "Sentiment" in df.columns:
        features.append("Sentiment")
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    hmm = GaussianHMM(n_components=n_components, covariance_type="full",
                      n_iter=2000, tol=1e-5, random_state=random_state)
    hmm.fit(X_scaled)
    df["Regime"] = hmm.predict(X_scaled)
    regime_stats = {}
    for regime in range(n_components):
        mask = df["Regime"] == regime
        std_ret = df.loc[mask, "LogRet"].std() or 1e-9
        regime_stats[regime] = {
            "mean_return": df.loc[mask, "LogRet"].mean(),
            "volatility": std_ret,
            "sharpe": df.loc[mask, "LogRet"].mean() / std_ret
        }
    bullish_regime = max(regime_stats, key=lambda k: regime_stats[k]["sharpe"])
    bearish_regime = min(regime_stats, key=lambda k: regime_stats[k]["mean_return"])
    df["Vol_Ratio"] = df["Volatility"] / df["Volatility"].rolling(20).mean()
    df["Signal"] = 0
    bull_condition = (
        (df["Regime"] == bullish_regime) &
        (df[f"SMA_{short_ma}"] > df[f"SMA_{long_ma}"]) &
        (df["MACD"] > df["MACD_Signal"]) &
        (df["RSI"] > 55) & (df["RSI"] < 75) &
        (df["Vol_Ratio"] < 1.2)
    )
    df.loc[bull_condition, "Signal"] = 1
    bear_condition = (
        ((df["Regime"] == bearish_regime) | (df["LogRet"].rolling(5).mean() < -regime_filter_threshold)) &
        (df[f"SMA_{short_ma}"] < df[f"SMA_{long_ma}"]) &
        (df["MACD"] < df["MACD_Signal"]) &
        (df["RSI"] < 60) & (df["RSI"] > 20)
    )
    df.loc[bear_condition, "Signal"] = -1
    df["Position_Size"] = 1.0
    df.loc[df["Vol_Ratio"] > 1.2, "Position_Size"] = 0.7
    df.loc[df["Vol_Ratio"] > 1.5, "Position_Size"] = 0.5
    df["Position"] = df["Signal"].shift().fillna(0) * df["Position_Size"].shift().fillna(1.0)
    df["Daily_Return"] = df["LogRet"] * df["Position"]
    df["Trade"] = df["Position"].diff().abs().fillna(0)
    df["Trade_Fee"] = df["Trade"] * fee_rate
    df["Net_Return"] = df["Daily_Return"] - df["Trade_Fee"]
    df["Strategy_Value"] = (1.0 + df["Daily_Return"]).cumprod()
    df["Net_Value"] = (1.0 + df["Net_Return"]).cumprod()
    df["Market_Value"] = (1.0 + df["LogRet"]).cumprod()
    return df


def evaluate_parameter_set(args):
    data, n_comp, short_ma, long_ma, volatility_window, regime_threshold = args
    if short_ma >= long_ma:
        return None, None, None
    df_result = run_strategy(data, n_components=n_comp, short_ma=short_ma, long_ma=long_ma,
                             volatility_window=volatility_window, regime_filter_threshold=regime_threshold,
                             fee_rate=0.0006, random_state=42)
    metrics = calculate_metrics(df_result, use_net=True)
    return (n_comp, short_ma, long_ma, volatility_window, regime_threshold), metrics, df_result


def download_cybotrade_source(exchange=None):
    API_KEY = "mcabNIyQGUE7IIRqBe4qCfckiAbuHzit6g4CL6pCLNuk2xaC"
    url = "https://api.datasource.cybotrade.rs/cryptoquant/btc/market-data/price-ohlcv"
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    LIMIT = 10000
    start_time_ts = int(dt.datetime(2021, 1, 1).timestamp() * 1000)
    max_allowed_time = int(dt.datetime(2025, 4, 11, 8, 0, 0).timestamp() * 1000)
    all_data = []
    while True:
        if start_time_ts >= max_allowed_time:
            print("✅ Reached latest available timestamp.")
            break
        params = {
            "start_time": start_time_ts,
            "limit": LIMIT,
            "window": "hour"
        }
        if exchange is not None:
            params["exchange"] = exchange
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print("❌ Error:", response.status_code, response.text)
            break
        data = response.json().get("data", [])
        if not data:
            print("No more data returned. Stopping.")
            break
        all_data.extend(data)
        last_ts = data[-1]["start_time"]
        start_time_ts = last_ts + 1
        print(f"Retrieved {len(all_data)} records so far...")
        time.sleep(1.5)
    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["start_time"], unit="ms")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    if exchange is not None:
        df = df.rename(columns={
            "open": "Open_src2",
            "high": "High_src2",
            "low": "Low_src2",
            "close": "Close_src2",
            "volume": "Volume_src2"
        })
    else:
        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })
    df = df.set_index("timestamp")
    df.sort_index(inplace=True)
    return df


def download_combined_data():
    df_primary = download_cybotrade_source(exchange=None)
    df_secondary = download_cybotrade_source(exchange="binance")
    df_combined = pd.merge(df_primary, df_secondary, left_index=True, right_index=True, how="inner")
    df_combined = df_combined.sort_index().fillna(method="ffill")
    return df_combined


def add_nlp_sentiment(df):
    np.random.seed(42)
    df["Sentiment"] = np.random.uniform(-1, 1, len(df))
    return df


def plot_strategy_results(ticker, df, params):
    n_comp, short_ma, long_ma, vol_window, regime_thresh = params
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df["Net_Value"], label="Strategy (Net of Fees)", linewidth=2)
    ax1.plot(df.index, df["Market_Value"], label="Buy & Hold", linestyle="--", alpha=0.7)
    ax1.set_title(f"{ticker} - HMM + MA Strategy Performance", fontsize=14)
    ax1.set_ylabel("Cumulative Return", fontsize=12)
    ax1.legend()
    ax1.grid(True)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(df.index, df["Close"], label="Primary Close", color="black", alpha=0.5, linewidth=1)
    if "Close_src2" in df.columns:
        ax2.plot(df.index, df["Close_src2"], label="Secondary Close", linestyle="--", alpha=0.7)
    ax2.plot(df.index, df[f"SMA_{short_ma}"], label=f"SMA {short_ma}", linestyle="--")
    ax2.plot(df.index, df[f"SMA_{long_ma}"], label=f"SMA {long_ma}", linestyle="--")
    unique_regimes = df["Regime"].unique()
    for regime in unique_regimes:
        regime_data = df[df["Regime"] == regime]
        if len(regime_data) > 0:
            for i in range(len(regime_data) - 1):
                if i == 0 or regime_data.iloc[i-1].name != regime_data.iloc[i].name - pd.Timedelta(hours=1):
                    start = regime_data.iloc[i].name
                    j = i
                    while j < len(regime_data) - 1 and regime_data.iloc[j+1].name == regime_data.iloc[j].name + pd.Timedelta(hours=1):
                        j += 1
                    end = regime_data.iloc[j].name
                    ax2.axvspan(start, end, alpha=0.2, color=f"C{regime}")
    buy_dates = df[df["Signal"] == 1].index
    ax2.scatter(buy_dates, df.loc[buy_dates, "Close"], marker="^", color="green", label="Buy Signal", s=80)
    sell_dates = df[df["Signal"] == -1].index
    ax2.scatter(sell_dates, df.loc[sell_dates, "Close"], marker="v", color="red", label="Sell Signal", s=80)
    ax2.set_title(f"{ticker} - Buy and Sell Signals", fontsize=14)
    ax2.set_ylabel("Price (USD)", fontsize=12)
    ax2.legend()
    ax2.grid(True)
    ax3 = fig.add_subplot(gs[2, 0])
    running_max = df["Net_Value"].cummax()
    drawdown = (df["Net_Value"] - running_max) / running_max * 100
    ax3.fill_between(df.index, drawdown, 0, color="red", alpha=0.3)
    ax3.set_title("Drawdown (%)", fontsize=14)
    ax3.set_ylabel("Drawdown %", fontsize=12)
    ax3.grid(True)
    ax4 = fig.add_subplot(gs[2, 1])
    regime_counts = df["Regime"].value_counts().sort_index()
    bars = ax4.bar(regime_counts.index, regime_counts.values)
    for i, regime in enumerate(regime_counts.index):
        mask = df["Regime"] == regime
        if mask.sum() > 0:
            ret = df.loc[mask, "LogRet"].mean() * 252 * 100
            ax4.text(regime, regime_counts[regime],
                     f"Return: {ret:.1f}%\nCount: {regime_counts[regime]}", ha='center', va='bottom')
            bars[i].set_color('green' if ret > 0 else 'red')
    ax4.set_title("Regime Distribution", fontsize=14)
    ax4.set_xlabel("Regime", fontsize=12)
    ax4.set_ylabel("Count", fontsize=12)
    ax4.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


def main():
    ticker = "BTC"
    print(f"\nProcessing {ticker} using multiple Cybotrade sources...")
    data = download_combined_data()
    data.dropna(inplace=True)
    if data.empty:
        print(f"No data available for {ticker}")
        return
    data = add_nlp_sentiment(data)
    train_end = dt.datetime.strptime("2023-01-01", "%Y-%m-%d")
    train_data = data[data.index < train_end].copy()
    test_data = data[data.index >= train_end].copy()
    N_COMPONENTS_LIST = [2, 3, 4, 5]
    SHORT_MA_LIST = [5, 8, 10, 12, 15, 20]
    LONG_MA_LIST = [20, 25, 30, 35, 40, 50, 60]
    VOLATILITY_WINDOW_LIST = [5, 10, 15]
    REGIME_THRESHOLD_LIST = [0.1, 0.2, 0.3]
    param_combinations = []
    for n_comp in N_COMPONENTS_LIST:
        for short_ma in SHORT_MA_LIST:
            for long_ma in LONG_MA_LIST:
                for vol_window in VOLATILITY_WINDOW_LIST:
                    for regime_thresh in REGIME_THRESHOLD_LIST:
                        if short_ma < long_ma:
                            param_combinations.append((train_data, n_comp, short_ma, long_ma, vol_window, regime_thresh))
    print(f"Testing {len(param_combinations)} parameter combinations for {ticker}...")
    results_list = Parallel(n_jobs=-1)(
        delayed(evaluate_parameter_set)(args) for args in param_combinations
    )
    all_results = []
    for params, metrics, df_result in results_list:
        if params is None or metrics is None:
            continue
        all_results.append((params, metrics, df_result))
    results = {}
    if all_results:
        best_overall = sorted(all_results, key=lambda x: x[1]["sharpe_ratio"], reverse=True)[0]
        best_params, best_metrics, _ = best_overall
        best_n_comp, best_short_ma, best_long_ma, best_vol_window, best_regime_thresh = best_params
        print(f"Best candidate on training data for {ticker}:")
        print(f"  n_components = {best_n_comp}")
        print(f"  short_ma = {best_short_ma}")
        print(f"  long_ma = {best_long_ma}")
        print(f"  volatility_window = {best_vol_window}")
        print(f"  regime_threshold = {best_regime_thresh}")
        print(f"Sharpe Ratio: {best_metrics['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {best_metrics['max_drawdown']:.4f}")
        print("Testing best candidate on out-of-sample data...")
        test_df = run_strategy(test_data, n_components=best_n_comp, short_ma=best_short_ma,
                               long_ma=best_long_ma, volatility_window=best_vol_window,
                               regime_filter_threshold=best_regime_thresh, fee_rate=0.0006, random_state=42)
        test_metrics = calculate_metrics(test_df, use_net=True)
        final_df = run_strategy(data, n_components=best_n_comp, short_ma=best_short_ma,
                                long_ma=best_long_ma, volatility_window=best_vol_window,
                                regime_filter_threshold=best_regime_thresh, fee_rate=0.0006, random_state=42)
        final_metrics = calculate_metrics(final_df, use_net=True)
        results[ticker] = {
            "params": best_params,
            "metrics": final_metrics,
            "df": final_df
        }
        print(f"\nFinal evaluation for {ticker}:")
        print(f"  n_components = {best_n_comp}")
        print(f"  short_ma = {best_short_ma}")
        print(f"  long_ma = {best_long_ma}")
        print(f"  volatility_window = {best_vol_window}")
        print(f"  regime_threshold = {best_regime_thresh}")
        print("Final metrics:")
        for k, v in final_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    else:
        print(f"No parameter sets produced results for {ticker}")
    print("\n===== SUMMARY OF RESULTS =====")
    summary_data = []
    for tick in results:
        metrics = results[tick]["metrics"]
        summary_data.append({
            "Ticker": tick,
            "Sharpe": metrics["sharpe_ratio"],
            "Return": metrics["total_return"],
            "Drawdown": metrics["max_drawdown"],
            "Trade Freq": metrics["trade_frequency"],
            "Win Rate": metrics["win_rate"]
        })
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        print(f"\nDisplaying graph for the analyzed ticker: {ticker}")
        plot_strategy_results(ticker, results[ticker]["df"], results[ticker]["params"])
    else:
        print("No strategies produced results.")


if __name__ == "__main__":
    main()


# In[ ]:




