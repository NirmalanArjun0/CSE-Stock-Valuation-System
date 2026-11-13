# ============================================================
# FULL TECH MODULE + SIGNALS + BACKTEST + REPORT + RECO TABLE
# + (2) WALK-FORWARD TESTING
# + (3) ATR RISK MGMT + POSITION SIZING (Stop/TP/Trailing)
# + (5) STRATEGY COMPARISON (3 strategies)
#
# Input : Technical data_all.csv
# Output: ./indicators_outputs (next to this .py file)
# ============================================================

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# SETTINGS (EDIT THESE)
# -----------------------
TECH_CSV = r"C:\Users\nirma\OneDrive\Desktop(1)\BScDS Project 1\Technical data_all.csv"

# Optional forecast folder (skip if None)
FORECAST_DIR = None  # e.g. r"C:\...\forecasts"

# Plot/report
PLOT_TAIL = 260
FIB_LOOKBACK = 120

# Indicators
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14

# Support/Resistance
SWING_WINDOW = 3
SR_LEVELS_MAX = 8

# If you want only these bases:
BASES_KEEP = {"COMB", "HNB", "HNBF", "LPL", "RHL", "SHOT"}  # set None for all

# -----------------------
# RISK MANAGEMENT (3)
# -----------------------
START_CAPITAL = 100000.0        # starting capital (Rs.)
RISK_PER_TRADE = 0.01           # 1% risk per trade
ATR_STOP_MULT = 2.0             # stop-loss distance = ATR * multiplier
ATR_TP_MULT = 3.0               # take-profit distance = ATR * multiplier (set None to disable)
USE_TRAILING_STOP = True        # trailing stop using ATR
MAX_POSITION_FRACTION = 1.0     # 1.0 => can use full capital (no leverage)

# -----------------------
# WALK-FORWARD (2)
# -----------------------
WF_TRAIN_BARS = 504   # ~2 years if daily (approx)
WF_TEST_BARS = 126    # ~6 months
WF_STEP_BARS = 126    # slide by test length
WF_MIN_BARS = 260     # require at least this many rows to do WFT

# -----------------------
# OUTPUT FOLDER FIX
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "indicators_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

print("RUNNING FROM (cwd):", os.getcwd())
print("SCRIPT LOCATION:", BASE_DIR)
print("SAVING OUTPUT TO:", OUT_DIR)

# -----------------------
# HELPERS
# -----------------------
def base_of(ticker: str) -> str:
    ticker = str(ticker).strip()
    return ticker.split(".")[0] if "." in ticker else ticker

def rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series: pd.Series, period=20, nstd=2):
    ma = series.rolling(period).mean()
    sd = series.rolling(period).std()
    upper = ma + nstd * sd
    lower = ma - nstd * sd
    return ma, upper, lower

def true_range(high, low, close):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def atr(high, low, close, period=14):
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def fib_levels(high_val: float, low_val: float):
    diff = high_val - low_val
    return {
        "23.6%": high_val - 0.236 * diff,
        "38.2%": high_val - 0.382 * diff,
        "50.0%": high_val - 0.5   * diff,
        "61.8%": high_val - 0.618 * diff,
    }

def pivot_points(high_val: float, low_val: float, close_val: float):
    P = (high_val + low_val + close_val) / 3.0
    R1 = 2*P - low_val
    S1 = 2*P - high_val
    R2 = P + (high_val - low_val)
    S2 = P - (high_val - low_val)
    return {"P":P, "R1":R1, "S1":S1, "R2":R2, "S2":S2}

def cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) <= b.shift(1)) & (a > b)

def cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.shift(1) >= b.shift(1)) & (a < b)

def find_volume_column(df: pd.DataFrame):
    candidates = ["Volume", "VOL", "Vol", "volume", "Turnover", "turnover", "Turnover (Rs.)", "Value", "Trade Value"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "vol" in c.lower():
            return c
    return None

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()

def is_local_max(x: pd.Series, i: int, w: int) -> bool:
    if i < w or i > len(x) - w - 1:
        return False
    window = x.iloc[i-w:i+w+1]
    return x.iloc[i] == window.max()

def is_local_min(x: pd.Series, i: int, w: int) -> bool:
    if i < w or i > len(x) - w - 1:
        return False
    window = x.iloc[i-w:i+w+1]
    return x.iloc[i] == window.min()

def support_resistance_levels(high: pd.Series, low: pd.Series, w=3, max_levels=8):
    swing_highs, swing_lows = [], []
    for i in range(len(high)):
        if is_local_max(high, i, w):
            swing_highs.append(float(high.iloc[i]))
        if is_local_min(low, i, w):
            swing_lows.append(float(low.iloc[i]))

    levels = swing_highs + swing_lows
    if not levels:
        return []

    median_price = np.median(levels)
    r = 1 if median_price >= 10 else 2
    rounded = [round(v, r) for v in levels]
    counts = pd.Series(rounded).value_counts()
    top = counts.head(max_levels).index.tolist()
    return sorted([float(v) for v in top])

def try_load_forecast(ticker: str):
    if not FORECAST_DIR:
        return None
    patterns = [
        os.path.join(FORECAST_DIR, f"{ticker}*forecast*.csv"),
        os.path.join(FORECAST_DIR, f"*{ticker}*.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    if not files:
        return None

    fpath = sorted(files)[0]
    try:
        f = pd.read_csv(fpath)
    except Exception:
        return None

    date_col = None
    for c in f.columns:
        if c.lower() in ("ds", "date", "datetime", "trade date"):
            date_col = c
            break
    if date_col is None:
        date_col = f.columns[0]

    f[date_col] = pd.to_datetime(f[date_col], errors="coerce")
    f = f.dropna(subset=[date_col]).copy()

    forecast_col = None
    for c in f.columns:
        if c.lower() in ("yhat", "forecast", "pred", "prediction", "y_pred", "yhat1"):
            forecast_col = c
            break
    if forecast_col is None:
        num_cols = [c for c in f.columns if c != date_col and pd.api.types.is_numeric_dtype(f[c])]
        if not num_cols:
            return None
        forecast_col = num_cols[0]

    out = pd.DataFrame({"date": f[date_col], "forecast": pd.to_numeric(f[forecast_col], errors="coerce")}).dropna()

    # lower/upper optional
    lower_col = None
    upper_col = None
    for c in f.columns:
        if c.lower() in ("lower", "lo", "yhat_lower", "lower_bound", "yhat_lo"):
            lower_col = c
        if c.lower() in ("upper", "hi", "yhat_upper", "upper_bound", "yhat_hi"):
            upper_col = c

    if lower_col and upper_col:
        out["lower"] = pd.to_numeric(f[lower_col], errors="coerce")
        out["upper"] = pd.to_numeric(f[upper_col], errors="coerce")
        out = out.dropna(subset=["lower", "upper"])
    else:
        out["lower"] = np.nan
        out["upper"] = np.nan

    out = out.sort_values("date").reset_index(drop=True)
    out.attrs["source_file"] = fpath
    return out

def save_fig(fig, path):
    fig.savefig(path, dpi=170, bbox_inches="tight")
    print("SAVED:", path)
    plt.close(fig)

# -----------------------
# STRATEGIES (5)
# -----------------------
def strategy_signals(df: pd.DataFrame, name: str):
    """
    Returns buy_signal, sell_signal boolean series for a given strategy name.
    Strategy set:
      S1_RSI_MACD
      S2_MA_CROSS
      S3_BB_MEANREV
    """
    if name == "S1_RSI_MACD":
        rsi_cross_buy = cross_up(df["RSI14"], pd.Series(30, index=df.index))
        macd_cross_buy = cross_up(df["MACD"], df["MACD_Signal"])
        buy = rsi_cross_buy & macd_cross_buy

        rsi_cross_sell = cross_down(df["RSI14"], pd.Series(70, index=df.index))
        macd_cross_sell = cross_down(df["MACD"], df["MACD_Signal"])
        sell = rsi_cross_sell | macd_cross_sell
        return buy.fillna(False), sell.fillna(False)

    if name == "S2_MA_CROSS":
        # Buy when SMA20 crosses above SMA50; sell when crosses below
        buy = cross_up(df["SMA20"], df["SMA50"])
        sell = cross_down(df["SMA20"], df["SMA50"])
        return buy.fillna(False), sell.fillna(False)

    if name == "S3_BB_MEANREV":
        # Buy when close below lower band AND RSI < 35
        buy = (df["Close (Rs.)"] < df["BB_Lower"]) & (df["RSI14"] < 35)
        # Sell when close returns to BB_MA OR RSI > 60
        sell = (df["Close (Rs.)"] >= df["BB_MA"]) | (df["RSI14"] > 60)
        return buy.fillna(False), sell.fillna(False)

    raise ValueError(f"Unknown strategy: {name}")

# -----------------------
# ATR RISK BACKTEST (3)
# -----------------------
def backtest_atr_risk(
    df: pd.DataFrame,
    buy_sig: pd.Series,
    sell_sig: pd.Series,
    start_capital=100000.0,
    risk_per_trade=0.01,
    atr_stop_mult=2.0,
    atr_tp_mult=3.0,
    use_trailing=True,
    max_position_fraction=1.0,
):
    """
    Long-only, one position at a time.
    Entry: at close on signal bar (simple approximation)
    Exit: on sell signal OR stop-loss OR take-profit OR trailing stop.
    Position size: based on risk per trade and stop distance.
    """
    close = df["Close (Rs.)"].astype(float).values
    high = df["High (Rs.)"].astype(float).values
    low = df["Low (Rs.)"].astype(float).values
    atrv = df["ATR"].astype(float).values

    n = len(df)
    equity = np.zeros(n, dtype=float)
    cash = start_capital
    pos_units = 0.0
    entry_price = np.nan
    stop_price = np.nan
    tp_price = np.nan
    trail_stop = np.nan
    in_pos = False

    trades = []

    def portfolio_value(i):
        return cash + pos_units * close[i]

    for i in range(n):
        price = close[i]
        a = atrv[i] if np.isfinite(atrv[i]) and atrv[i] > 0 else np.nan

        # Update trailing stop if in position
        if in_pos and use_trailing and np.isfinite(a):
            new_trail = price - atr_stop_mult * a
            if not np.isfinite(trail_stop):
                trail_stop = new_trail
            else:
                trail_stop = max(trail_stop, new_trail)

        # Check exits
        exit_reason = None
        if in_pos:
            # stop-loss check using low (intrabar approx)
            if np.isfinite(stop_price) and low[i] <= stop_price:
                exit_reason = "STOP"
                exit_price = stop_price
            # take-profit
            elif atr_tp_mult is not None and np.isfinite(tp_price) and high[i] >= tp_price:
                exit_reason = "TAKE_PROFIT"
                exit_price = tp_price
            # trailing stop
            elif use_trailing and np.isfinite(trail_stop) and low[i] <= trail_stop:
                exit_reason = "TRAIL_STOP"
                exit_price = trail_stop
            # signal exit
            elif bool(sell_sig.iloc[i]):
                exit_reason = "SELL_SIGNAL"
                exit_price = price

            if exit_reason is not None:
                cash += pos_units * exit_price
                pnl_pct = (exit_price / entry_price) - 1.0 if entry_price > 0 else np.nan
                trades.append({
                    "entry_index": entry_i,
                    "exit_index": i,
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "pnl_pct": float(pnl_pct),
                    "exit_reason": exit_reason
                })
                pos_units = 0.0
                entry_price = np.nan
                stop_price = np.nan
                tp_price = np.nan
                trail_stop = np.nan
                in_pos = False

        # Check entry
        if (not in_pos) and bool(buy_sig.iloc[i]):
            if np.isfinite(a):
                # risk amount
                risk_cash = cash * risk_per_trade
                stop_dist = atr_stop_mult * a
                if stop_dist > 0:
                    # units based on risk
                    units = risk_cash / stop_dist

                    # cap by max position fraction of equity
                    max_cash_to_use = cash * max_position_fraction
                    units_cap = max_cash_to_use / price if price > 0 else 0.0
                    units = min(units, units_cap)

                    cost = units * price
                    if units > 0 and cost <= cash:
                        cash -= cost
                        pos_units = units
                        entry_price = price
                        entry_i = i
                        stop_price = price - stop_dist
                        tp_price = price + (atr_tp_mult * a) if atr_tp_mult is not None else np.nan
                        trail_stop = price - stop_dist if use_trailing else np.nan
                        in_pos = True

        equity[i] = portfolio_value(i)

    equity_series = pd.Series(equity, index=df.index)

    # metrics
    ret = equity_series.pct_change().fillna(0.0)
    total_return = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1.0
    peak = equity_series.cummax()
    dd = (equity_series / peak) - 1.0
    max_dd = float(dd.min())
    sharpe = float((ret.mean() / ret.std()) * np.sqrt(252)) if ret.std() > 1e-12 else np.nan

    trades_df = pd.DataFrame(trades)
    n_trades = len(trades_df)
    win_rate = float((trades_df["pnl_pct"] > 0).mean()) if n_trades > 0 else np.nan

    metrics = {
        "final_equity": float(equity_series.iloc[-1]),
        "total_return_pct": float(total_return * 100.0),
        "max_drawdown_pct": float(max_dd * 100.0),
        "sharpe": sharpe,
        "num_trades": int(n_trades),
        "win_rate_pct": float(win_rate * 100.0) if np.isfinite(win_rate) else np.nan
    }
    return equity_series, trades_df, metrics

# -----------------------
# WALK-FORWARD (2)
# -----------------------
def walk_forward_eval(df: pd.DataFrame, buy_sig: pd.Series, sell_sig: pd.Series,
                      train_bars=504, test_bars=126, step_bars=126):
    """
    Rule-based strategies don't 'train', but we still evaluate in rolling out-of-sample folds.
    Each fold: use only the test segment for performance (signals already causal).
    """
    n = len(df)
    results = []
    start = 0

    # ensure enough data for first fold
    if n < (train_bars + test_bars):
        return results

    # fold starts at end of train window
    fold_start = train_bars
    fold_id = 0

    while (fold_start + test_bars) <= n:
        test_slice = slice(fold_start, fold_start + test_bars)
        df_test = df.iloc[test_slice].copy()
        buy_test = buy_sig.iloc[test_slice].copy()
        sell_test = sell_sig.iloc[test_slice].copy()

        # backtest only on this fold
        eq, tr, met = backtest_atr_risk(
            df_test, buy_test, sell_test,
            start_capital=START_CAPITAL,
            risk_per_trade=RISK_PER_TRADE,
            atr_stop_mult=ATR_STOP_MULT,
            atr_tp_mult=ATR_TP_MULT,
            use_trailing=USE_TRAILING_STOP,
            max_position_fraction=MAX_POSITION_FRACTION
        )

        results.append({
            "fold": fold_id,
            "test_start": str(df_test["Trade Date"].iloc[0].date()),
            "test_end": str(df_test["Trade Date"].iloc[-1].date()),
            **met
        })

        fold_id += 1
        fold_start += step_bars

    return results

# -----------------------
# LOAD + CLEAN
# -----------------------
if not os.path.exists(TECH_CSV):
    print(f"❌ File not found: {TECH_CSV}")
    sys.exit(1)

df = pd.read_csv(TECH_CSV)

needed = ["Stock", "Trade Date", "Close (Rs.)"]
for c in needed:
    if c not in df.columns:
        print(f"❌ Missing column: {c}. Found: {list(df.columns)}")
        sys.exit(1)

df["Stock"] = df["Stock"].astype(str).str.strip()
df["Trade Date"] = pd.to_datetime(df["Trade Date"], errors="coerce")

for c in ["Open (Rs.)", "High (Rs.)", "Low (Rs.)", "Close (Rs.)"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["Stock", "Trade Date", "Close (Rs.)"]).copy()
df = df.sort_values(["Stock", "Trade Date"])
df = df.drop_duplicates(subset=["Stock", "Trade Date"], keep="last")

df["Base"] = df["Stock"].apply(base_of)
if BASES_KEEP:
    df = df[df["Base"].isin(BASES_KEEP)].copy()

tickers = sorted(df["Stock"].unique().tolist())
print(" Tickers found:", tickers)
if len(tickers) == 0:
    print("⚠️ No tickers after filtering. Set BASES_KEEP=None or adjust bases.")
    sys.exit(0)

vol_col = find_volume_column(df)
print(" Volume column found:" if vol_col else "ℹ️ No volume column found:", vol_col)

# -----------------------
# MAIN LOOP
# -----------------------
strategy_names = ["S1_RSI_MACD", "S2_MA_CROSS", "S3_BB_MEANREV"]

recommendation_rows = []
overall_strategy_rows = []
walk_forward_rows = []
best_strategy_rows = []

for ticker in tickers:
    g = df[df["Stock"] == ticker].copy().sort_values("Trade Date").reset_index(drop=True)

    # Ensure OHLC
    if "High (Rs.)" not in g.columns or g["High (Rs.)"].isna().all():
        g["High (Rs.)"] = g["Close (Rs.)"]
    if "Low (Rs.)" not in g.columns or g["Low (Rs.)"].isna().all():
        g["Low (Rs.)"] = g["Close (Rs.)"]
    g["High (Rs.)"] = g["High (Rs.)"].fillna(g["Close (Rs.)"])
    g["Low (Rs.)"]  = g["Low (Rs.)"].fillna(g["Close (Rs.)"])

    # Indicators on full history
    close_full = g["Close (Rs.)"].astype(float)
    g["SMA20"] = close_full.rolling(20).mean()
    g["SMA50"] = close_full.rolling(50).mean()
    g["SMA200"] = close_full.rolling(200).mean()
    g["RSI14"] = rsi(close_full, RSI_PERIOD)
    g["MACD"], g["MACD_Signal"], g["MACD_Hist"] = macd(close_full)
    g["BB_MA"], g["BB_Upper"], g["BB_Lower"] = bollinger(close_full, BB_PERIOD, BB_STD)
    g["ATR"] = atr(g["High (Rs.)"], g["Low (Rs.)"], g["Close (Rs.)"], ATR_PERIOD)

    # Volume indicators if available
    if vol_col:
        g[vol_col] = pd.to_numeric(g[vol_col], errors="coerce").fillna(0.0)
        g["VOL_SMA20"] = g[vol_col].rolling(20).mean()
        g["OBV"] = obv(g["Close (Rs.)"], g[vol_col])

    if len(g) < WF_MIN_BARS:
        print(f"⚠️ {ticker}: too few rows for walk-forward ({len(g)}). It will still do overall backtest.")
    # Tail for report
    gt = g.tail(PLOT_TAIL).copy()
    if len(gt) < 60:
        print(f"⚠️ Skipping report for {ticker}: not enough rows ({len(gt)})")
        continue

    # Levels
    fib_window = gt.tail(min(FIB_LOOKBACK, len(gt)))
    swing_high = float(fib_window["High (Rs.)"].max())
    swing_low = float(fib_window["Low (Rs.)"].min())
    fib = fib_levels(swing_high, swing_low)

    last = gt.iloc[-1]
    piv = pivot_points(float(last["High (Rs.)"]), float(last["Low (Rs.)"]), float(last["Close (Rs.)"]))
    sr_levels = support_resistance_levels(gt["High (Rs.)"], gt["Low (Rs.)"], w=SWING_WINDOW, max_levels=SR_LEVELS_MAX)

    # Optional forecast overlay
    fcast = try_load_forecast(ticker)
    fcast_note = ""
    if fcast is not None and len(fcast) > 0:
        if fcast["lower"].isna().all() or fcast["upper"].isna().all():
            resid_std = gt["Close (Rs.)"].diff().tail(60).std()
            if np.isfinite(resid_std) and resid_std > 0:
                fcast["lower"] = fcast["forecast"] - 1.96 * resid_std
                fcast["upper"] = fcast["forecast"] + 1.96 * resid_std
                fcast_note = " (band≈1.96*recent_vol)"
        fcast_note = fcast_note or " (from file)"

    # -----------------------
    # STRATEGY COMPARISON (5) + OVERALL BACKTEST (3)
    # -----------------------
    strat_metrics = []
    strat_trades = {}
    strat_equity = {}

    for sname in strategy_names:
        buy_sig, sell_sig = strategy_signals(g, sname)
        eq, trades_df, met = backtest_atr_risk(
            g, buy_sig, sell_sig,
            start_capital=START_CAPITAL,
            risk_per_trade=RISK_PER_TRADE,
            atr_stop_mult=ATR_STOP_MULT,
            atr_tp_mult=ATR_TP_MULT,
            use_trailing=USE_TRAILING_STOP,
            max_position_fraction=MAX_POSITION_FRACTION,
        )
        met_row = {"Ticker": ticker, "Base": base_of(ticker), "Strategy": sname, **met}
        overall_strategy_rows.append(met_row)
        strat_metrics.append((sname, met))
        strat_trades[sname] = trades_df
        strat_equity[sname] = eq

        # Walk-forward for this strategy
        wf = walk_forward_eval(g, buy_sig, sell_sig, WF_TRAIN_BARS, WF_TEST_BARS, WF_STEP_BARS)
        for fold_row in wf:
            walk_forward_rows.append({
                "Ticker": ticker,
                "Base": base_of(ticker),
                "Strategy": sname,
                **fold_row
            })

    # Choose BEST strategy by return (you can change to Sharpe if you want)
    best_s = max(strat_metrics, key=lambda x: (x[1]["total_return_pct"] if np.isfinite(x[1]["total_return_pct"]) else -1e9))[0]
    best_met = dict([m for (n, m) in strat_metrics if n == best_s][0])

    best_strategy_rows.append({
        "Ticker": ticker,
        "Base": base_of(ticker),
        "Best_Strategy": best_s,
        "Best_Total_Return_%": best_met["total_return_pct"],
        "Best_Max_Drawdown_%": best_met["max_drawdown_pct"],
        "Best_Sharpe": best_met["sharpe"],
        "Best_Num_Trades": best_met["num_trades"],
        "Best_Win_Rate_%": best_met["win_rate_pct"]
    })

    # For the report and recommendation table, use BEST strategy signals on the tail window
    buy_best, sell_best = strategy_signals(g, best_s)
    buy_tail = buy_best.loc[gt.index]
    sell_tail = sell_best.loc[gt.index]

    latest_signal = "HOLD"
    if bool(buy_best.iloc[-1]):
        latest_signal = "BUY"
    elif bool(sell_best.iloc[-1]):
        latest_signal = "SELL"

    # Simple label table fields
    lat_price = float(last["Close (Rs.)"])
    lat_rsi = float(last["RSI14"]) if np.isfinite(last["RSI14"]) else np.nan
    lat_macd = float(last["MACD"]) if np.isfinite(last["MACD"]) else np.nan
    lat_sig = float(last["MACD_Signal"]) if np.isfinite(last["MACD_Signal"]) else np.nan

    rsi_stat = "OVERSOLD" if np.isfinite(lat_rsi) and lat_rsi <= 30 else ("OVERBOUGHT" if np.isfinite(lat_rsi) and lat_rsi >= 70 else "NEUTRAL")
    macd_stat = "BULLISH" if np.isfinite(lat_macd) and np.isfinite(lat_sig) and lat_macd > lat_sig else "BEARISH"

    recommendation_rows.append({
        "Ticker": ticker,
        "Base": base_of(ticker),
        "Last_Date": str(gt["Trade Date"].iloc[-1].date()),
        "Last_Close": round(lat_price, 4),
        "RSI14": None if not np.isfinite(lat_rsi) else round(lat_rsi, 2),
        "RSI_Status": rsi_stat,
        "MACD_Status": macd_stat,
        "Best_Strategy": best_s,
        "Latest_Signal": latest_signal,
        "SR_Levels": "; ".join([str(x) for x in sr_levels[:6]]),
        "Pivot_P": round(piv["P"], 4),
        "Fib_Levels": f"{round(fib['23.6%'],2)},{round(fib['38.2%'],2)},{round(fib['50.0%'],2)},{round(fib['61.8%'],2)}",
        "Risk_Setting": f"Risk {int(RISK_PER_TRADE*100)}% | SL={ATR_STOP_MULT}ATR | TP={ATR_TP_MULT}ATR | Trail={USE_TRAILING_STOP}"
    })

    # -----------------------
    # REPORT (still 3 panels, but uses BEST strategy + risk backtest metrics)
    # -----------------------
    # pick equity for best strategy, and metrics from best row already
    eq_best = strat_equity[best_s].loc[gt.index] if len(strat_equity[best_s]) >= len(gt) else None
    met_best = [m for (n, m) in strat_metrics if n == best_s][0]

    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(gt["Trade Date"], gt["Close (Rs.)"], label="Close")
    ax1.plot(gt["Trade Date"], gt["SMA20"], label="SMA20")
    ax1.plot(gt["Trade Date"], gt["SMA50"], label="SMA50")
    ax1.plot(gt["Trade Date"], gt["SMA200"], label="SMA200")
    ax1.plot(gt["Trade Date"], gt["BB_Upper"], linestyle="--", label="BB Upper")
    ax1.plot(gt["Trade Date"], gt["BB_Lower"], linestyle="--", label="BB Lower")

    for k, v in fib.items():
        ax1.hlines(v, gt["Trade Date"].iloc[0], gt["Trade Date"].iloc[-1], linestyles=":", linewidth=1)
        ax1.text(gt["Trade Date"].iloc[0], v, f"Fib {k}", va="bottom", fontsize=8)

    for name in ["P", "R1", "S1"]:
        ax1.hlines(piv[name], gt["Trade Date"].iloc[0], gt["Trade Date"].iloc[-1], linestyles="-.", linewidth=1)

    for lvl in sr_levels:
        ax1.hlines(lvl, gt["Trade Date"].iloc[0], gt["Trade Date"].iloc[-1], linestyles="--", linewidth=0.9)

    ax1.scatter(gt.loc[buy_tail, "Trade Date"], gt.loc[buy_tail, "Close (Rs.)"], marker="^", s=70, label="BUY")
    ax1.scatter(gt.loc[sell_tail, "Trade Date"], gt.loc[sell_tail, "Close (Rs.)"], marker="v", s=70, label="SELL")

    if fcast is not None and len(fcast) > 0:
        ax1.plot(fcast["date"], fcast["forecast"], label=f"Forecast{fcast_note}")
        if fcast["lower"].notna().any() and fcast["upper"].notna().any():
            ax1.fill_between(fcast["date"], fcast["lower"], fcast["upper"], alpha=0.2, label="Confidence band")

    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.25)
    ax1.set_title(
        f"{ticker} — BEST={best_s} | "
        f"Ret={met_best['total_return_pct']:.1f}% DD={met_best['max_drawdown_pct']:.1f}% "
        f"Sharpe={(met_best['sharpe'] if np.isfinite(met_best['sharpe']) else np.nan):.2f} "
        f"Trades={met_best['num_trades']} Win={(met_best['win_rate_pct'] if np.isfinite(met_best['win_rate_pct']) else np.nan):.1f}% "
        f"| Risk: {int(RISK_PER_TRADE*100)}% SL={ATR_STOP_MULT}ATR TP={ATR_TP_MULT}ATR Trail={USE_TRAILING_STOP}"
    )

    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax2.plot(gt["Trade Date"], gt["RSI14"], label="RSI(14)")
    ax2.axhline(70, linestyle="--")
    ax2.axhline(30, linestyle="--")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left")
    ax2.set_title("RSI(14)")

    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
    ax3.plot(gt["Trade Date"], gt["MACD"], label="MACD")
    ax3.plot(gt["Trade Date"], gt["MACD_Signal"], label="Signal")
    ax3.bar(gt["Trade Date"], gt["MACD_Hist"], alpha=0.35, label="Hist")
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="upper left")
    ax3.set_title("MACD(12,26,9)")

    plt.tight_layout()
    report_path = os.path.join(OUT_DIR, f"{ticker}_REPORT.png")
    save_fig(fig, report_path)

# -----------------------
# SAVE TABLE OUTPUTS
# -----------------------
reco_table = pd.DataFrame(recommendation_rows).sort_values(["Base", "Ticker"]).reset_index(drop=True)
reco_path = os.path.join(OUT_DIR, "recommendations_table.csv")
reco_table.to_csv(reco_path, index=False)
print(" Saved:", reco_path)

overall_comp = pd.DataFrame(overall_strategy_rows).sort_values(["Base", "Ticker", "Strategy"]).reset_index(drop=True)
overall_comp_path = os.path.join(OUT_DIR, "strategy_comparison_overall.csv")
overall_comp.to_csv(overall_comp_path, index=False)
print(" Saved:", overall_comp_path)

wf_df = pd.DataFrame(walk_forward_rows)
wf_path = os.path.join(OUT_DIR, "walk_forward_results.csv")
wf_df.to_csv(wf_path, index=False)
print(" Saved:", wf_path)

best_df = pd.DataFrame(best_strategy_rows).sort_values(["Base", "Ticker"]).reset_index(drop=True)
best_path = os.path.join(OUT_DIR, "best_strategy_per_ticker.csv")
best_df.to_csv(best_path, index=False)
print(" Saved:", best_path)

print("\n===== TOP RECOMMENDATIONS TABLE (first 30) =====")
print(reco_table.head(30))

print("\nDONE  Check folder:", OUT_DIR)
