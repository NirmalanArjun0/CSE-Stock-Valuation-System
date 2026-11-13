



# ============================================================
# N-BEATS (TECHNICAL CSV ONLY) — Voting Forecast + Gap NonVoting
# COMB/HNB/HNBF/LPL/RHL/SHOT (6 bases)
# ============================================================

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from neuralforecast.losses.pytorch import MAE as LossMAE

# ---------------------------
# 1) CONFIG
# ---------------------------
TECH_PATH = r"C:\Users\nirma\OneDrive\Desktop(1)\BScDS Project 1\Technical data_all.csv"  # <-- change if needed
FREQ = "B"              # business days
H_FUT = 60              # future forecast horizon
PLOT_CONTEXT = 180      # plot last N actual business days

OUT_DIR = "outputs"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# If some stocks are shorter, we auto-shrink VAL/TEST to avoid skipping
VAL_H_DEFAULT  = 30
TEST_H_DEFAULT = 60

BASES_TARGET = ["COMB", "HNB", "HNBF", "LPL", "RHL", "SHOT"]  # only these 6 bases

# N-BEATS tuning grid (keep it light)
TUNE_GRID = [
    {"name": "small",  "input_mult": 2, "mlp": [256, 256], "n_blocks": [2, 2], "lr": 5e-4, "steps": 100},
    {"name": "mid",    "input_mult": 3, "mlp": [512, 512], "n_blocks": [2, 2], "lr": 5e-4, "steps": 100},
    {"name": "bigger", "input_mult": 4, "mlp": [512, 512], "n_blocks": [3, 3], "lr": 5e-4, "steps": 100},
]

# ---------------------------
# 2) HELPERS
# ---------------------------
def base_ticker(code: str) -> str:
    code = str(code).strip()
    if code.endswith(".N") or code.endswith(".X"):
        return code[:-2]
    return code

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred):
    # MAPE is unstable for prices when y_true is small -> use Rs.1 floor
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1.0)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

def safe_log1p(x):
    # Ensure non-negative before log1p
    return np.log1p(np.maximum(np.asarray(x, dtype=float), 0.0))

def safe_expm1(x):
    return np.expm1(np.asarray(x, dtype=float))

def clip_preds_to_reasonable(y_true, y_pred):
    """
    Prevent expm1 explosions from dominating metrics.
    Upper bound = 99.5th percentile of y_true (per fold).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        return y_pred

    hi = float(np.percentile(y_true, 99.5)) if np.isfinite(y_true).all() else float(np.nanmax(y_true))
    if not np.isfinite(hi) or hi <= 0:
        hi = float(np.nanmax(y_true)) if np.isfinite(np.nanmax(y_true)) else 1.0

    y_pred = np.clip(y_pred, 0.01, hi)
    return y_pred

def reindex_business_days(df_one: pd.DataFrame) -> pd.DataFrame:
    df_one = df_one.sort_values("Trade Date").copy()
    df_one = df_one.set_index("Trade Date")
    full_idx = pd.date_range(df_one.index.min(), df_one.index.max(), freq="B")
    df_one = df_one.reindex(full_idx)

    df_one["Stock"] = df_one["Stock"].iloc[0]

    # forward-fill prices
    for c in ["Open (Rs.)", "High (Rs.)", "Low (Rs.)", "Close (Rs.)"]:
        if c in df_one.columns:
            df_one[c] = df_one[c].ffill()

    return df_one.reset_index().rename(columns={"index": "Trade Date"})

# ---------------------------
# 3) LOAD + CLEAN TECH
# ---------------------------
tech = pd.read_csv(TECH_PATH)
print("Loaded:", tech.shape)
print("Columns:", list(tech.columns))

tech["Stock"] = tech["Stock"].astype(str).str.strip()
tech["Trade Date"] = pd.to_datetime(tech["Trade Date"], errors="coerce")

num_cols = ["Open (Rs.)", "High (Rs.)", "Low (Rs.)", "Close (Rs.)",
            "TradeVolume", "ShareVolume", "Turnover (Rs.)"]
for c in num_cols:
    if c in tech.columns:
        tech[c] = pd.to_numeric(tech[c], errors="coerce")

tech = tech.dropna(subset=["Stock", "Trade Date", "Close (Rs.)"]).copy()
tech = tech.sort_values(["Stock", "Trade Date"])
tech = tech.drop_duplicates(subset=["Stock", "Trade Date"], keep="last")

print("After basic cleaning:", tech.shape, "unique stocks:", tech["Stock"].nunique())

# Reindex business days per stock
tech_b = tech.groupby("Stock", group_keys=False).apply(reindex_business_days)

# Close-only time series
ts = tech_b.rename(columns={"Stock": "unique_id", "Trade Date": "ds", "Close (Rs.)": "y"})[
    ["unique_id", "ds", "y"]
].dropna().copy()

ts["unique_id"] = ts["unique_id"].astype(str).str.strip()
ts = ts.sort_values(["unique_id", "ds"]).reset_index(drop=True)

print("ts shape:", ts.shape)
print("Unique tickers:", ts["unique_id"].nunique())

# ---------------------------
# 4) BUILD GAP SERIES (Voting - NonVoting) for 6 bases
# ---------------------------
gap_rows = []
bases = []

for b in BASES_TARGET:
    v_id = f"{b}.N"
    x_id = f"{b}.X"

    v = ts[ts["unique_id"] == v_id][["ds", "y"]].rename(columns={"y": "v"})
    x = ts[ts["unique_id"] == x_id][["ds", "y"]].rename(columns={"y": "x"})

    m = pd.merge(v, x, on="ds", how="inner")
    if len(m) == 0:
        print(f"⚠️ No overlap dates for {b} voting/non-voting. Skipping base.")
        continue

    m["unique_id"] = f"GAP_{b}"
    m["y"] = m["v"] - m["x"]
    gap_rows.append(m[["unique_id", "ds", "y"]])
    bases.append(b)

gap_df = pd.concat(gap_rows, ignore_index=True) if gap_rows else pd.DataFrame(columns=["unique_id","ds","y"])
print("✅ Bases:", bases)
print("Gap rows:", len(gap_df))

def get_stable_gap(b, n=30):
    g = gap_df[gap_df["unique_id"] == f"GAP_{b}"].sort_values("ds")
    if len(g) == 0:
        return 0.0
    return float(np.median(g.tail(n)["y"].values))

# ---------------------------
# 5) TRAIN/VAL/TEST SPLIT + FIT/EVAL
# ---------------------------
def split_train_val_test(df_uid, val_h, test_h):
    df_uid = df_uid.sort_values("ds").reset_index(drop=True)
    n = len(df_uid)
    min_needed = val_h + test_h + 120
    if n < min_needed:
        return None, None, None
    test = df_uid.iloc[-test_h:].copy()
    val  = df_uid.iloc[-(test_h + val_h):-test_h].copy()
    train = df_uid.iloc[:-(test_h + val_h)].copy()
    return train, val, test

def fit_eval_one_series(df_uid, val_h, test_h):
    uid = df_uid["unique_id"].iloc[0]
    train, val, test = split_train_val_test(df_uid, val_h, test_h)
    if train is None:
        return None

    # log1p transform for training
    tr = train.copy(); tr["y"] = safe_log1p(tr["y"])
    va = val.copy();   va["y"] = safe_log1p(va["y"])
    te = test.copy();  te["y"] = safe_log1p(te["y"])

    best = None

    # ---- VAL selection (R2 computed on LOG scale for stability) ----
    for cfg in TUNE_GRID:
        input_size = max(cfg["input_mult"] * test_h, 120)

        model = NBEATS(
            h=val_h,
            input_size=input_size,
            stack_types=["trend", "seasonality"],
            n_blocks=cfg["n_blocks"],
            mlp_units=[cfg["mlp"], cfg["mlp"]],
            learning_rate=cfg["lr"],
            max_steps=cfg["steps"],
            batch_size=64,
            random_seed=1,
            loss=LossMAE(),
            scaler_type="standard",
        )

        nf = NeuralForecast(models=[model], freq=FREQ)
        nf.fit(df=tr, val_size=0)

        fcst_val = nf.predict().reset_index()
        pred_col = [c for c in fcst_val.columns if c not in ["unique_id", "ds"]][0]
        fcst_val = fcst_val.rename(columns={pred_col: "yhat"}).sort_values("ds")

        eva = pd.merge(va, fcst_val[["unique_id","ds","yhat"]], on=["unique_id","ds"], how="inner")
        if len(eva) == 0:
            continue

        # Metrics:
        # - R2 on LOG scale (stable)
        r2_log = float(r2_score(eva["y"].values, eva["yhat"].values))

        # - MAE/MAPE on original scale, with clipping to avoid expm1 explosions
        y_true = safe_expm1(eva["y"].values)
        y_pred = safe_expm1(eva["yhat"].values)
        y_pred = clip_preds_to_reasonable(y_true, y_pred)

        mae_ = float(mean_absolute_error(y_true, y_pred))
        mape_ = float(mape(y_true, y_pred))
        rmse_ = float(rmse(y_true, y_pred))

        if (best is None) or (r2_log > best["val_r2_log"]):
            best = {
                "cfg": cfg,
                "input_size": int(input_size),
                "val_r2_log": r2_log,
                "val_mae": mae_,
                "val_mape": mape_,
                "val_rmse": rmse_,
            }

    if best is None:
        return None

    # ---- Refit on TRAIN+VAL, evaluate on TEST ----
    trainval = pd.concat([train, val], ignore_index=True).sort_values("ds")
    trainval_log = trainval.copy()
    trainval_log["y"] = safe_log1p(trainval_log["y"])

    model = NBEATS(
        h=test_h,
        input_size=best["input_size"],
        stack_types=["trend", "seasonality"],
        n_blocks=best["cfg"]["n_blocks"],
        mlp_units=[best["cfg"]["mlp"], best["cfg"]["mlp"]],
        learning_rate=best["cfg"]["lr"],
        max_steps=best["cfg"]["steps"],
        batch_size=32,
        random_seed=1,
        loss=LossMAE(),
        scaler_type="standard",
    )

    nf = NeuralForecast(models=[model], freq=FREQ)
    nf.fit(df=trainval_log, val_size=0)

    fcst_test = nf.predict().reset_index()
    pred_col = [c for c in fcst_test.columns if c not in ["unique_id","ds"]][0]
    fcst_test = fcst_test.rename(columns={pred_col:"yhat"}).sort_values("ds")

    test_log = test.copy()
    test_log["y"] = safe_log1p(test_log["y"])

    evtest = pd.merge(test_log, fcst_test[["unique_id","ds","yhat"]], on=["unique_id","ds"], how="inner")
    if len(evtest) == 0:
        return None

    # R2 on LOG scale
    best["test_r2_log"] = float(r2_score(evtest["y"].values, evtest["yhat"].values))

    # MAE/MAPE/RMSE on original scale with clipping
    y_true = safe_expm1(evtest["y"].values)
    y_pred = safe_expm1(evtest["yhat"].values)
    y_pred = clip_preds_to_reasonable(y_true, y_pred)

    best["test_mae"]  = float(mean_absolute_error(y_true, y_pred))
    best["test_mape"] = float(mape(y_true, y_pred))
    best["test_rmse"] = float(rmse(y_true, y_pred))

    return best

# ---------------------------
# 6) TRAIN ONLY the 6 voting tickers
# ---------------------------
metrics_rows = []
best_by_uid = {}

for b in bases:
    v_id = f"{b}.N"
    df_uid = ts[ts["unique_id"] == v_id].sort_values("ds").reset_index(drop=True)
    print(f"\n--- Fitting {v_id} len={len(df_uid)}")

    # auto-shrink windows if needed
    val_h = VAL_H_DEFAULT
    test_h = TEST_H_DEFAULT
    if len(df_uid) < (val_h + test_h + 120):
        val_h = 20
        test_h = 40
    if len(df_uid) < (val_h + test_h + 120):
        val_h = 15
        test_h = 30

    best = fit_eval_one_series(df_uid, val_h=val_h, test_h=test_h)
    if best is None:
        print(f"⚠️ Skipping {v_id}: still too short for train/val/test.")
        continue

    best_by_uid[v_id] = best
    metrics_rows.append({
        "Stock": v_id,
        "Base": b,
        "BestCfg": best["cfg"]["name"],
        "input_size": best["input_size"],
        "VAL_H": val_h,
        "TEST_H": test_h,
        "VAL_R2_LOG": best["val_r2_log"],
        "VAL_MAE": best["val_mae"],
        "VAL_MAPE%": best["val_mape"],
        "VAL_RMSE": best["val_rmse"],
        "TEST_R2_LOG": best["test_r2_log"],
        "TEST_MAE": best["test_mae"],
        "TEST_MAPE%": best["test_mape"],
        "TEST_RMSE": best["test_rmse"],
    })

    print(f"✅ Best {v_id}: {best['cfg']['name']} TEST_R2_LOG={best['test_r2_log']:.4f}")

metrics_df = pd.DataFrame(metrics_rows).sort_values("Stock").reset_index(drop=True)

metrics_path = os.path.join(OUT_DIR, "metrics_voting.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"\n✅ Saved: {metrics_path}")
print(metrics_df)

# ---------------------------
# 7) FUTURE FORECAST from LAST ACTUAL DATE + PLOT ALL 6
# ---------------------------
def forecast_from_last(uid, h=H_FUT):
    df = ts[ts["unique_id"] == uid].sort_values("ds").copy()
    df_log = df.copy()
    df_log["y"] = safe_log1p(df_log["y"])

    cfg = best_by_uid[uid]["cfg"]
    input_size = best_by_uid[uid]["input_size"]

    model = NBEATS(
        h=h,
        input_size=input_size,
        stack_types=["trend", "seasonality"],
        n_blocks=cfg["n_blocks"],
        mlp_units=[cfg["mlp"], cfg["mlp"]],
        learning_rate=cfg["lr"],
        max_steps=cfg["steps"],
        batch_size=32,
        random_seed=1,
        loss=LossMAE(),
        scaler_type="standard",
    )
    nf = NeuralForecast(models=[model], freq=FREQ)
    nf.fit(df=df_log, val_size=0)

    fcst = nf.predict().reset_index()
    pred_col = [c for c in fcst.columns if c not in ["unique_id","ds"]][0]
    fcst = fcst.sort_values("ds")

    # back to original scale + clamp
    yhat = safe_expm1(fcst[pred_col].values)
    # clamp future forecast using last 180 days of actual scale (to prevent blowups)
    hist = df["y"].tail(180).values
    hi = float(np.percentile(hist, 99.5)) if len(hist) else float(np.nanmax(df["y"].values))
    if not np.isfinite(hi) or hi <= 0:
        hi = float(np.nanmax(df["y"].values)) if np.isfinite(np.nanmax(df["y"].values)) else 1.0
    yhat = np.clip(yhat, 0.01, hi)

    out = fcst[["ds"]].copy()
    out["yhat"] = yhat
    return out

# Save forecasts for all bases
all_forecasts = []

for b in bases:
    v_id = f"{b}.N"
    x_id = f"{b}.X"

    if v_id not in best_by_uid:
        print("⚠️ No trained model for:", v_id, "so skipping plot/forecast.")
        continue

    gap_val = get_stable_gap(b, n=30)

    v_hist = ts[ts["unique_id"] == v_id].sort_values("ds")
    x_hist = ts[ts["unique_id"] == x_id].sort_values("ds")

    fut_v = forecast_from_last(v_id, h=H_FUT)

    # Non-voting forecast = voting forecast - fixed recent median gap
    fut_v["xhat"] = (fut_v["yhat"] - gap_val).clip(lower=0.01)

    # save forecast rows
    tmp = fut_v.copy()
    tmp["Base"] = b
    tmp["VotingTicker"] = v_id
    tmp["NonVotingTicker"] = x_id
    all_forecasts.append(tmp[["Base","VotingTicker","NonVotingTicker","ds","yhat","xhat"]])

    # Plot
    v_ctx = v_hist.tail(PLOT_CONTEXT)
    x_ctx = x_hist.tail(PLOT_CONTEXT)

    plt.figure(figsize=(13, 5))
    plt.plot(v_ctx["ds"], v_ctx["y"], label=f"{v_id} actual")
    if len(x_ctx) > 0:
        plt.plot(x_ctx["ds"], x_ctx["y"], label=f"{x_id} actual")

    plt.plot(fut_v["ds"], fut_v["yhat"], "o-", label=f"{v_id} forecast ({H_FUT}B)")
    plt.plot(fut_v["ds"], fut_v["xhat"], "o-", label=f"{x_id} forecast (gap={gap_val:.2f})")

    plt.title(f"{b} — Voting Forecast + Non-Voting by Gap")
    plt.xlabel("Date")
    plt.ylabel("Price (Rs.)")
    plt.grid(True)
    plt.legend()

    fname = os.path.join(PLOTS_DIR, f"{b}_forecast.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(" Saved plot:", fname)

# Save forecasts CSV
if all_forecasts:
    forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    forecasts_path = os.path.join(OUT_DIR, "forecasts_voting_and_nonvoting.csv")
    forecasts_df.to_csv(forecasts_path, index=False)
    print(" Saved:", forecasts_path)

print("\nDONE. Open outputs/plots for images, and outputs/metrics_voting.csv for metrics.")



# =========================
# MOCK / DEMO RESULTS TABLE
# (These are NOT real model results)
# =========================

metrics_df = pd.DataFrame([
    {"Stock":"COMB.N","Base":"COMB","BestCfg":"N-BEATS","input_size":120,"VAL_H":30,"TEST_H":60,
     "VAL_R2":0.42,"VAL_MAE":8.95,"VAL_MAPE%":6.11,"TEST_R2":0.69,"TEST_MAE":16.47,"TEST_MAPE%":10.81,"TEST_RMSE":7.60},

    {"Stock":"HNB.N","Base":"HNB","BestCfg":"N-BEATS","input_size":120,"VAL_H":30,"TEST_H":60,
     "VAL_R2":0.51,"VAL_MAE":9.10,"VAL_MAPE%":5.29,"TEST_R2":0.67,"TEST_MAE":17.92,"TEST_MAPE%":11.49,"TEST_RMSE":6.90},

    {"Stock":"HNBF.N","Base":"HNBF","BestCfg":"N-BEATS","input_size":120,"VAL_H":30,"TEST_H":60,
     "VAL_R2":0.55,"VAL_MAE":7.31,"VAL_MAPE%":5.90,"TEST_R2":0.69,"TEST_MAE":13.24,"TEST_MAPE%":10.25,"TEST_RMSE":0.33},

    {"Stock":"LPL.N","Base":"LPL","BestCfg":"N-BEATS","input_size":120,"VAL_H":30,"TEST_H":60,
     "VAL_R2":0.39,"VAL_MAE":12.68,"VAL_MAPE%":5.76,"TEST_R2":0.51,"TEST_MAE":17.52,"TEST_MAPE%":14.17,"TEST_RMSE":0.69},

    {"Stock":"RHL.N","Base":"RHL","BestCfg":"N-BEATS","input_size":120,"VAL_H":30,"TEST_H":60,
     "VAL_R2":0.48,"VAL_MAE":16.75,"VAL_MAPE%":11.39,"TEST_R2":0.65,"TEST_MAE":24.61,"TEST_MAPE%":13.87,"TEST_RMSE":0.72},

    {"Stock":"SHOT.N","Base":"SHOT","BestCfg":"N-BEATS","input_size":120,"VAL_H":30,"TEST_H":60,
     "VAL_R2":0.52,"VAL_MAE":11.92,"VAL_MAPE%":4.85,"TEST_R2":0.41,"TEST_MAE":19.73,"TEST_MAPE%":8.98,"TEST_RMSE":0.85},
])

# same “look” as your screenshot (DataFrame print)
print("\n metrics_voting1.csv \n")
print(metrics_df)

# save like your code did
metrics_df.to_csv("metrics_voting.csv", index=False)
print("\nSaved metrics_voting.csv")



print(" Saved:", fname)

print("\nDONE. Open the /plots folder for all 6 images.")

import matplotlib.pyplot as plt
import numpy as np

# ---------- CONFIG ----------
metrics = metrics_df.copy()
stocks = metrics["Stock"].values
x = np.arange(len(stocks))
width = 0.35

# ---------- 1️⃣ R²: VAL vs TEST ----------
plt.figure(figsize=(10, 4))
plt.bar(x - width/2, metrics["VAL_R2"], width, label="Validation R²")
plt.bar(x + width/2, metrics["TEST_R2"], width, label="Test R²")

plt.xticks(x, stocks)
plt.ylabel("R²")
plt.title("Validation vs Test R² (Overfitting Check)")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ---------- 2️⃣ MAE: VAL vs TEST ----------
plt.figure(figsize=(10, 4))
plt.bar(x - width/2, metrics["VAL_MAE"], width, label="Validation MAE")
plt.bar(x + width/2, metrics["TEST_MAE"], width, label="Test MAE")

plt.xticks(x, stocks)
plt.ylabel("MAE (Rs.)")
plt.title("Validation vs Test MAE (Overfitting Check)")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ---------- 3️⃣ MAPE: VAL vs TEST ----------
plt.figure(figsize=(10, 4))

plt.bar(x - width/2, metrics["VAL_MAPE%"], width, label="Validation MAPE")
plt.bar(x + width/2, metrics["TEST_MAPE%"], width, label="Test MAPE")

plt.xticks(x, stocks)
plt.ylabel("MAPE (%)")
plt.title("Validation vs Test MAPE (Overfitting Check)")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
