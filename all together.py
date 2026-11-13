import os
import subprocess
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- your 3 scripts (keep your names)
SCRIPT_NBEATS = os.path.join(BASE_DIR, "main.py")
SCRIPT_DCF    = os.path.join(BASE_DIR, "dcf_analysis.py")
SCRIPT_TECH   = os.path.join(BASE_DIR, "indicator.py")

# -------------------------
# OPTION A: RUN/SKIP SWITCHES
# -------------------------
RUN_NBEATS = False   # True only when you want to regenerate metrics_voting.csv
RUN_DCF    = True
RUN_TECH   = True

def run_script(path):
    print("\n==============================")
    print("RUN:", os.path.basename(path))
    print("==============================")
    subprocess.run(["python", path], check=True)

def safe_read_csv(path):
    if not os.path.exists(path):
        print("❌ Missing output file:", path)
        return None
    try:
        df = pd.read_csv(path)
        if df.shape[0] == 0:
            print("⚠️ File exists but is empty:", path)
        return df
    except Exception as e:
        print("❌ Failed reading:", path, "|", e)
        return None

def base_from_ticker(t):
    t = str(t).strip()
    return t.split(".")[0] if "." in t else t

def ensure_cols(df, required, df_name):
    """Create missing columns as NaN so merges/prints won't crash."""
    if df is None:
        return None
    for c in required:
        if c not in df.columns:
            df[c] = np.nan
            print(f"⚠️ Added missing column '{c}' in {df_name} as NaN")
    return df

def pick_existing_cols(df, cols):
    """Return only columns that exist (avoids KeyError)."""
    return [c for c in cols if c in df.columns]

# -------------------------
# 1) RUN MODULES (OPTION A)
# -------------------------
if RUN_NBEATS:
    run_script(SCRIPT_NBEATS)
else:
    print("\n(SKIP) main.py (N-BEATS) not running. Using existing metrics_voting.csv")

if RUN_DCF:
    run_script(SCRIPT_DCF)
else:
    print("\n(SKIP) dcf_analusis.py not running. Using existing dcf_outputs/dcf_summary.csv")

if RUN_TECH:
    run_script(SCRIPT_TECH)
else:
    print("\n(SKIP) indicator.py not running. Using existing indicators_outputs/*.csv")

# -------------------------
# 2) LOAD OUTPUT CSVs
# -------------------------
metrics_voting = safe_read_csv(os.path.join(BASE_DIR, "metrics_voting.csv"))
dcf_summary    = safe_read_csv(os.path.join(BASE_DIR, "dcf_outputs", "dcf_summary.csv"))
reco_table     = safe_read_csv(os.path.join(BASE_DIR, "indicators_outputs", "recommendations_table.csv"))
best_strat     = safe_read_csv(os.path.join(BASE_DIR, "indicators_outputs", "best_strategy_per_ticker.csv"))

if metrics_voting is None or dcf_summary is None or reco_table is None:
    raise SystemExit(
        "Some required outputs are missing.\n"
        "Need:\n"
        " - metrics_voting.csv\n"
        " - dcf_outputs/dcf_summary.csv\n"
        " - indicators_outputs/recommendations_table.csv\n"
    )

# -------------------------
# 3) NORMALIZE KEYS (Base)
# -------------------------
metrics_voting = ensure_cols(metrics_voting, ["Stock"], "metrics_voting")
metrics_voting["Base"] = metrics_voting["Stock"].apply(base_from_ticker)

dcf_summary = ensure_cols(dcf_summary, ["Stock"], "dcf_summary")
dcf_summary["Base"] = dcf_summary["Stock"].astype(str).str.strip()

reco_table = ensure_cols(reco_table, ["Base"], "reco_table")
reco_table["Base"] = reco_table["Base"].astype(str).str.strip()

# BEST STRATEGY: normalize columns
if best_strat is not None:
    if "Ticker" in best_strat.columns and "Base" not in best_strat.columns:
        best_strat["Base"] = best_strat["Ticker"].apply(base_from_ticker)

    rename_map = {}
    if "Best_Strategy" not in best_strat.columns:
        if "Strategy" in best_strat.columns:
            rename_map["Strategy"] = "Best_Strategy"
        elif "BestStrategy" in best_strat.columns:
            rename_map["BestStrategy"] = "Best_Strategy"
        elif "best_strategy" in best_strat.columns:
            rename_map["best_strategy"] = "Best_Strategy"

    best_strat = best_strat.rename(columns=rename_map)

best_strat = best_strat if best_strat is not None else pd.DataFrame({"Base": []})
best_strat = ensure_cols(
    best_strat,
    ["Base", "Best_Strategy", "Best_Total_Return_%", "Best_Max_Drawdown_%", "Best_Sharpe", "Best_Num_Trades", "Best_Win_Rate_%"],
    "best_strat"
)

# Ensure DCF columns exist
dcf_summary = ensure_cols(
    dcf_summary,
    ["Market_Price", "DCF_Fair_Value_per_Share", "PE_Ratio", "TTM_EPS_Sum", "g_%", "r_%"],
    "dcf_summary"
)

# Ensure metrics columns exist
metrics_voting = ensure_cols(
    metrics_voting,
    ["TEST_R2", "TEST_MAE", "TEST_MAPE%", "TEST_RMSE", "BestCfg", "input_size", "VAL_H", "TEST_H"],
    "metrics_voting"
)

# -------------------------
# 4) MERGE INTO MASTER TABLE
# -------------------------
master = reco_table.merge(
    best_strat[["Base","Best_Strategy","Best_Total_Return_%","Best_Max_Drawdown_%","Best_Sharpe","Best_Num_Trades","Best_Win_Rate_%"]],
    on="Base", how="left"
).merge(
    dcf_summary[["Base","Market_Price","DCF_Fair_Value_per_Share","PE_Ratio","TTM_EPS_Sum","g_%","r_%"]],
    on="Base", how="left"
).merge(
    metrics_voting[["Base","TEST_R2","TEST_MAE","TEST_MAPE%","TEST_RMSE","BestCfg","input_size","VAL_H","TEST_H"]],
    on="Base", how="left"
)

# -------------------------
# 5) NORMALIZED HYBRID SCORING (0–100)
# -------------------------
# DCF undervaluation %
master["DCF_Undervaluation_%"] = np.where(
    (master["Market_Price"].notna()) &
    (master["DCF_Fair_Value_per_Share"].notna()) &
    (master["Market_Price"] > 0),
    (master["DCF_Fair_Value_per_Share"] / master["Market_Price"] - 1) * 100,
    np.nan
)

# A) DCF score: cap extremes and scale to 0..100
dcf_clip = master["DCF_Undervaluation_%"].clip(-50, 100)
master["DCF_Score"] = (dcf_clip + 50) / 150 * 100  # -50->0, +100->100

# B) Technical score: combine return + sharpe then cap 0..100
ret_part = master["Best_Total_Return_%"].clip(-20, 50) * 1.2
shp_part = master["Best_Sharpe"].clip(-1, 2) * 20
master["Tech_Score_Norm"] = (ret_part + shp_part).clip(0, 100)

# C) Forecast score: clamp R2 to 0..1 and scale
master["Forecast_Score"] = master["TEST_R2"].clip(0, 1) * 100

# Final hybrid score (0..100)
master["Hybrid_Score"] = (
    0.40 * master["DCF_Score"].fillna(50) +
    0.35 * master["Tech_Score_Norm"].fillna(50) +
    0.25 * master["Forecast_Score"].fillna(50)
)

# -------------------------
# 6) LABEL FROM SCORE + DEDUP ONE ROW PER BASE
# -------------------------
def hybrid_label(score):
    if score >= 80:
        return "STRONG BUY"
    if score >= 65:
        return "BUY"
    if score >= 45:
        return "HOLD"
    if score >= 30:
        return "SELL"
    return "STRONG SELL"

master["Hybrid_Recommendation"] = master["Hybrid_Score"].apply(hybrid_label)

# Deduplicate: keep best row per Base
master = (
    master.sort_values("Hybrid_Score", ascending=False)
          .groupby("Base", as_index=False)
          .first()
)

master = master.sort_values("Hybrid_Score", ascending=False).reset_index(drop=True)

# -------------------------
# 7) SAVE + SAFE PRINT
# -------------------------
out_path = os.path.join(BASE_DIR, "MASTER_summary.csv")
master.to_csv(out_path, index=False)
print("\n✅ SAVED MASTER FILE:", out_path)

show_cols = [
    "Base","Hybrid_Recommendation","Hybrid_Score",
    "Latest_Signal","DCF_Undervaluation_%",
    "Best_Strategy","Best_Total_Return_%","Best_Sharpe",
    "TEST_R2","TEST_MAPE%"
]
show_cols = pick_existing_cols(master, show_cols)

print("\n===== MASTER SUMMARY (Top 20) =====")
print(master[show_cols].head(20))

print("\nℹ️ Columns available in MASTER_summary.csv:")
print(list(master.columns))




