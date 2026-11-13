# ============================================================
# DCF VALUATION — VOTING SHARES ONLY (.N / base tickers)
# Input : Financial data stocks_all.csv
# Output: dcf_outputs/dcf_summary.csv + charts
#
# Fixes included:
#  Adds latest-quarter financial columns to dcf_summary.csv:
#    Cash_and_Cash_Equivalents (LKR), NAVPS (Rs.), Revenue (LKR),
#    Operating_Profit (LKR), Net_Income (LKR)
#  Robust column matching (handles spaces/underscores/case differences)
#  Handles numbers with commas (e.g., "1,234,567")
#  Prevents matplotlib savefig crash in system_runner by:
#    - forcing Agg backend
#    - removing plt.show()
#  Silences date parse warnings with dayfirst=True
# ============================================================

import os
import sys
import numpy as np
import pandas as pd

#  Force non-interactive backend BEFORE pyplot import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------
FIN_CSV_PATH = r"C:\Users\nirma\OneDrive\Desktop(1)\BScDS Project 1\Financial data stocks_all.csv"
# ---------------------------
# 1) CONFIG
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)


# -----------------------------
# OUTPUT DIRS (ALWAYS NEXT TO THIS PY FILE)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "dcf_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# DCF assumptions
G_TERMINAL = 0.03
FORECAST_YEARS = 5

# Voting stocks (base tickers you want)
TARGET = ["COMB", "HNB", "HNBF", "LPL", "RHL", "SHOT"]

# If you INSIST: for March-year-end companies use last 2 quarters for "TTM" (NOT standard)
MARCH_USE_2Q = False

# -----------------------
# HELPERS (column matching)
# -----------------------
def _norm_key(x: str) -> str:
    """Normalize a column name for fuzzy matching."""
    return str(x).strip().lower().replace(" ", "").replace("_", "")

def find_col(df_: pd.DataFrame, wanted: str):
    """
    Find actual column name in df that matches wanted, ignoring spaces/underscores/case.
    Returns actual column name (string) or None.
    """
    w = _norm_key(wanted)
    for c in df_.columns:
        if _norm_key(c) == w:
            return c
    return None

def clean_numeric_series(s: pd.Series) -> pd.Series:
    """Remove commas/spaces and convert to numeric safely."""
    s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    return pd.to_numeric(s, errors="coerce")

def get_latest_value(latest_row: pd.DataFrame, col_name: str):
    """Return latest value for a column if present; else None."""
    if not col_name or latest_row is None or latest_row.empty:
        return None
    if col_name not in latest_row.columns:
        return None
    v = latest_row[col_name].iloc[0]
    if pd.isna(v):
        return None
    return v

def safe_savefig(path: str, dpi: int = 180):
    """Save figure safely without crashing system_runner."""
    try:
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
    except Exception as e:
        print(f"⚠️ Chart save failed for {os.path.basename(path)}: {e}")
    finally:
        plt.close()

# -----------------------
# LOAD DATA
# -----------------------
if not os.path.exists(FIN_CSV_PATH):
    print("❌ File not found:", FIN_CSV_PATH)
    sys.exit(1)

df = pd.read_csv(FIN_CSV_PATH)

# Clean header spaces
df.columns = [str(c).strip() for c in df.columns]

# Required columns (robust match)
stock_col = find_col(df, "Stock")
report_end_col = find_col(df, "Report_End")
quarter_col = find_col(df, "Quarter")
eps_col = find_col(df, "EPS (Rs.)")
navps_col = find_col(df, "NAVPS (Rs.)")

required_real = [stock_col, report_end_col, quarter_col, eps_col, navps_col]
if any(c is None for c in required_real):
    print("❌ Missing one of required columns (after matching):")
    print("Need: Stock, Report_End, Quarter, EPS (Rs.), NAVPS (Rs.)")
    print("Your columns are:", list(df.columns))
    sys.exit(1)

# Rename to standard internal names for easier code
df = df.rename(columns={
    stock_col: "Stock",
    report_end_col: "Report_End",
    quarter_col: "Quarter",
    eps_col: "EPS (Rs.)",
    navps_col: "NAVPS (Rs.)",
})

# Parse dates (dayfirst avoids warnings for DD/MM/YYYY)
df["Stock"] = df["Stock"].astype(str).str.strip()
df["Report_End"] = pd.to_datetime(df["Report_End"], errors="coerce", dayfirst=True)
df["Quarter"] = df["Quarter"].astype(str).str.strip()

# Optional FY_End
fy_end_real = find_col(df, "FY_End")
if fy_end_real and fy_end_real in df.columns:
    df = df.rename(columns={fy_end_real: "FY_End"})
    df["FY_End"] = pd.to_datetime(df["FY_End"], errors="coerce", dayfirst=True)

# Numeric conversions (EPS/NAVPS + financial columns)
for c in ["EPS (Rs.)", "NAVPS (Rs.)"]:
    df[c] = clean_numeric_series(df[c])

# Financial columns you want (robust match + rename)
wanted_fin_cols = {
    "Cash_and_Cash_Equivalents (LKR)": "Cash_and_Cash_Equivalents (LKR)",
    "Revenue (LKR)": "Revenue (LKR)",
    "Operating_Profit (LKR)": "Operating_Profit (LKR)",
    "Net_Income (LKR)": "Net_Income (LKR)",
}
real_to_std = {}
for wanted, std in wanted_fin_cols.items():
    real = find_col(df, wanted)
    if real and real in df.columns:
        real_to_std[real] = std

if real_to_std:
    df = df.rename(columns=real_to_std)
    for std in real_to_std.values():
        df[std] = clean_numeric_series(df[std])

df = df.dropna(subset=["Stock", "Report_End"]).copy()

# Keep only target tickers
df = df[df["Stock"].isin(TARGET)].copy()

print(" Loaded rows:", len(df))
print(" Stocks found:", sorted(df["Stock"].unique()))

if df.empty:
    print("❌ No rows after filtering TARGET tickers. Check Stock codes in CSV.")
    sys.exit(1)

# -----------------------
# DCF helpers
# -----------------------
def is_march_year_end(stock_df: pd.DataFrame) -> bool:
    if "FY_End" in stock_df.columns:
        fy_end = stock_df["FY_End"].dropna()
        if len(fy_end) > 0:
            return int(fy_end.iloc[-1].month) == 3
    return False

def pick_ttm_rows(stock_df: pd.DataFrame) -> pd.DataFrame:
    s = stock_df.sort_values("Report_End").dropna(subset=["EPS (Rs.)"]).copy()
    if s.empty:
        return s
    n = 4
    if MARCH_USE_2Q and is_march_year_end(s):
        n = 2
    n = min(n, len(s))
    return s.tail(n).copy()

def dcf_from_eps_per_share(current_eps_annual, g, r, gt, years=5):
    if current_eps_annual is None or not np.isfinite(current_eps_annual):
        return np.nan
    if current_eps_annual <= 0:
        return np.nan
    if r <= gt:
        return np.nan

    eps_fore = [current_eps_annual * (1 + g) ** t for t in range(1, years + 1)]
    terminal = eps_fore[-1] * (1 + gt) / (r - gt)

    pv = 0.0
    for t, cf in enumerate(eps_fore, start=1):
        pv += cf / (1 + r) ** t
    pv += terminal / (1 + r) ** years
    return float(pv)

# -----------------------
# MANUAL ASSUMPTIONS
# -----------------------
MANUAL_G = {"COMB": 0.12, "HNB": 0.12, "HNBF": 0.10, "LPL": 0.12, "RHL": 0.12, "SHOT": 0.10}
MANUAL_R = {"COMB": 0.15, "HNB": 0.15, "HNBF": 0.18, "LPL": 0.10, "RHL": 0.18, "SHOT": 0.18}
MARKET_PRICE = {"COMB": 193.00, "HNB": 390.00, "HNBF": 8.50, "LPL": 15.80, "RHL": 30.00, "SHOT": 25.00}

# -----------------------
# MAIN LOOP + VISUALS
# -----------------------
summary_rows = []

for stock in TARGET:
    s = df[df["Stock"] == stock].copy()
    if s.empty:
        continue

    s = s.sort_values("Report_End").copy()

    # latest quarter row
    latest_row = s.tail(1)

    cash_eq = get_latest_value(latest_row, "Cash_and_Cash_Equivalents (LKR)")
    navps_latest = get_latest_value(latest_row, "NAVPS (Rs.)")
    revenue = get_latest_value(latest_row, "Revenue (LKR)")
    op_profit = get_latest_value(latest_row, "Operating_Profit (LKR)")
    net_income = get_latest_value(latest_row, "Net_Income (LKR)")

    eps_q = s[["Report_End", "Quarter", "EPS (Rs.)"]].dropna().copy()
    nav_q = s[["Report_End", "Quarter", "NAVPS (Rs.)"]].dropna().copy()

    # Rolling TTM EPS
    eps_for_roll = eps_q.set_index("Report_End")["EPS (Rs.)"].sort_index()
    ttm_roll = eps_for_roll.rolling(4).sum().dropna()

    # TTM EPS from last N quarters
    ttm_rows = pick_ttm_rows(s)
    quarters_used = len(ttm_rows)
    ttm_eps_sum = float(ttm_rows["EPS (Rs.)"].sum()) if quarters_used > 0 else np.nan

    g = MANUAL_G.get(stock, 0.06)
    r = MANUAL_R.get(stock, 0.18)

    fair = dcf_from_eps_per_share(ttm_eps_sum, g=g, r=r, gt=G_TERMINAL, years=FORECAST_YEARS)

    last_report = ttm_rows["Report_End"].max() if quarters_used > 0 else s["Report_End"].max()

    mkt_price = MARKET_PRICE.get(stock, np.nan)
    pe_ratio = (mkt_price / ttm_eps_sum) if (np.isfinite(mkt_price) and np.isfinite(ttm_eps_sum) and ttm_eps_sum > 0) else np.nan

    summary_rows.append({
        "Stock": stock,
        "Last_Report_End": str(last_report.date()) if pd.notna(last_report) else None,
        "Quarters_Used_for_TTM": quarters_used,
        "TTM_EPS_Sum": None if not np.isfinite(ttm_eps_sum) else round(ttm_eps_sum, 4),
        "Market_Price": None if not np.isfinite(mkt_price) else round(mkt_price, 2),
        "PE_Ratio": None if not np.isfinite(pe_ratio) else round(pe_ratio, 2),
        "g_%": round(g * 100, 2),
        "r_%": round(r * 100, 2),
        "gt_%": round(G_TERMINAL * 100, 2),
        "DCF_Fair_Value_per_Share": None if not np.isfinite(fair) else round(fair, 2),

        # Latest-quarter financials
        "Cash_and_Cash_Equivalents (LKR)": cash_eq,
        "NAVPS (Rs.)": navps_latest,
        "Revenue (LKR)": revenue,
        "Operating_Profit (LKR)": op_profit,
        "Net_Income (LKR)": net_income,
    })

    # ---------- Charts ----------
    if not eps_q.empty:
        plt.figure(figsize=(9, 4))
        plt.plot(eps_q["Report_End"], eps_q["EPS (Rs.)"], marker="o")
        plt.title(f"{stock} — Quarterly EPS (Rs.)")
        plt.xlabel("Report End")
        plt.ylabel("EPS (Rs.)")
        plt.grid(True, alpha=0.3)
        safe_savefig(os.path.join(OUT_DIR, f"{stock}_eps_quarterly.png"))

    if len(ttm_roll) > 0:
        plt.figure(figsize=(9, 4))
        plt.plot(ttm_roll.index, ttm_roll.values, marker="o")
        plt.title(f"{stock} — Rolling TTM EPS (sum of last 4 quarters)")
        plt.xlabel("Report End")
        plt.ylabel("TTM EPS (Rs.)")
        plt.grid(True, alpha=0.3)
        safe_savefig(os.path.join(OUT_DIR, f"{stock}_eps_ttm_rolling.png"))

    if not nav_q.empty:
        plt.figure(figsize=(9, 4))
        plt.plot(nav_q["Report_End"], nav_q["NAVPS (Rs.)"], marker="o")
        plt.title(f"{stock} — Quarterly NAVPS (Rs.)")
        plt.xlabel("Report End")
        plt.ylabel("NAVPS (Rs.)")
        plt.grid(True, alpha=0.3)
        safe_savefig(os.path.join(OUT_DIR, f"{stock}_navps_quarterly.png"))

# Save summary
out = pd.DataFrame(summary_rows)
if out.empty:
    print("❌ No output rows generated.")
    sys.exit(1)

out = out.sort_values("Stock").reset_index(drop=True)
out_path = os.path.join(OUT_DIR, "dcf_summary.csv")
out.to_csv(out_path, index=False)

print("\n DCF done.")
print(out)
print("\n Saved:", out_path)
print(" Charts saved in:", OUT_DIR)

# Combined bar chart (save only)
plt.figure(figsize=(8, 4))
plt.bar(out["Stock"], out["DCF_Fair_Value_per_Share"].fillna(0))
plt.title("DCF Fair Value per Share (EPS-based proxy)")
plt.xlabel("Stock")
plt.ylabel("DCF Fair Value (Rs.)")
safe_savefig(os.path.join(OUT_DIR, "dcf_fair_value_bar.png"))
