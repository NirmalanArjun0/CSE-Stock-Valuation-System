import os
import sys
import time
import uuid
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import re

import pandas as pd
import requests
from flask import Flask, request, jsonify, make_response

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# -----------------------------
# Scripts (same folder)
# -----------------------------
SCRIPT_NBEATS = BASE_DIR / "main.py"
SCRIPT_DCF    = BASE_DIR / "dcf_analysis.py"
SCRIPT_TECH   = BASE_DIR / "indicator.py"
SCRIPT_ALL    = BASE_DIR / "all together.py"
SCRIPT_SYSTEM = BASE_DIR / "system_runner.py"  # optional

# -----------------------------
# Output files
# -----------------------------
OUT_MASTER  = BASE_DIR / "MASTER_summary.csv"
OUT_METRICS = BASE_DIR / "metrics_voting.csv"
OUT_DCF     = BASE_DIR / "dcf_outputs" / "dcf_summary.csv"
OUT_RECO    = BASE_DIR / "indicators_outputs" / "recommendations_table.csv"
OUT_BEST    = BASE_DIR / "indicators_outputs" / "best_strategy_per_ticker.csv"

# Charts folders
FOLDER_DCF         = BASE_DIR / "dcf_outputs"
FOLDER_TECH        = BASE_DIR / "indicators_outputs"
FOLDER_NBEATS_PLOT = Path(r"F:\3 BSc.DS\BScDS Project\plots")

# Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"

# -----------------------------
# In-memory chat sessions
# -----------------------------
SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_SECONDS = 60 * 60 * 6  # 6 hours

# -----------------------------
# Hide columns in MASTER_summary UI
# -----------------------------
MASTER_HIDE_COLS = [
    # Risk / strategy
    "Risk_Setting", "RISK_SETTING", "Risk_Settings", "RISK_SETTINGS",
    "Best_Num_Trades", "BEST_NUM_TRADES",
    "Best_Win_Rate_%", "BEST_WIN_RATE_%", "Last_Date",

    # g / r
    "g%", "G%", "g_%", "G_%",
    "r%", "R%", "r_%", "R_%",

    # Test metrics
    "Test_R2", "TEST_R2",
    "Test_MAE", "TEST_MAE",
    "Test_MAPE%", "TEST_MAPE%",
    "Test_RMSE", "TEST_RMSE",

    # Config
    "BestCfg", "BESTCFG",
    "input_size", "INPUT_SIZE",
    "VAL_H", "val_h",
    "TEST_H", "test_h",

    # DCF / Tech / Forecast / Hybrid
    "DCF_Undervaluation_%", "DCF_UNDERVALUATION_%",
    "DCF_Score", "DCF_SCORE",
    "Tech_Score_Norm", "TECH_SCORE_NORM",
    "Forecast_Score", "FORECAST_SCORE",
    "Hybrid_Score", "HYBRID_SCORE",

    # Remove extra tech cols from master (your request)
    "Last_Close",
    "RSI14",
    "RSI_Status",
    "MACD_Status",
    "Best_Strategy_x",
    "Latest_Signal",
    "SR_Levels",
    "Pivot_P",
    "Fib_Levels",
    "Best_Strategy_y",
    "Best_Total_Return_%",
    "Best_Max_Drawdown_%",
    "Best_Sharpe",
]

RECO_HIDE_COLS = [
    "Risk_Setting", "RISK_SETTING", "Risk_Settings", "RISK_SETTINGS",
]

# =============================
# ✅ System prompt (single source of truth)
# =============================
SYSTEM_PROMPT = (
    "You are an expert finance, investing, and data-science assistant embedded inside a stock analysis dashboard.\n"
    "You can answer BOTH:\n"
    "A) Questions about stocks contained in the dashboard CSV outputs (MASTER_summary, recommendations_table, dcf_summary, metrics_voting)\n"
    "B) General finance/market/valuation questions (DCF, PE, PBV, growth, risk, portfolio, technical indicators, macro impacts, etc.)\n"
    "\n"
    "DATA INTEGRITY (VERY IMPORTANT)\n"
    "1) If the user asks about a stock/ticker that exists in the provided tables, use the table data as the primary source of truth.\n"
    "2) You may explain valuation concepts beyond the tables (DCF logic, assumptions, risks, industry context), but NEVER invent missing numbers.\n"
    "3) If the user asks for a number (EPS, PE, revenue, fair value, etc.) and it is NOT in the tables, say it is not available in the provided tables and ask what source/time period they want.\n"
    "4) Do not hallucinate: never claim 'the table shows X' unless that value is actually in the provided context.\n"
    "5) When you reference table information, explicitly say 'From your dashboard tables:' then mention the exact field names you used.\n"
    "\n"
    "VALUATION & INVESTMENT ANALYSIS STYLE\n"
    "6) Use a structured framework when relevant: Business quality + Financial strength + Valuation + Momentum/Technical + Forecast + Risk.\n"
    "7) Forecasts are model projections, not certainty. Interpret reliability using metrics (R²/MAE/MAPE/RMSE) if available.\n"
    "8) Technical indicators/signals are probabilities; never guarantee outcomes.\n"
    "9) If table values look suspicious (negative R², huge errors, missing quarters), flag it and explain possible reasons.\n"
    "\n"
    "USER EXPERIENCE\n"
    "10) Ask at most 1–2 follow-up questions if needed (ticker, timeframe, voting vs non-voting, etc.).\n"
    "11) Use simple language; avoid heavy math unless asked.\n"
    "12) Provide actionable next steps.\n"
    "13) If the user asks for financial advice, add a light disclaimer: educational info only, not guaranteed.\n"
    "\n"
    "OUTPUT FORMAT\n"
    "Use this structure by default:\n"
    "Summary:\n"
    "- ...\n"
    "Key points:\n"
    "- ...\n"
    "Risks:\n"
    "- ...\n"
    "Next steps:\n"
    "- ...\n"
)

STOCK_REPORT_INSTRUCTIONS = (
    "When the user asks about a specific stock/ticker that exists in the tables, give a more detailed explanation (not only short bullet points).\n"
    "Explain what the dashboard outputs mean (Hybrid decision, DCF, Technical, Forecast, Strategy stats).\n"
    "Use table numbers as ground truth.\n"
    "If a requested number is missing, clearly say it is not available in the provided tables.\n"
)

# =============================
# Helpers
# =============================
def now_ts() -> float:
    return time.time()

def cleanup_sessions():
    cutoff = now_ts() - SESSION_TTL_SECONDS
    dead = [sid for sid, v in SESSIONS.items() if v.get("last_seen", 0) < cutoff]
    for sid in dead:
        SESSIONS.pop(sid, None)

def get_session_id() -> str:
    cleanup_sessions()
    sid = request.cookies.get("sid")
    if not sid or sid not in SESSIONS:
        sid = str(uuid.uuid4())
        SESSIONS[sid] = {
            "last_seen": now_ts(),
            "chat_messages": [
                {"role": "system", "content": SYSTEM_PROMPT}
            ],
            # ✅ track last base we injected context for (prevents repeating same stock block every message)
            "last_context_base": None,
        }
    else:
        SESSIONS[sid]["last_seen"] = now_ts()
    return sid

def run_script(path: Path) -> Dict[str, Any]:
    if not path or not path.exists():
        return {"ok": False, "logs": f"❌ Missing script: {path}"}

    cmd = [sys.executable, str(path)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR), check=True)
        logs = ""
        if p.stdout:
            logs += "---- STDOUT ----\n" + p.stdout + "\n"
        if p.stderr:
            logs += "---- STDERR ----\n" + p.stderr + "\n"
        return {"ok": True, "logs": logs.strip() or "✅ Done."}
    except subprocess.CalledProcessError as e:
        logs = ""
        if e.stdout:
            logs += "---- STDOUT ----\n" + e.stdout + "\n"
        if e.stderr:
            logs += "---- STDERR ----\n" + e.stderr + "\n"
        return {"ok": False, "logs": logs.strip() or "❌ Script failed."}
    except Exception as e:
        return {"ok": False, "logs": f"❌ Error: {e}"}

def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            return pd.DataFrame({"_error": [f"Failed to read {path.name}: {e}"]})
    return None

def base_from_any(x: Any) -> str:
    x = str(x).strip()
    return x.split(".")[0] if "." in x else x

def ensure_base(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "Base" not in df.columns:
        if "Stock" in df.columns:
            df["Base"] = df["Stock"].apply(base_from_any)
        elif "unique_id" in df.columns:
            df["Base"] = df["unique_id"].apply(base_from_any)
    return df

def drop_cols(df: Optional[pd.DataFrame], cols_to_drop: List[str]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

def force_ticker_dot_n(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Force UI ticker to Base + .N (only for display)."""
    if df is None or df.empty:
        return df
    df = ensure_base(df).copy()
    if "Ticker" in df.columns and "Base" in df.columns:
        df["Ticker"] = df["Base"].astype(str).str.strip() + ".N"
    return df

def list_images(folder: Path) -> List[str]:
    if not folder.exists():
        return []
    exts = (".png", ".jpg", ".jpeg", ".webp")
    return [p.name for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]

def df_to_json_table(df: Optional[pd.DataFrame], max_rows=500) -> Dict[str, Any]:
    if df is None:
        return {"exists": False, "columns": [], "rows": []}
    df2 = df.head(max_rows).copy().fillna("")
    return {"exists": True, "columns": list(df2.columns), "rows": df2.astype(str).values.tolist()}

def detect_base_from_text(text: str, known_bases: List[str]) -> Optional[str]:
    """Return a Base (e.g., COMB) if mentioned in the user's text."""
    t = (text or "").upper()
    for b in known_bases:
        bb = str(b).strip().upper()
        if not bb:
            continue
        if re.search(rf"\b{re.escape(bb)}\b", t):
            return str(b).strip()
    return None

def build_context_text_for_base(
    base: Optional[str],
    df_master: Optional[pd.DataFrame],
    df_reco: Optional[pd.DataFrame],
    df_dcf: Optional[pd.DataFrame],
    df_metrics: Optional[pd.DataFrame],
) -> str:
    base = (base or "").strip()
    if not base:
        return ""

    def pick_cols(df: pd.DataFrame, wanted: List[str]) -> pd.DataFrame:
        cols = [c for c in wanted if c in df.columns]
        if not cols:
            return df
        return df[cols]

    parts: List[str] = []

    def add_block(df: Optional[pd.DataFrame], name: str, wanted_cols: List[str]):
        if df is None or df.empty:
            return
        df2 = ensure_base(df)
        if df2 is None or df2.empty or "Base" not in df2.columns:
            return
        row = df2[df2["Base"].astype(str).str.strip() == base]
        if row.empty:
            return
        row = pick_cols(row, wanted_cols).head(1)
        parts.append(f"{name}:\n{row.to_string(index=False)}")

    # MASTER (hybrid final)
    add_block(df_master, "MASTER_summary.csv (Hybrid output)", [
        "Base", "Ticker",
        "Hybrid_Recommendation", "Hybrid_Score",
        "DCF_Score", "DCF_Undervaluation_%",
        "Tech_Score_Norm",
        "Forecast_Score",
    ])

    # RECO (signals/strategy)
    add_block(df_reco, "recommendations_table.csv (Signals/Strategy)", [
        "Base", "Ticker",
        "Latest_Signal", "Best_Strategy",
        "Best_Total_Return_%", "Best_Max_Drawdown_%", "Best_Sharpe",
        "Best_Num_Trades", "Best_Win_Rate_%"
    ])

    # DCF (valuation + financials)
    add_block(df_dcf, "dcf_summary.csv (Valuation + Financials)", [
        "Base", "Ticker",
        "Fair_Value", "Current_Price", "DCF_Undervaluation_%",
        "NAVPS (Rs.)", "Cash_and_Cash_Equivalents (LKR)",
        "Revenue (LKR)", "Operating_Profit (LKR)", "Net_Income (LKR)"
    ])

    # METRICS (N-BEATS validation/test)
    add_block(df_metrics, "metrics_voting.csv (Forecast metrics)", [
        "Base", "Stock",
        "VAL_R2", "VAL_MAE", "VAL_MAPE%", "VAL_RMSE",
        "TEST_R2", "TEST_MAE", "TEST_MAPE%", "TEST_RMSE",
        "input_size", "VAL_H", "TEST_H", "BestCfg"
    ])

    return "\n\n".join([p for p in parts if p]).strip()

def trim_chat(messages: List[Dict[str, str]], keep_last: int = 28) -> List[Dict[str, str]]:
    """Keep first system message + last N messages to control context size."""
    if not messages:
        return messages
    sysmsg = [messages[0]] if messages[0].get("role") == "system" else []
    rest = messages[1:] if sysmsg else messages
    if len(rest) <= keep_last:
        return sysmsg + rest
    return sysmsg + rest[-keep_last:]

def ollama_chat(model: str, messages: list, temperature=0.4) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_ctx": 8192,
            "top_p": 0.9,
            "repeat_penalty": 1.1
        }
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]

# --- Enrichment: add financial cols from dcf_summary into MASTER view
FIN_COLS_FROM_DCF = [
    "Cash_and_Cash_Equivalents (LKR)",
    "NAVPS (Rs.)",
    "Revenue (LKR)",
    "Operating_Profit (LKR)",
    "Net_Income (LKR)",
]

def enrich_master_with_dcf_financials(df_master: Optional[pd.DataFrame], df_dcf: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df_master is None or df_master.empty or df_dcf is None or df_dcf.empty:
        return df_master
    dfm = ensure_base(df_master).copy()
    dfd = ensure_base(df_dcf).copy()
    if "Base" not in dfm.columns or "Base" not in dfd.columns:
        return df_master
    keep_cols = ["Base"] + [c for c in FIN_COLS_FROM_DCF if c in dfd.columns]
    if len(keep_cols) == 1:
        return df_master
    dfd_small = dfd[keep_cols].drop_duplicates(subset=["Base"]).copy()
    return dfm.merge(dfd_small, on="Base", how="left")

# --- Enrichment: add best stats into recommendations_table view
BEST_STATS_COLS = ["Best_Total_Return_%", "Best_Max_Drawdown_%", "Best_Sharpe"]

def enrich_reco_with_best_stats(df_reco: Optional[pd.DataFrame], df_best: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df_reco is None or df_reco.empty or df_best is None or df_best.empty:
        return df_reco
    dr = ensure_base(df_reco).copy()
    db = ensure_base(df_best).copy()
    if "Base" not in dr.columns or "Base" not in db.columns:
        return df_reco
    keep_cols = ["Base"] + [c for c in BEST_STATS_COLS if c in db.columns]
    if len(keep_cols) == 1:
        return df_reco
    db_small = db[keep_cols].drop_duplicates(subset=["Base"]).copy()
    return dr.merge(db_small, on="Base", how="left")

# =============================
# Routes
# =============================
@app.route("/")
def index():
    sid = get_session_id()

    html = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Hybrid Stock Decision Dashboard</title>
<style>
:root { --bg:#f6f7fb; --card:#fff; --text:#111; --muted:#6b7280; --primary:#2563eb; --border:#e5e7eb; --shadow:0 10px 25px rgba(0,0,0,.06); }
* { box-sizing:border-box; }
body { margin:0; font-family: Arial, sans-serif; background:var(--bg); color:var(--text); }
.layout { display:flex; min-height:100vh; }
.sidebar{ width:290px; background:#0b1220; color:#e5e7eb; padding:16px; position:sticky; top:0; height:100vh; overflow:auto; }
.brand { font-size:18px; font-weight:700; line-height:1.2; margin-bottom:8px; }
.caption { color:#a1a1aa; font-size:14px; margin-bottom:14px; }
.side-block { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.06); padding:12px; border-radius:12px; margin-bottom:12px; }
.side-title { font-size:15px; font-weight:700; margin-bottom:10px; }
.btn{ width:100%; padding:10px 12px; border:0; border-radius:10px; cursor:pointer; background:var(--primary); color:white; font-weight:700; margin-bottom:8px; }
.btn:hover { filter:brightness(.95); }
.btn.secondary { background:#111827; border:1px solid rgba(255,255,255,.08); }
label { font-size:14px; color:#cbd5e1; display:block; margin:10px 0 6px; }
input, select{ width:100%; padding:10px; border-radius:10px; border:1px solid rgba(255,255,255,.14); background:#0f172a; color:#e5e7eb; }
.main { flex:1; padding:18px; }
.header{ background:var(--card); border:1px solid var(--border); border-radius:16px; padding:16px; box-shadow:var(--shadow); margin-bottom:14px; }
.h1 { font-size:22px; font-weight:800; margin:0; }
.h2 { font-size:15px; color:var(--muted); margin:6px 0 0; }
.tabs { display:flex; gap:10px; margin:14px 0; flex-wrap:wrap; }
.tab{ padding:10px 12px; background:var(--card); border:1px solid var(--border); border-radius:999px; cursor:pointer; font-weight:700; font-size:15px; }
.tab.active { background:#eef2ff; border-color:#c7d2fe; }
.card{ background:var(--card); border:1px solid var(--border); border-radius:16px; padding:14px; box-shadow:var(--shadow); margin-bottom:14px; }
.row { display:flex; gap:12px; flex-wrap:wrap; }
.col { flex:1; min-width:320px; }
.small { font-size:14px; color:var(--muted); }
.logs{ background:#0b1220; color:#cbd5e1; padding:12px; border-radius:14px; overflow:auto; max-height:240px; border:1px solid rgba(255,255,255,.08); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; white-space:pre-wrap; }
.table-wrap { overflow:auto; border:1px solid var(--border); border-radius:12px; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { padding:8px; border-bottom:1px solid var(--border); white-space:nowrap; }
th { position:sticky; top:0; background:#fafafa; text-align:left; }
.grid-img { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap:10px; }
.grid-img img { width:100%; border-radius:14px; border:1px solid var(--border); background:white; }
.hr { height:1px; background:var(--border); margin:12px 0; }
.hidden { display:none; }
.chatbox{ height:80vh; overflow:auto; border:1px solid var(--border); border-radius:14px; padding:10px; background:#fbfbff; }
.msg { padding:10px; border-radius:12px; margin:8px 0; white-space:pre-wrap; }
.msg.user { background:#e0ecff; }
.msg.bot { background:#f3f4f6; }
.chatrow { display:flex; gap:10px; margin-top:10px; }
.chatrow input { flex:1; padding:12px; border-radius:12px; border:1px solid var(--border); background:white; color:#111; }
.chatrow button { padding:12px 14px; border-radius:12px; border:0; background:var(--primary); color:white; font-weight:800; cursor:pointer; }
.chatrow button:hover { filter:brightness(.95); }

/*  NEW: horizontal chart buttons */
.chart-buttons{
  display:flex;
  flex-wrap:wrap;
  gap:8px;
  margin-top:10px;
}
/* chart selection buttons */
/* chart selection buttons - DEFAULT */
.chip{
  border:1px solid #93c5fd;
  background:#dbeafe;        /* light blue background */
  padding:10px 14px;
  border-radius:999px;
  cursor:pointer;
  font-size:14px;            /* +2 font size */
  font-weight:700;
  color:#111827;             /* dark text */
  transition: all 0.15s ease;
}

/* hover */
.chip:hover{
  background:#bfdbfe;        /* slightly darker blue */
}

/* ACTIVE (clicked) */
.chip.active{
  background:#ffffff;        /* white background */
  border-color:#93c5fd;
  color:#2563eb;             /* light blue text */
  box-shadow: 0 0 0 2px rgba(59,130,246,0.15);
}

</style>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="brand">📈 Hybrid Stock Decision Dashboard</div>
    <div class="caption">N-BEATS + DCF + Technical + Hybrid + Ollama Chat</div>

    <div class="side-block">
      <div class="side-title">Controls</div>
      <button class="btn" onclick="runScript('nbeats')">▶ Run N-BEATS (main.py)</button>
      <button class="btn" onclick="runScript('dcf')">▶ Run DCF (dcf_analysis.py)</button>
      <button class="btn" onclick="runScript('tech')">▶ Run Technical (indicator.py)</button>
      <button class="btn" onclick="runScript('all')">▶ Run Hybrid Merge (all together.py)</button>
      <button class="btn secondary" onclick="reloadAll()">🔄 Reload Outputs</button>
      <button class="btn secondary" onclick="runScript('system')">▶ Run Final System (optional)</button>
    </div>

    <div class="side-block">
      <div class="side-title">Ollama Chat</div>
      <label>Model name</label>
      <input id="ollama_model" value="llama3" />
      <label>Temperature</label>
      <input id="ollama_temp" type="number" step="0.05" value="0.40" />
      <label><input type="checkbox" id="use_context" checked /> Add CSV context once</label>
      <div class="small">Ollama must run at http://localhost:11434</div>
    </div>
  </aside>

  <main class="main">
    <div class="header">
      <div class="h1">Dashboard</div>
      <div class="h2">Sidebar controls + 4 tabs + details + charts + chat.</div>
    </div>

    <div class="tabs">
      <div class="tab active" data-tab="tab_reco" onclick="openTab('tab_reco')">📊 Stock Details</div>
      <div class="tab" data-tab="tab_charts" onclick="openTab('tab_charts')">📁 Reports & Charts</div>
      <div class="tab" data-tab="tab_chat" onclick="openTab('tab_chat')">🤖 Ollama Chat</div>
    </div>

    <div class="card">
      <div class="small">Run Logs</div>
      <div id="logs" class="logs">Ready.</div>
    </div>

    <!-- TAB 1 -->
    <section id="tab_reco" class="tabpane">
      <div class="card">
        <div class="row">
          <div class="col" style="max-width:520px;">
            <div class="small">Search stock (Base or Ticker)</div>
            <input id="search_base" placeholder="e.g., COMB / HNB / LPL ..." />
            <button class="btn secondary" style="margin-top:10px;" onclick="applyRecoFilters()">Apply Filters</button>
            <div class="small" style="margin-top:8px;">Tip: clear the box and click Apply to reset.</div>
          </div>
        </div>
      </div>

      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap;">
          <div>
            <b>MASTER_summary.csv</b>
            <div class="small">Filtered view.</div>
          </div>
          <div class="small" id="master_count"></div>
        </div>
        <div id="tbl_master" class="table-wrap"></div>
      </div>

      <div class="card">
        <b>recommendations_table.csv</b>
        <div id="reco_count" class="small" style="margin:6px 0;"></div>
        <div id="tbl_reco" class="table-wrap"></div>
      </div>
    </section>


    <!-- TAB 3 -->
    <section id="tab_charts" class="tabpane hidden">

      <div class="card">
        <div class="small">Filter charts by stock (Base)</div>
        <input id="charts_search" placeholder="e.g., COMB / HNB / HNBF / LPL ..." />

        <div style="display:flex; gap:10px; margin-top:10px;">
          <button class="btn secondary" onclick="applyChartsFilter()">Apply</button>
          <button class="btn secondary" onclick="showAllCharts()">Show All</button>
        </div>

        <div class="small" id="charts_filter_hint" style="margin-top:8px;"></div>
      </div>

      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <b style="cursor:pointer;" onclick="toggleChartBlock('dcf')">DCF Charts (dcf_outputs) ▾</b>
          <div class="small" id="dcf_hint"></div>
        </div>

        <div id="dcf_picker" class="hidden" style="margin-top:10px;">
          <div class="small">Select DCF chart</div>
          <div id="dcf_buttons" class="chart-buttons"></div>
          <div id="dcf_preview" style="margin-top:10px;"></div>
          <div class="hr"></div>
        </div>

        <div id="imgs_dcf" class="grid-img"></div>
      </div>

      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <b style="cursor:pointer;" onclick="toggleChartBlock('tech')">Technical Charts/Reports (indicators_outputs) ▾</b>
          <div class="small" id="tech_hint"></div>
        </div>

        <div id="tech_picker" class="hidden" style="margin-top:10px;">
          <div class="small">Select Technical chart</div>
          <div id="tech_buttons" class="chart-buttons"></div>
          <div id="tech_preview" style="margin-top:10px;"></div>
          <div class="hr"></div>
        </div>

        <div id="imgs_tech" class="grid-img"></div>
      </div>

      <div class="card">
        <b>N-BEATS Forecast Plots (plots)</b>
        <div id="imgs_nb" class="grid-img"></div>
      </div>

    </section>

    <!-- TAB 4 -->
    <section id="tab_chat" class="tabpane hidden">
      <div class="card">
        <b>🤖 Ollama Chat (Local)</b>
        <div class="small">Chat keeps memory for this browser session.</div>
        <div id="chatbox" class="chatbox"></div>

        <div class="chatrow">
          <input id="chat_input" placeholder="Ask about recommendations, risks, signals, or what the tables mean." />
          <button onclick="sendChat()">Send</button>
          <button style="background:#111827;" onclick="resetChat()">Reset</button>
        </div>
      </div>
    </section>

  </main>
</div>

<script>
let CACHE = {
  master_all:null, reco_all:null,
  master:null, reco:null,
  images:{dcf:[], tech:[], nbeats:[]},
  bases:[]
};

function openTab(id) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`.tab[data-tab="${id}"]`).classList.add('active');
  document.querySelectorAll('.tabpane').forEach(p => p.classList.add('hidden'));
  document.getElementById(id).classList.remove('hidden');
}

function setLogs(text) {
  const el = document.getElementById('logs');
  if (el) el.textContent = text || '';
}

function escapeHtml(s) {
  return String(s)
    .replaceAll('&','&amp;').replaceAll('<','&lt;')
    .replaceAll('>','&gt;').replaceAll('"','&quot;')
    .replaceAll("'",'&#039;');
}

function renderTable(containerId, tableObj) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = '';
  if (!tableObj || !tableObj.exists) {
    el.innerHTML = '<div class="small">No file found.</div>';
    return;
  }
  const cols = tableObj.columns || [];
  const rows = tableObj.rows || [];
  if (cols.length === 0) {
    el.innerHTML = '<div class="small">No columns to display.</div>';
    return;
  }
  const table = document.createElement('table');
  const thead = document.createElement('thead');
  const trh = document.createElement('tr');
  cols.forEach(c => { const th = document.createElement('th'); th.textContent = c; trh.appendChild(th); });
  thead.appendChild(trh);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  rows.forEach(r => {
    const tr = document.createElement('tr');
    cols.forEach((_, i) => {
      const td = document.createElement('td');
      td.innerHTML = escapeHtml(r[i] ?? '');
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  el.appendChild(table);
}

function fillDropdown(selectId, items) {
  const sel = document.getElementById(selectId);
  if (!sel) return;
  sel.innerHTML = '';
  (items || []).forEach(b => {
    const opt = document.createElement('option');
    opt.value = b;
    opt.textContent = b;
    sel.appendChild(opt);
  });
}

function renderImages(containerId, group, files) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = '';
  if (!files || files.length === 0) {
    el.innerHTML = '<div class="small">No images found.</div>';
    return;
  }
  files.slice(0, 24).forEach(name => {
    const img = document.createElement('img');
    img.src = '/images/' + group + '/' + encodeURIComponent(name);
    img.alt = name;
    el.appendChild(img);
  });
}

/* ---------- chart pickers (BUTTONS) ---------- */
function toggleChartBlock(group) {
  const picker = document.getElementById(group + "_picker");
  if (!picker) return;
  picker.classList.toggle("hidden");
}

function makePrettyLabel(filename) {
  const f = filename.toLowerCase();
  if (f.includes("_eps_quarterly")) return "Quarterly EPS (Rs.)";
  if (f.includes("_eps_ttm_rolling")) return "Rolling TTM EPS (4Q sum)";
  if (f.includes("_navps_quarterly")) return "Quarterly NAVPS (Rs.)";
  if (f.includes("dcf_fair_value")) return "DCF Fair Value";
  if (f.includes("eps_trend")) return "EPS Trend";
  if (f.includes("cash")) return "Cash / Cash Equivalents";
  if (f.includes("fcf")) return "Free Cash Flow";
  if (f.includes("rsi")) return "RSI";
  if (f.includes("macd")) return "MACD";
  if (f.includes("boll")) return "Bollinger Bands";
  if (f.includes("ema")) return "EMA";
  if (f.includes("sma")) return "SMA";
  return filename;
}

function renderSingleImage(containerId, group, filename) {
  const el = document.getElementById(containerId);
  if (!el) return;
  if (!filename) {
    el.innerHTML = '<div class="small">No chart selected.</div>';
    return;
  }
  el.innerHTML = `
    <div style="border:1px solid var(--border); border-radius:14px; padding:10px; background:white;">
      <img src="/images/${group}/${encodeURIComponent(filename)}"
           style="width:100%; max-height:520px; object-fit:contain; border-radius:12px;" />
      <div class="small" style="margin-top:8px;">${escapeHtml(makePrettyLabel(filename))}</div>
    </div>
  `;
}

function clearActiveChips(containerId){
  const box = document.getElementById(containerId);
  if (!box) return;
  box.querySelectorAll(".chip").forEach(b => b.classList.remove("active"));
}

function renderChartButtons(group, files) {
  const containerId = (group === "dcf") ? "dcf_buttons" : "tech_buttons";
  const previewId   = (group === "dcf") ? "dcf_preview"  : "tech_preview";

  const box = document.getElementById(containerId);
  if (!box) return;

  box.innerHTML = "";

  if (!files || files.length === 0) {
    box.innerHTML = '<div class="small">No charts found.</div>';
    document.getElementById(previewId).innerHTML = "";
    return;
  }

  files.forEach((fname, idx) => {
    const btn = document.createElement("button");
    btn.className = "chip";
    btn.type = "button";
    btn.textContent = makePrettyLabel(fname);

    btn.onclick = () => {
      clearActiveChips(containerId);
      btn.classList.add("active");
      renderSingleImage(previewId, group, fname);
    };

    box.appendChild(btn);

    if (idx === 0) setTimeout(() => btn.click(), 0);
  });
}

function buildPickersForBase(base) {
  const key = base.toLowerCase();

  const dcfFiles  = (CACHE.images.dcf  || []).filter(f => f.toLowerCase().includes(key));
  const techFiles = (CACHE.images.tech || []).filter(f => f.toLowerCase().includes(key));

  document.getElementById("dcf_picker").classList.remove("hidden");
  document.getElementById("tech_picker").classList.remove("hidden");

  document.getElementById("dcf_hint").textContent  = dcfFiles.length ? `${dcfFiles.length} charts` : "No charts";
  document.getElementById("tech_hint").textContent = techFiles.length ? `${techFiles.length} charts` : "No charts";

  renderChartButtons("dcf", dcfFiles);
  renderChartButtons("tech", techFiles);
}

function hidePickers() {
  document.getElementById("dcf_picker").classList.add("hidden");
  document.getElementById("tech_picker").classList.add("hidden");
  document.getElementById("dcf_hint").textContent = "";
  document.getElementById("tech_hint").textContent = "";

  const dcfBtn = document.getElementById("dcf_buttons");
  const techBtn = document.getElementById("tech_buttons");
  if (dcfBtn) dcfBtn.innerHTML = "";
  if (techBtn) techBtn.innerHTML = "";

  document.getElementById("dcf_preview").innerHTML = "";
  document.getElementById("tech_preview").innerHTML = "";
}
/* --------------------------------------- */

async function runScript(which) {
  setLogs('Running ' + which + ' ...');
  const r = await fetch('/api/run', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({which}) });
  const j = await r.json();
  setLogs((j.ok ? '✅ ' : '❌ ') + (j.logs || ''));
  await reloadAll();
}

async function reloadAll() {
  setLogs('Loading outputs...');
  const r = await fetch('/api/data');
  const j = await r.json();

  CACHE.master_all = j.master;
  CACHE.reco_all   = j.reco;
  CACHE.master     = j.master;
  CACHE.reco       = j.reco;

  CACHE.images = j.images || {dcf:[], tech:[], nbeats:[]};
  CACHE.bases  = j.bases || [];

  renderTable('tbl_master', CACHE.master);
  renderTable('tbl_reco', CACHE.reco);

  const mc = document.getElementById('master_count');
  if (mc && CACHE.master && CACHE.master.exists) mc.textContent = 'Rows shown: ' + (CACHE.master.rows?.length || 0);

  const rc = document.getElementById('reco_count');
  if (rc && CACHE.reco && CACHE.reco.exists) rc.textContent = 'Rows shown: ' + (CACHE.reco.rows?.length || 0);

  fillDropdown('base_sel', CACHE.bases);

  renderImages('imgs_dcf', 'dcf', CACHE.images.dcf || []);
  renderImages('imgs_tech', 'tech', CACHE.images.tech || []);
  renderImages('imgs_nb', 'nbeats', CACHE.images.nbeats || []);

  setLogs(' Loaded.');
}

function applyChartsFilter() {
  const base = (document.getElementById('charts_search').value || '').trim();
  if (!base) { showAllCharts(); return; }
  const key = base.toLowerCase();

  renderImages('imgs_dcf', 'dcf', (CACHE.images.dcf || []).filter(f => f.toLowerCase().includes(key)));
  renderImages('imgs_tech', 'tech', (CACHE.images.tech || []).filter(f => f.toLowerCase().includes(key)));
  renderImages('imgs_nb', 'nbeats', (CACHE.images.nbeats || []).filter(f => f.toLowerCase().includes(key)));

  buildPickersForBase(base);
  document.getElementById('charts_filter_hint').textContent = 'Showing charts for: ' + base.toUpperCase();
}

function showAllCharts() {
  renderImages('imgs_dcf', 'dcf', CACHE.images.dcf || []);
  renderImages('imgs_tech', 'tech', CACHE.images.tech || []);
  renderImages('imgs_nb', 'nbeats', CACHE.images.nbeats || []);
  hidePickers();
  const hint = document.getElementById('charts_filter_hint');
  if (hint) hint.textContent = '';
}

function applyRecoFilters() {
  const search = (document.getElementById('search_base')?.value || '').trim().toLowerCase();
  if (!CACHE.master_all || !CACHE.reco_all) return;

  if (!search) {
    CACHE.master = CACHE.master_all;
    CACHE.reco   = CACHE.reco_all;
    renderTable('tbl_master', CACHE.master);
    renderTable('tbl_reco', CACHE.reco);
    document.getElementById('master_count').textContent = 'Rows shown: ' + (CACHE.master.rows?.length || 0);
    document.getElementById('reco_count').textContent   = 'Rows shown: ' + (CACHE.reco.rows?.length || 0);
    return;
  }

  const mCols = CACHE.master_all.columns || [];
  const mRows = CACHE.master_all.rows || [];
  const mBase = mCols.indexOf('Base');
  const mTick = mCols.indexOf('Ticker');

  const masterFiltered = mRows.filter(r => {
    const b = (mBase >= 0) ? String(r[mBase] || '').toLowerCase() : '';
    const t = (mTick >= 0) ? String(r[mTick] || '').toLowerCase() : '';
    return b.includes(search) || t.includes(search);
  });

  CACHE.master = {exists:true, columns:mCols, rows:masterFiltered};
  renderTable('tbl_master', CACHE.master);
  document.getElementById('master_count').textContent = 'Rows shown: ' + masterFiltered.length;

  const rCols = CACHE.reco_all.columns || [];
  const rRows = CACHE.reco_all.rows || [];
  const rBase = rCols.indexOf('Base');
  const rTick = rCols.indexOf('Ticker');

  const recoFiltered = rRows.filter(r => {
    const b = (rBase >= 0) ? String(r[rBase] || '').toLowerCase() : '';
    const t = (rTick >= 0) ? String(r[rTick] || '').toLowerCase() : '';
    return b.includes(search) || t.includes(search);
  });

  CACHE.reco = {exists:true, columns:rCols, rows:recoFiltered};
  renderTable('tbl_reco', CACHE.reco);
  document.getElementById('reco_count').textContent = 'Rows shown: ' + recoFiltered.length;
}

async function loadDetails() {
  const base = document.getElementById('base_sel').value;
  if (!base) return;
  setLogs('Loading details for ' + base + ' ...');
  const r = await fetch('/api/details?base=' + encodeURIComponent(base));
  const j = await r.json();
  renderTable('d_master', j.master);
  renderTable('d_reco', j.reco);
  renderTable('d_dcf', j.dcf);
  renderTable('d_metrics', j.metrics);
  setLogs(' Details loaded.');
}

function addChat(role, text) {
  const box = document.getElementById('chatbox');
  const div = document.createElement('div');
  div.className = 'msg ' + (role === 'user' ? 'user' : 'bot');
  div.textContent = text;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

async function sendChat() {
  const inp = document.getElementById('chat_input');
  const msg = (inp.value || '').trim();
  if (!msg) return;
  addChat('user', msg);
  inp.value = '';

  const model = (document.getElementById('ollama_model').value || 'llama3').trim();
  const temp = parseFloat(document.getElementById('ollama_temp').value || '0.4');
  const use_context = document.getElementById('use_context').checked;

  const r = await fetch('/api/chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message: msg, model, temperature: temp, use_context}) });
  const j = await r.json();
  addChat('bot', j.reply || '(no reply)');
}

async function resetChat() {
  await fetch('/api/chat_reset', { method:'POST' });
  document.getElementById('chatbox').innerHTML = '';
  setLogs('Chat reset.');
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && document.activeElement && document.activeElement.id === 'chat_input') sendChat();
  if (e.key === 'Enter' && document.activeElement && document.activeElement.id === 'search_base') applyRecoFilters();
  if (e.key === 'Enter' && document.activeElement && document.activeElement.id === 'charts_search') applyChartsFilter();
});

reloadAll();
</script>
</body>
</html>
"""
    resp = make_response(html)
    resp.set_cookie("sid", sid, max_age=SESSION_TTL_SECONDS, httponly=False, samesite="Lax")
    return resp

@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json(force=True) or {}
    which = (data.get("which") or "").strip().lower()
    mapping = {"nbeats": SCRIPT_NBEATS, "dcf": SCRIPT_DCF, "tech": SCRIPT_TECH, "all": SCRIPT_ALL, "system": SCRIPT_SYSTEM}
    script = mapping.get(which)
    if not script:
        return jsonify({"ok": False, "logs": f"Unknown 'which': {which}"}), 400
    if which == "system" and not script.exists():
        return jsonify({"ok": False, "logs": "system_runner.py not found (optional)."}), 200
    return jsonify(run_script(script))

@app.route("/api/data", methods=["GET"])
def api_data():
    df_master = ensure_base(read_csv_if_exists(OUT_MASTER))
    df_dcf    = ensure_base(read_csv_if_exists(OUT_DCF))
    df_master = enrich_master_with_dcf_financials(df_master, df_dcf)
    df_master = force_ticker_dot_n(df_master)
    df_master = drop_cols(df_master, MASTER_HIDE_COLS)

    df_metrics = ensure_base(read_csv_if_exists(OUT_METRICS))
    df_reco    = ensure_base(read_csv_if_exists(OUT_RECO))
    df_best    = ensure_base(read_csv_if_exists(OUT_BEST))
    df_reco    = enrich_reco_with_best_stats(df_reco, df_best)
    df_reco    = drop_cols(df_reco, RECO_HIDE_COLS)

    bases = set()
    for df in [df_master, df_reco, df_metrics]:
        if df is not None and not df.empty and "Base" in df.columns:
            bases.update(df["Base"].dropna().astype(str).str.strip().tolist())
    bases = sorted([b for b in bases if b])

    return jsonify({
        "master":  df_to_json_table(df_master),
        "reco":    df_to_json_table(df_reco),
        "dcf":     df_to_json_table(df_dcf),
        "metrics": df_to_json_table(df_metrics),
        "bases":   bases,
        "images": {
            "dcf": list_images(FOLDER_DCF),
            "tech": list_images(FOLDER_TECH),
            "nbeats": list_images(FOLDER_NBEATS_PLOT),
        }
    })

@app.route("/api/details", methods=["GET"])
def api_details():
    base = (request.args.get("base") or "").strip()

    df_master  = ensure_base(read_csv_if_exists(OUT_MASTER))
    df_dcf     = ensure_base(read_csv_if_exists(OUT_DCF))
    df_metrics = ensure_base(read_csv_if_exists(OUT_METRICS))
    df_reco    = ensure_base(read_csv_if_exists(OUT_RECO))

    df_master = enrich_master_with_dcf_financials(df_master, df_dcf)
    df_master = force_ticker_dot_n(df_master)
    df_master = drop_cols(df_master, MASTER_HIDE_COLS)
    df_reco   = drop_cols(df_reco, RECO_HIDE_COLS)

    def filt(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return df
        df = ensure_base(df)
        if "Base" in df.columns:
            return df[df["Base"].astype(str).str.strip() == base]
        return df

    return jsonify({
        "master":  df_to_json_table(filt(df_master), 200),
        "reco":    df_to_json_table(filt(df_reco), 200),
        "dcf":     df_to_json_table(filt(df_dcf), 200),
        "metrics": df_to_json_table(filt(df_metrics), 200),
    })

@app.route("/images/<group>/<filename>")
def serve_image(group, filename):
    group = group.lower()
    if group == "dcf":
        folder = FOLDER_DCF
    elif group == "tech":
        folder = FOLDER_TECH
    elif group == "nbeats":
        folder = FOLDER_NBEATS_PLOT
    else:
        return "Invalid group", 404

    path = folder / filename
    if not path.exists():
        return "Not found", 404

    data = path.read_bytes()
    ext = path.suffix.lower()
    mime = "image/png" if ext == ".png" else ("image/webp" if ext == ".webp" else "image/jpeg")

    resp = make_response(data)
    resp.headers["Content-Type"] = mime
    return resp

@app.route("/api/chat_reset", methods=["POST"])
def api_chat_reset():
    sid = get_session_id()
    SESSIONS[sid]["chat_messages"] = [{"role": "system", "content": SYSTEM_PROMPT}]
    SESSIONS[sid]["last_context_base"] = None
    return jsonify({"ok": True})

@app.route("/api/chat", methods=["POST"])
def api_chat():
    sid = get_session_id()
    data = request.get_json(force=True) or {}

    user_text = (data.get("message") or "").strip()
    model = (data.get("model") or "llama3.2:3b").strip()
    temp = float(data.get("temperature") or 0.4)
    use_context = bool(data.get("use_context", True))

    if not user_text:
        return jsonify({"ok": False, "reply": "Empty message"}), 400

    # Load all outputs
    df_master  = ensure_base(read_csv_if_exists(OUT_MASTER))
    df_master  = drop_cols(df_master, MASTER_HIDE_COLS)

    df_reco    = ensure_base(read_csv_if_exists(OUT_RECO))
    df_reco    = drop_cols(df_reco, RECO_HIDE_COLS)

    df_dcf     = ensure_base(read_csv_if_exists(OUT_DCF))
    df_metrics = ensure_base(read_csv_if_exists(OUT_METRICS))

    # Build known bases list (from all tables)
    known_bases: List[str] = []
    for df in [df_master, df_reco, df_dcf, df_metrics]:
        if df is not None and not df.empty and "Base" in df.columns:
            known_bases += df["Base"].dropna().astype(str).str.strip().tolist()
    known_bases = sorted(list(set([b for b in known_bases if b])))

    # Detect whether user is asking about a specific stock base
    detected_base = detect_base_from_text(user_text, known_bases)

    # Inject context ONLY if checkbox enabled AND a known base is mentioned
    #  also prevent re-injecting same stock block repeatedly
    if use_context and detected_base and SESSIONS[sid].get("last_context_base") != detected_base:
        ctx = build_context_text_for_base(detected_base, df_master, df_reco, df_dcf, df_metrics)
        if ctx:
            SESSIONS[sid]["chat_messages"].append({"role": "system", "content": STOCK_REPORT_INSTRUCTIONS})
            SESSIONS[sid]["chat_messages"].append({
                "role": "system",
                "content": (
                    "Use the following dashboard CSV data as ground truth for this stock. "
                    "Do not invent numbers that are not shown.\n\n" + ctx
                )
            })
        SESSIONS[sid]["last_context_base"] = detected_base

    # Add user message
    SESSIONS[sid]["chat_messages"].append({"role": "user", "content": user_text})

    #  keep chat memory from becoming too big
    SESSIONS[sid]["chat_messages"] = trim_chat(SESSIONS[sid]["chat_messages"], keep_last=28)

    try:
        reply = ollama_chat(model, SESSIONS[sid]["chat_messages"], temperature=temp)
        SESSIONS[sid]["chat_messages"].append({"role": "assistant", "content": reply})
        SESSIONS[sid]["chat_messages"] = trim_chat(SESSIONS[sid]["chat_messages"], keep_last=28)
        return jsonify({"ok": True, "reply": reply})
    except Exception as e:
        return jsonify({
            "ok": False,
            "reply": f"Ollama call failed: {e}\nCheck: Ollama is running, model name is correct, and URL is reachable."
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
