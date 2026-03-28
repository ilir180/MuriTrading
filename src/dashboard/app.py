"""
MuriTrading – Phase 4: Live Dashboard
Streamlit App: Lädt Modelle, holt aktuelle XRP-Daten, berechnet Features, zeigt Predictions.
Start: streamlit run src/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
import requests as _requests
import time as _time
from datetime import datetime, timezone

try:
    import ccxt
except ImportError:
    ccxt = None

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

# ── Projekt-Root für Imports ───────────────────────────────────
# Lokal: ~/MuriTrading, Streamlit Cloud: /mount/src/muritrading
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from src.features.build_features import add_indicators

# ── Pfade ──────────────────────────────────────────────────────
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# ── Konfiguration ─────────────────────────────────────────────
SYMBOL = "XRP/USDT"
TIMEFRAMES = {"15m": 200, "1h": 250, "4h": 250, "1d": 250}
CONFIDENCE_THRESH = 0.65


# ═══════════════════════════════════════════════════════════════
#  SVG LOGO
# ═══════════════════════════════════════════════════════════════

LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 580 140" height="70" style="max-width:360px">
  <defs>
    <linearGradient id="t_rg" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#7c3aed"/><stop offset="55%" stop-color="#4f46e5"/><stop offset="100%" stop-color="#10b981"/></linearGradient>
    <linearGradient id="t_ag" x1="0%" y1="100%" x2="100%" y2="0%"><stop offset="0%" stop-color="#4f46e5"/><stop offset="100%" stop-color="#34d399"/></linearGradient>
    <linearGradient id="t_bg" x1="20%" y1="10%" x2="80%" y2="90%"><stop offset="0%" stop-color="#1e2442"/><stop offset="100%" stop-color="#111827"/></linearGradient>
    <linearGradient id="t_tb" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#818cf8"/><stop offset="100%" stop-color="#34d399"/></linearGradient>
    <filter id="t_sg"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    <clipPath id="t_bc"><circle cx="70" cy="70" r="44"/></clipPath>
  </defs>
  <circle cx="70" cy="70" r="60" fill="none" stroke="url(#t_rg)" stroke-width="1.5" opacity="0.25"/>
  <circle cx="70" cy="70" r="52" fill="none" stroke="url(#t_rg)" stroke-width="2.5" opacity="0.5" filter="url(#t_sg)"/>
  <circle cx="70" cy="70" r="44" fill="url(#t_bg)" stroke="url(#t_rg)" stroke-width="1.5"/>
  <g clip-path="url(#t_bc)" opacity="0.85">
    <line x1="48" y1="42" x2="48" y2="98" stroke="#7c3aed" stroke-width="1.2" opacity="0.6"/>
    <rect x="43" y="52" width="10" height="26" rx="1.5" fill="#7c3aed" opacity="0.5" stroke="#7c3aed" stroke-width="0.8"/>
    <line x1="64" y1="38" x2="64" y2="95" stroke="#4f46e5" stroke-width="1.2" opacity="0.6"/>
    <rect x="59" y="48" width="10" height="30" rx="1.5" fill="#4f46e5" opacity="0.4" stroke="#4f46e5" stroke-width="0.8"/>
    <line x1="80" y1="34" x2="80" y2="88" stroke="#10b981" stroke-width="1.2" opacity="0.6"/>
    <rect x="75" y="40" width="10" height="28" rx="1.5" fill="#10b981" opacity="0.45" stroke="#10b981" stroke-width="0.8"/>
    <line x1="96" y1="30" x2="96" y2="82" stroke="#34d399" stroke-width="1.2" opacity="0.6"/>
    <rect x="91" y="36" width="10" height="24" rx="1.5" fill="#34d399" opacity="0.4" stroke="#34d399" stroke-width="0.8"/>
    <line x1="40" y1="85" x2="102" y2="38" stroke="url(#t_ag)" stroke-width="2" stroke-linecap="round" opacity="0.7"/>
  </g>
  <line x1="96" y1="68" x2="116" y2="28" stroke="url(#t_ag)" stroke-width="4" stroke-linecap="round"/>
  <polygon points="116,28 104,32 110,44" fill="url(#t_ag)"/>
  <circle cx="107" cy="48" r="5" fill="#0f0f1a"/>
  <text x="150" y="82" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif" font-size="50" font-weight="800" letter-spacing="-1" fill="#e2e8f0">Murati</text>
  <text x="348" y="82" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif" font-size="50" font-weight="800" letter-spacing="-1" fill="url(#t_tb)">Trading</text>
  <text x="151" y="103" font-family="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif" font-size="12.5" font-weight="500" letter-spacing="3.5" fill="#475569">XRP &#183; PREDICTION SYSTEM</text>
  <line x1="151" y1="110" x2="540" y2="110" stroke="url(#t_rg)" stroke-width="1.5" opacity="0.35"/>
</svg>
"""


# ═══════════════════════════════════════════════════════════════
#  SHARED CSS (embedded in every st.html iframe)
# ═══════════════════════════════════════════════════════════════

SHARED_CSS = """
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    background:transparent;color:#e2e8f0}

  .hero-section {
    background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 60%,#0f2744 100%);
    padding:40px 24px 32px;text-align:center;
    border-bottom:1px solid #1e2942;border-radius:0 0 20px 20px;
  }
  .hero-subtitle{color:#64748b;font-size:0.9rem;margin-top:12px}
  .kpi-row{display:flex;gap:14px;justify-content:center;flex-wrap:wrap;margin-top:28px}
  .kpi-card{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);
    border-radius:14px;padding:18px 26px;min-width:150px;text-align:center;
    backdrop-filter:blur(10px);transition:border-color 0.3s,transform 0.2s}
  .kpi-card:hover{border-color:rgba(255,255,255,0.2);transform:translateY(-2px)}
  .kpi-value{font-size:2rem;font-weight:800;line-height:1}
  .kpi-label{font-size:0.7rem;color:#64748b;margin-top:6px;text-transform:uppercase;letter-spacing:0.05em}
  .kpi-delta{font-size:0.75rem;margin-top:4px;font-weight:600}
  .c-blue{color:#38bdf8}.c-green{color:#34d399}.c-purple{color:#c084fc}
  .c-red{color:#f87171}.c-yellow{color:#fbbf24}.c-white{color:#e2e8f0}

  .signal-badge{display:inline-block;font-weight:700;font-size:1rem;padding:6px 18px;border-radius:8px;letter-spacing:0.05em}
  .signal-long{background:rgba(16,185,129,0.15);color:#34d399;border:1px solid rgba(52,211,153,0.4)}
  .signal-short{background:rgba(248,113,113,0.15);color:#f87171;border:1px solid rgba(248,113,113,0.4)}
  .signal-neutral{background:rgba(251,191,36,0.12);color:#fbbf24;border:1px solid rgba(251,191,36,0.35)}

  .notice-box{background:rgba(79,70,229,0.12);border:1px solid rgba(79,70,229,0.35);
    border-radius:12px;padding:14px 20px;margin:20px 0;font-size:0.82rem;color:#a5b4fc;line-height:1.6}
  .notice-box strong{color:#e2e8f0}

  .section-header{font-size:0.75rem;font-weight:700;letter-spacing:0.15em;color:#4f46e5;
    text-transform:uppercase;margin-bottom:16px;margin-top:8px}

  .pred-row{display:flex;gap:14px;justify-content:center;flex-wrap:wrap}
  .pred-card{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
    border-radius:14px;padding:20px 28px;min-width:180px;text-align:center;flex:1}
  .pred-card.highlight{border-color:rgba(79,70,229,0.4);background:rgba(79,70,229,0.08)}
  .pred-price{font-size:1.6rem;font-weight:800;color:#e2e8f0;line-height:1}
  .pred-delta{font-size:0.85rem;font-weight:600;margin-top:4px}
  .pred-label{font-size:0.7rem;color:#64748b;margin-top:6px;text-transform:uppercase;letter-spacing:0.05em}

  .mtf-table{width:100%;border-collapse:collapse;background:#1a1f35;
    border-radius:14px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.3)}
  .mtf-table thead th{padding:12px 14px;text-align:left;font-size:0.72rem;font-weight:600;
    color:#475569;text-transform:uppercase;letter-spacing:0.05em;
    border-bottom:2px solid #2d3748;background:#151827}
  .mtf-table tbody td{padding:13px 14px;border-bottom:1px solid #1e2942;font-size:0.87rem;color:#e2e8f0}
  .mtf-table tbody tr:hover td{background:#1e2942}
  .mtf-table tbody tr:last-child td{border-bottom:none}
  .pill-bull{background:rgba(16,185,129,0.15);color:#34d399;border:1px solid rgba(52,211,153,0.35);
    font-weight:700;font-size:0.78rem;padding:3px 10px;border-radius:6px}
  .pill-bear{background:rgba(248,113,113,0.15);color:#f87171;border:1px solid rgba(248,113,113,0.35);
    font-weight:700;font-size:0.78rem;padding:3px 10px;border-radius:6px}

  .confluence-bar-bg{background:rgba(255,255,255,0.06);border-radius:8px;height:12px;width:100%;margin-top:10px;overflow:hidden}
  .confluence-bar-fill{height:100%;border-radius:8px;transition:width 0.5s ease}

  .muri-footer{text-align:center;padding:24px;color:#1e2942;font-size:0.75rem;margin-top:40px}
</style>
"""

# CSS that targets the parent Streamlit page (not iframes)
STREAMLIT_OVERRIDES = """
<style>
  .stApp { background: #0f0f1a !important; }
  header[data-testid="stHeader"] { background: transparent !important; }
  #MainMenu, footer, .stDeployButton { display: none !important; }
  .stExpander { border-color: #1e2942 !important; }
  div[data-testid="stExpander"] details {
    background: #1a1f35 !important;
    border: 1px solid #1e2942 !important;
    border-radius: 14px !important;
  }
  div[data-testid="stExpander"] summary {
    color: #a5b4fc !important;
    font-weight: 600 !important;
  }
  .stPlotlyChart { border-radius: 14px; overflow: hidden; }
  hr { border-color: #1e2942 !important; opacity: 0.5 !important; }
  /* Reduce gaps between iframe blocks */
  .element-container:has(iframe) { margin-bottom: -1rem !important; }
  iframe { background: transparent !important; border: none !important; }

  /* Dark mode for dataframes and JSON */
  div[data-testid="stDataFrame"] table { background: #1a1f35 !important; color: #e2e8f0 !important; }
  div[data-testid="stDataFrame"] th { background: #151827 !important; color: #475569 !important; }
  div[data-testid="stDataFrame"] td { color: #e2e8f0 !important; border-color: #1e2942 !important; }
  div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }
  div[data-testid="stJson"] { background: #1a1f35 !important; border-radius: 14px; padding: 16px !important; }
  pre { background: #1a1f35 !important; color: #e2e8f0 !important; }

  /* Streamlit dataframe internal overrides */
  .stDataFrame [data-testid="glideDataEditor"] { background: #1a1f35 !important; }
  div[data-baseweb="table"] { background: #1a1f35 !important; }

  /* Force dark on all internal containers */
  .stApp > div { color: #e2e8f0; }
  .stApp section[data-testid="stSidebar"] { background: #0f172a !important; }

  /* Expander content area dark */
  div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
    background: #1a1f35 !important;
  }
</style>
"""


def render_html(html_body, height=200):
    """Rendert HTML als self-contained Block mit eingebetteten Styles."""
    full = f"<html><head>{SHARED_CSS}</head><body>{html_body}</body></html>"
    components.html(full, height=height, scrolling=False)


# ═══════════════════════════════════════════════════════════════
#  DATEN & MODELL
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "rb") as f:
        rf = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "rb") as f:
        xgb = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    return rf, xgb, meta


@st.cache_data(ttl=30)
def fetch_live_candles(timeframe, limit):
    """Holt Kerzen - versucht ccxt, Fallback auf Binance REST API."""
    candles = None
    # Versuch 1: ccxt
    if ccxt is not None:
        try:
            exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
            candles = exchange.fetch_ohlcv(SYMBOL, timeframe=timeframe, limit=limit)
        except Exception:
            pass
    # Fallback: direkte Binance REST API (mehrere Endpunkte)
    if candles is None:
        binance_urls = [
            "https://data-api.binance.vision/api/v3/klines",
            "https://api.binance.com/api/v3/klines",
            "https://api.binance.us/api/v3/klines",
        ]
        for url in binance_urls:
            try:
                resp = _requests.get(url, params={"symbol": "XRPUSDT", "interval": timeframe, "limit": limit}, timeout=15)
                resp.raise_for_status()
                candles = [[c[0], float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in resp.json()]
                break
            except Exception:
                continue
        if candles is None:
            raise Exception("Alle Binance-Endpunkte fehlgeschlagen")
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns=["timestamp"])
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def build_live_features(feature_cols):
    df_15m = fetch_live_candles("15m", TIMEFRAMES["15m"])
    df_1h  = fetch_live_candles("1h",  TIMEFRAMES["1h"])
    df_4h  = fetch_live_candles("4h",  TIMEFRAMES["4h"])
    df_1d  = fetch_live_candles("1d",  TIMEFRAMES["1d"])

    df_15m = add_indicators(df_15m, prefix="15m_")
    df_1h  = add_indicators(df_1h,  prefix="1h_")
    df_4h  = add_indicators(df_4h,  prefix="4h_")
    df_1d  = add_indicators(df_1d,  prefix="1d_")

    df_15m_sel = df_15m[[c for c in df_15m.columns if c.startswith("15m_")]].copy()
    df_base = pd.merge_asof(
        df_1h.sort_index(), df_15m_sel.sort_index(),
        left_index=True, right_index=True, direction="backward",
    )
    df_4h_sel = df_4h[[c for c in df_4h.columns if c.startswith("4h_")]].copy()
    df_base = pd.merge_asof(
        df_base.sort_index(), df_4h_sel.sort_index(),
        left_index=True, right_index=True, direction="backward",
    )
    df_1d_sel = df_1d[[c for c in df_1d.columns if c.startswith("1d_")]].copy()
    df_base = pd.merge_asof(
        df_base.sort_index(), df_1d_sel.sort_index(),
        left_index=True, right_index=True, direction="backward",
    )

    bull_signals = [
        "1h_ema_9_above_21", "1h_ema_21_above_50", "1h_macd_above",
        "4h_ema_9_above_21", "4h_ema_21_above_50", "4h_macd_above",
        "1d_ema_9_above_21", "1d_ema_21_above_50", "1d_macd_above",
    ]
    bear_signals = ["1h_rsi_overbought", "4h_rsi_overbought", "1d_rsi_overbought"]
    avail_bull = [s for s in bull_signals if s in df_base.columns]
    avail_bear = [s for s in bear_signals if s in df_base.columns]
    if avail_bull:
        df_base["confluence_bull"] = df_base[avail_bull].sum(axis=1)
    if avail_bear:
        df_base["confluence_bear"] = df_base[avail_bear].sum(axis=1)
    df_base["confluence_net"] = df_base.get("confluence_bull", 0) - df_base.get("confluence_bear", 0)

    latest = df_base[feature_cols].dropna()
    if latest.empty:
        return None, df_1h, df_base
    return latest.iloc[[-1]], df_1h, df_base


def predict_ensemble(rf, xgb, X):
    rf_prob  = rf.predict_proba(X)[:, 1]
    xgb_prob = xgb.predict_proba(X)[:, 1]
    ensemble_prob = (rf_prob + xgb_prob) / 2.0
    return float(ensemble_prob[0]), float(rf_prob[0]), float(xgb_prob[0])


# ═══════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════

CHART_BG = "#0f0f1a"
CHART_GRID = "#1e2942"

def create_price_chart(df_1h):
    df = df_1h.tail(72)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.75, 0.25],
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="XRP/USDT",
        increasing_line_color="#34d399", decreasing_line_color="#7c3aed",
        increasing_fillcolor="#34d399", decreasing_fillcolor="#7c3aed",
    ), row=1, col=1)
    if "1h_ema_9" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["1h_ema_9"], name="EMA 9",
            line=dict(color="#818cf8", width=1.2),
        ), row=1, col=1)
    if "1h_ema_21" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["1h_ema_21"], name="EMA 21",
            line=dict(color="#38bdf8", width=1.2),
        ), row=1, col=1)
    colors = ["#34d399" if c >= o else "#7c3aed" for o, c in zip(df["open"], df["close"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], name="Volume",
        marker_color=colors, opacity=0.4,
    ), row=2, col=1)
    fig.update_layout(
        height=480, template="plotly_dark",
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        xaxis_rangeslider_visible=False, showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
    )
    fig.update_xaxes(gridcolor=CHART_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=CHART_GRID, zeroline=False)
    return fig


def create_rsi_chart(df_base):
    df = df_base.tail(72)
    fig = go.Figure()
    rsi_map = {"1h_rsi_14": ("#818cf8", "1H"), "4h_rsi_14": ("#c084fc", "4H"), "1d_rsi_14": ("#38bdf8", "1D")}
    for col, (color, label) in rsi_map.items():
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=f"{label} RSI",
                line=dict(color=color, width=1.5),
            ))
    fig.add_hrect(y0=70, y1=90, fillcolor="#f87171", opacity=0.06, line_width=0)
    fig.add_hrect(y0=10, y1=30, fillcolor="#34d399", opacity=0.06, line_width=0)
    fig.add_hline(y=70, line_dash="dash", line_color="#f87171", opacity=0.4)
    fig.add_hline(y=30, line_dash="dash", line_color="#34d399", opacity=0.4)
    fig.add_hline(y=50, line_dash="dot", line_color="#475569", opacity=0.3)
    fig.update_layout(
        height=250, template="plotly_dark",
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        yaxis=dict(range=[10, 90]),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11, color="#64748b")),
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
    )
    fig.update_xaxes(gridcolor=CHART_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=CHART_GRID, zeroline=False)
    return fig


# ═══════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ═══════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="MuriTrading – XRP Live", page_icon=":chart_with_upwards_trend:", layout="wide")

    # Global Streamlit overrides (works on parent page)
    st.markdown(STREAMLIT_OVERRIDES, unsafe_allow_html=True)

    # ── Modelle laden ──────────────────────────────────────────
    rf, xgb, meta = load_models()
    feature_cols = meta["feature_cols"]

    # ── Live-Daten & Prediction ────────────────────────────────
    with st.spinner(""):
        X_live, df_1h, df_base = build_live_features(feature_cols)

    if X_live is None:
        st.error("Nicht genügend Daten für Prediction.")
        return

    ensemble_prob, rf_prob, xgb_prob = predict_ensemble(rf, xgb, X_live)
    confidence = abs(ensemble_prob - 0.5) * 2
    is_bull = ensemble_prob >= 0.5

    current_price = df_1h["close"].iloc[-1]
    prev_price = df_1h["close"].iloc[-2]
    price_change = (current_price - prev_price) / prev_price

    if confidence < 0.30:
        signal_text, signal_class = "NEUTRAL", "signal-neutral"
    elif is_bull:
        signal_text, signal_class = "LONG", "signal-long"
    else:
        signal_text, signal_class = "SHORT", "signal-short"

    med_return = (ensemble_prob - 0.5) * 0.01
    min_return = med_return - 0.003
    max_return = med_return + 0.003
    price_min = current_price * (1 + min_return)
    price_med = current_price * (1 + med_return)
    price_max = current_price * (1 + max_return)

    latest_row = df_base.iloc[-1]
    confluence_net = latest_row.get("confluence_net", 0)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Vorformatierte Strings
    price_color = "c-green" if price_change >= 0 else "c-red"
    delta_sign = "+" if price_change >= 0 else ""
    conf_color = "c-green" if confidence >= 0.50 else "c-yellow" if confidence >= 0.30 else "c-red"

    v = {
        "price": f"{current_price:.4f}",
        "delta": f"{delta_sign}{price_change:.2%}",
        "ens": f"{ensemble_prob:.0%}",
        "conf": f"{confidence:.0%}",
        "rf": f"{rf_prob:.0%}",
        "xgb": f"{xgb_prob:.0%}",
        "confnet": f"{confluence_net:.0f}",
        "pmin": f"{price_min:.4f}",
        "pmed": f"{price_med:.4f}",
        "pmax": f"{price_max:.4f}",
        "rmin": f"{min_return:+.3%}",
        "rmed": f"{med_return:+.3%}",
        "rmax": f"{max_return:+.3%}",
    }

    min_col = "c-red" if min_return < 0 else "c-green"
    med_col = "c-red" if med_return < 0 else "c-green"
    max_col = "c-green"

    # ══════════════════════════════════════════════════════════
    #  HERO SECTION
    # ══════════════════════════════════════════════════════════
    render_html(f"""
    <div class="hero-section">
      <div style="display:flex;justify-content:center;margin-bottom:16px">
        {LOGO_SVG}
      </div>
      <div class="hero-subtitle">{now_str} UTC &middot; Live Prediction &middot; Auto-Refresh 60s</div>
      <div class="kpi-row">
        <div class="kpi-card">
          <div class="kpi-value {price_color}">${v['price']}</div>
          <div class="kpi-label">XRP/USDT</div>
          <div class="kpi-delta {price_color}">{v['delta']}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-value c-white"><span class="signal-badge {signal_class}">{signal_text}</span></div>
          <div class="kpi-label">Signal</div>
          <div class="kpi-delta" style="color:#64748b">Ensemble {v['ens']} bull</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-value {conf_color}">{v['conf']}</div>
          <div class="kpi-label">Confidence</div>
          <div class="kpi-delta" style="color:#64748b">RF {v['rf']} &middot; XGB {v['xgb']}</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-value c-purple">{v['confnet']}<span style="font-size:1rem;color:#64748b"> / 9</span></div>
          <div class="kpi-label">Confluence</div>
          <div class="kpi-delta" style="color:#64748b">MTF Score</div>
        </div>
      </div>
    </div>
    """, height=430)

    # ══════════════════════════════════════════════════════════
    #  PREDICTION SECTION
    # ══════════════════════════════════════════════════════════
    render_html(f"""
    <div class="section-header">Prediction &middot; N&auml;chste 2 Stunden</div>
    <div class="pred-row">
      <div class="pred-card">
        <div class="pred-label">Worst Case (10. Pz.)</div>
        <div class="pred-price">${v['pmin']}</div>
        <div class="pred-delta {min_col}">{v['rmin']}</div>
      </div>
      <div class="pred-card highlight">
        <div class="pred-label">Erwartet (Median)</div>
        <div class="pred-price">${v['pmed']}</div>
        <div class="pred-delta {med_col}">{v['rmed']}</div>
      </div>
      <div class="pred-card">
        <div class="pred-label">Best Case (90. Pz.)</div>
        <div class="pred-price">${v['pmax']}</div>
        <div class="pred-delta {max_col}">{v['rmax']}</div>
      </div>
    </div>
    """, height=160)

    # ══════════════════════════════════════════════════════════
    #  NOTICE BOX
    # ══════════════════════════════════════════════════════════
    n_train = f"{meta['n_train']:,}"
    render_html(f"""
    <div class="notice-box">
      <strong>Modell-Logik:</strong>
      Ensemble aus Random Forest + XGBoost, trainiert auf {n_train} Kerzen mit 46 MTF-Features.
      Signal wird ausgegeben bei Confidence &gt; 30%. Prediction-Horizont: 2 Stunden (2 Kerzen).
      Train-Ende: <strong>{meta['train_end']}</strong>
    </div>
    """, height=100)

    # ══════════════════════════════════════════════════════════
    #  CHARTS
    # ══════════════════════════════════════════════════════════
    render_html('<div class="section-header">Charts</div>', height=40)

    chart_col1, chart_col2 = st.columns([2, 1])
    with chart_col1:
        st.plotly_chart(create_price_chart(df_base), use_container_width=True)
    with chart_col2:
        st.plotly_chart(create_rsi_chart(df_base), use_container_width=True)

    # ══════════════════════════════════════════════════════════
    #  MTF CONFLUENCE TABLE
    # ══════════════════════════════════════════════════════════
    mtf_rows = ""
    for tf, label in [("1h", "1 Stunde"), ("4h", "4 Stunden"), ("1d", "1 Tag")]:
        rsi = latest_row.get(f"{tf}_rsi_14", None)
        ema_cross = latest_row.get(f"{tf}_ema_9_above_21", None)
        macd = latest_row.get(f"{tf}_macd_above", None)
        bb_pos = latest_row.get(f"{tf}_bb_pos", None)
        stoch = latest_row.get(f"{tf}_stoch_rsi", None)

        rsi_str = f"{rsi:.1f}" if pd.notna(rsi) else "–"
        rsi_color = "c-red" if pd.notna(rsi) and rsi > 70 else "c-green" if pd.notna(rsi) and rsi < 30 else "c-white"
        ema_pill = '<span class="pill-bull">Bullisch</span>' if ema_cross == 1 else '<span class="pill-bear">Bearish</span>' if ema_cross == 0 else "–"
        macd_pill = '<span class="pill-bull">Bullisch</span>' if macd == 1 else '<span class="pill-bear">Bearish</span>' if macd == 0 else "–"
        bb_str = f"{bb_pos:.2f}" if pd.notna(bb_pos) else "–"
        stoch_str = f"{stoch:.2f}" if pd.notna(stoch) else "–"

        mtf_rows += f"""<tr>
          <td style="font-weight:600">{label}</td>
          <td class="{rsi_color}" style="font-family:monospace">{rsi_str}</td>
          <td>{ema_pill}</td><td>{macd_pill}</td>
          <td style="font-family:monospace;color:#c084fc">{bb_str}</td>
          <td style="font-family:monospace;color:#38bdf8">{stoch_str}</td>
        </tr>"""

    conf_bar_pct = int(max(0, min(100, (confluence_net + 3) / 12 * 100)))
    conf_bar_color = "#34d399" if confluence_net > 2 else "#fbbf24" if confluence_net >= 0 else "#f87171"

    render_html(f"""
    <div class="section-header">Multi-Timeframe Confluence</div>
    <table class="mtf-table">
      <thead><tr>
        <th>Timeframe</th><th>RSI (14)</th><th>EMA 9/21</th>
        <th>MACD</th><th style="text-align:center">BB Pos</th><th style="text-align:center">Stoch RSI</th>
      </tr></thead>
      <tbody>{mtf_rows}</tbody>
    </table>
    <div style="margin-top:16px;display:flex;align-items:center;gap:12px">
      <span style="font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.05em;font-weight:600">Confluence</span>
      <div class="confluence-bar-bg" style="flex:1">
        <div class="confluence-bar-fill" style="width:{conf_bar_pct}%;background:{conf_bar_color}"></div>
      </div>
      <span style="font-size:0.9rem;font-weight:700;color:{conf_bar_color}">{v['confnet']} / 9</span>
    </div>
    """, height=280)

    # ══════════════════════════════════════════════════════════
    #  EXPANDABLE DETAILS (als HTML-Tabellen im Dark Style)
    # ══════════════════════════════════════════════════════════
    with st.expander("Feature-Details (Top 20)"):
        feature_values = X_live.iloc[0].to_dict()
        sorted_features = sorted(feature_values.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
        feat_rows = "".join(
            f'<tr><td style="color:#a5b4fc">{name}</td><td style="font-family:monospace;color:#e2e8f0;text-align:right">{val:.4f}</td></tr>'
            for name, val in sorted_features
        )
        render_html(f"""
        <table class="mtf-table" style="font-size:0.85rem">
          <thead><tr><th>Feature</th><th style="text-align:right">Wert</th></tr></thead>
          <tbody>{feat_rows}</tbody>
        </table>
        """, height=620)

    with st.expander("Modell-Info"):
        info_items = [
            ("Trainiert am", meta["trained_at"]),
            ("Features", str(len(feature_cols))),
            ("Train Samples", f"{meta['n_train']:,}"),
            ("Test Samples", f"{meta['n_test']:,}"),
            ("Train Ende", meta["train_end"]),
            ("Test Start", meta["test_start"]),
        ]
        info_rows = "".join(
            f'<tr><td style="color:#64748b;font-weight:600">{k}</td><td style="color:#e2e8f0;font-family:monospace">{val}</td></tr>'
            for k, val in info_items
        )
        render_html(f"""
        <table class="mtf-table" style="font-size:0.85rem">
          <tbody>{info_rows}</tbody>
        </table>
        """, height=230)

    # ── FOOTER ─────────────────────────────────────────────────
    render_html(f'<div class="muri-footer">MuriTrading &middot; {now_str} UTC &middot; XRP Prediction System</div>', height=60)

    # Auto-Refresh
    _time.sleep(60)
    st.rerun()


if __name__ == "__main__":
    main()
