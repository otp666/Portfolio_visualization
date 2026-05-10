import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, datetime
import warnings

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="metric-container"] {
    background: rgba(128,128,128,0.06);
    border: 0.5px solid rgba(128,128,128,0.18);
    border-radius: 10px;
    padding: 14px 16px;
}
[data-testid="stMetricDelta"] svg { display: none; }
div[data-testid="stForm"] { border: none; padding: 0; }
</style>
""", unsafe_allow_html=True)

COLORS = [
    "#534AB7", "#D85A30", "#1D9E75", "#378ADD",
    "#EF9F27", "#D4537E", "#5DCAA5", "#63991F",
    "#BA7517", "#0C447C", "#993C1D", "#3B6D11",
]

PRESETS = {
    "60/40 Classic":   {"SPY": 60, "BND": 40},
    "QQQ / BND 70/30": {"QQQ": 70, "BND": 30},
    "TQQQ / BND 25/75":{"TQQQ": 25, "BND": 75},
    "All-World":       {"VTI": 50, "VEA": 30, "VWO": 20},
    "Gold + Bonds":    {"GLD": 30, "TLT": 40, "BND": 30},
}

# ── Session state ──────────────────────────────────────────────────────────────
if "tickers" not in st.session_state:
    st.session_state.tickers = ["SPY", "QQQ", "BND"]
if "weights" not in st.session_state:
    st.session_state.weights = {"SPY": 60, "QQQ": 30, "BND": 10}

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices, return a tidy DataFrame."""
    try:
        raw = yf.download(
            list(tickers), start=start, end=end,
            auto_adjust=True, progress=False, threads=True,
        )
        if raw.empty:
            return pd.DataFrame()
        # Handle MultiIndex (multiple tickers) vs flat (single ticker)
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw["Close"].copy()
        else:
            ticker = tickers[0]
            df = raw[["Close"]].rename(columns={"Close": ticker})
        return df.dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def validate_ticker(ticker: str) -> tuple[bool, str]:
    try:
        sample = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
        if sample.empty:
            return False, ""
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ticker
        return True, name
    except Exception:
        return False, ""


def norm_weights(tickers, raw_weights):
    total = sum(raw_weights.get(t, 0) for t in tickers)
    if total == 0:
        return {t: 1 / len(tickers) for t in tickers}
    return {t: raw_weights.get(t, 0) / total for t in tickers}


def compute_metrics(rets: pd.Series, rf: float = 0.045, initial: float = 10_000):
    cum = (1 + rets).cumprod()
    years = max(len(rets) / 252, 0.01)

    total_ret  = (cum.iloc[-1] - 1) * 100
    cagr       = ((cum.iloc[-1] ** (1 / years)) - 1) * 100
    rolling_pk = cum.cummax()
    dd         = (cum - rolling_pk) / rolling_pk * 100
    max_dd     = dd.min()
    ann_vol    = rets.std() * np.sqrt(252) * 100
    ann_ret    = rets.mean() * 252 * 100
    sharpe     = (ann_ret / 100 - rf) / (ann_vol / 100) if ann_vol else 0
    calmar     = cagr / abs(max_dd) if max_dd else 0
    best_yr    = (rets.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100).max()
    worst_yr   = (rets.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100).min()

    return dict(
        cagr=cagr, total_ret=total_ret, max_dd=max_dd,
        ann_vol=ann_vol, sharpe=sharpe, calmar=calmar,
        best_yr=best_yr, worst_yr=worst_yr,
        cum=cum * initial, dd=dd,
    )


def annual_returns(rets: pd.Series) -> pd.Series:
    return rets.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100


def delta(port_val, bench_val, bench_name, fmt=".1f", inverse=False):
    if bench_val is None:
        return None, None
    diff = port_val - bench_val
    label = f"{'+'if diff>=0 else ''}{diff:{fmt}} vs {bench_name}"
    color = ("normal" if diff >= 0 else "inverse") if not inverse else ("inverse" if diff >= 0 else "normal")
    return label, color


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Portfolio")

    # ── Presets ────────────────────────────────────────────────────────────────
    with st.expander("Quick presets"):
        for preset_name, preset_w in PRESETS.items():
            if st.button(preset_name, use_container_width=True):
                st.session_state.tickers = list(preset_w.keys())
                st.session_state.weights = dict(preset_w)
                st.rerun()

    st.divider()

    # ── Add ticker ─────────────────────────────────────────────────────────────
    st.subheader("Add ticker")
    with st.form("add_form", clear_on_submit=True):
        new_t = st.text_input(
            "Ticker symbol", placeholder="e.g. AAPL, TQQQ, GLD, MSFT",
            label_visibility="collapsed",
        )
        add_btn = st.form_submit_button("Add to portfolio", use_container_width=True)

    if add_btn and new_t:
        ticker = new_t.upper().strip()
        if ticker in st.session_state.tickers:
            st.warning(f"{ticker} already added.")
        else:
            with st.spinner(f"Looking up {ticker}…"):
                ok, name = validate_ticker(ticker)
            if ok:
                st.session_state.tickers.append(ticker)
                st.session_state.weights[ticker] = 10
                st.success(f"Added **{ticker}** — {name}")
                st.rerun()
            else:
                st.error(f"Could not find **{ticker}** on Yahoo Finance.")

    st.divider()

    # ── Weights ────────────────────────────────────────────────────────────────
    st.subheader("Allocations")
    to_remove = None
    for t in st.session_state.tickers:
        col_lbl, col_btn = st.columns([5, 1])
        col_lbl.markdown(f"**{t}**")
        if col_btn.button("✕", key=f"rm_{t}"):
            to_remove = t
        w = st.slider(
            t, 0, 100,
            value=st.session_state.weights.get(t, 10),
            key=f"sl_{t}",
            label_visibility="collapsed",
        )
        st.session_state.weights[t] = w

    if to_remove:
        st.session_state.tickers.remove(to_remove)
        st.session_state.weights.pop(to_remove, None)
        st.rerun()

    total_raw = sum(st.session_state.weights.get(t, 0) for t in st.session_state.tickers)
    st.caption(f"Raw total: **{total_raw}** → auto-normalized to 100%")

    st.divider()

    # ── Settings ───────────────────────────────────────────────────────────────
    st.subheader("Settings")
    start_date = st.date_input("Start date", date(2015, 1, 1))
    end_date   = st.date_input("End date", date.today())
    benchmark  = st.selectbox("Benchmark", ["SPY", "QQQ", "^GSPC", "IWM", "TLT", "None"])
    rf_pct     = st.number_input("Risk-free rate (%)", 0.0, 10.0, 4.5, 0.1)
    initial    = st.number_input("Initial ($)", 1_000, 10_000_000, 10_000, 1_000)
    rf = rf_pct / 100

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("Portfolio Visualizer")
st.caption("Fetch real price data for any ETF or stock and compare portfolio strategies.")

if not st.session_state.tickers:
    st.info("Add tickers in the sidebar to build your portfolio.")
    st.stop()

# ── Fetch ──────────────────────────────────────────────────────────────────────
fetch_set = list(st.session_state.tickers)
if benchmark != "None" and benchmark not in fetch_set:
    fetch_set.append(benchmark)

with st.spinner("Fetching market data…"):
    prices = fetch_prices(tuple(sorted(fetch_set)), str(start_date), str(end_date))

if prices.empty:
    st.error("No data returned. Check tickers and date range.")
    st.stop()

available = [t for t in st.session_state.tickers if t in prices.columns]
if not available:
    st.error("None of the selected tickers have data in this date range.")
    st.stop()

nw = norm_weights(available, st.session_state.weights)
rets = prices[available].pct_change().dropna()
port_rets = sum(nw[t] * rets[t] for t in nw)
m = compute_metrics(port_rets, rf, initial)

# Benchmark
bm, bm_m = None, None
if benchmark != "None" and benchmark in prices.columns:
    bm_rets = prices[benchmark].pct_change().dropna().reindex(port_rets.index).fillna(0)
    bm_m = compute_metrics(bm_rets, rf, initial)
    bm = benchmark

# ── Metrics row ────────────────────────────────────────────────────────────────
st.subheader("Performance summary")
c1, c2, c3, c4, c5, c6 = st.columns(6)

def safe_delta(port_val, metric_key, inverse=False):
    if bm_m is None: return None
    diff = port_val - bm_m[metric_key]
    sign = "+" if diff >= 0 else ""
    d_str = f"{sign}{diff:.2f} vs {bm}"
    if inverse:
        d_color = "inverse" if diff >= 0 else "normal"
    else:
        d_color = "normal" if diff >= 0 else "inverse"
    return d_str

c1.metric("CAGR",          f"{m['cagr']:.1f}%",          safe_delta(m['cagr'], 'cagr'))
c2.metric("Total return",  f"{m['total_ret']:.1f}%",      safe_delta(m['total_ret'], 'total_ret'))
c3.metric("Max drawdown",  f"{m['max_dd']:.1f}%",         safe_delta(m['max_dd'], 'max_dd', inverse=True))
c4.metric("Volatility",    f"{m['ann_vol']:.1f}%",        safe_delta(m['ann_vol'], 'ann_vol', inverse=True))
c5.metric("Sharpe ratio",  f"{m['sharpe']:.2f}",          safe_delta(m['sharpe'], 'sharpe'))
c6.metric("Calmar ratio",  f"{m['calmar']:.2f}",          safe_delta(m['calmar'], 'calmar'))

st.divider()

# ── Growth chart ───────────────────────────────────────────────────────────────
fig_g = go.Figure()
fig_g.add_trace(go.Scatter(
    x=m["cum"].index, y=m["cum"].round(2),
    name="Portfolio", line=dict(color="#534AB7", width=2.5),
    hovertemplate="%{x|%d %b %Y}  $%{y:,.0f}<extra>Portfolio</extra>",
))
if bm_m:
    fig_g.add_trace(go.Scatter(
        x=bm_m["cum"].index, y=bm_m["cum"].round(2),
        name=bm, line=dict(color="#B4B2A9", width=1.5, dash="dot"),
        hovertemplate=f"%{{x|%d %b %Y}}  $%{{y:,.0f}}<extra>{bm}</extra>",
    ))
fig_g.update_layout(
    title=f"Growth of ${initial:,.0f}",
    height=320, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=48, b=0),
    yaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="rgba(128,128,128,0.1)"),
    xaxis=dict(showgrid=False),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_g, use_container_width=True)

# ── Drawdown chart ─────────────────────────────────────────────────────────────
fig_d = go.Figure()
fig_d.add_trace(go.Scatter(
    x=m["dd"].index, y=m["dd"].round(2),
    name="Portfolio", fill="tozeroy",
    line=dict(color="#E24B4A", width=1.5),
    fillcolor="rgba(226,75,74,0.10)",
    hovertemplate="%{x|%d %b %Y}  %{y:.1f}%<extra>Drawdown</extra>",
))
if bm_m:
    fig_d.add_trace(go.Scatter(
        x=bm_m["dd"].index, y=bm_m["dd"].round(2),
        name=bm, line=dict(color="#B4B2A9", width=1.5, dash="dot"),
        hovertemplate=f"%{{x|%d %b %Y}}  %{{y:.1f}}%<extra>{bm} DD</extra>",
    ))
fig_d.update_layout(
    title="Drawdown from peak",
    height=210, hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=48, b=0),
    yaxis=dict(ticksuffix="%", gridcolor="rgba(128,128,128,0.1)"),
    xaxis=dict(showgrid=False),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_d, use_container_width=True)

# ── Annual returns ─────────────────────────────────────────────────────────────
st.subheader("Annual returns")
ann = annual_returns(port_rets)

fig_ann = go.Figure()
fig_ann.add_trace(go.Bar(
    x=[str(y.year) for y in ann.index],
    y=ann.round(2),
    name="Portfolio",
    marker_color=["#1D9E75" if v >= 0 else "#E24B4A" for v in ann],
    text=[f"{v:.1f}%" for v in ann],
    textposition="outside",
    hovertemplate="%{x}: %{y:.1f}%<extra>Portfolio</extra>",
))
if bm_m:
    bm_ann = annual_returns(bm_rets)
    fig_ann.add_trace(go.Bar(
        x=[str(y.year) for y in bm_ann.index],
        y=bm_ann.round(2),
        name=bm,
        marker_color="rgba(180,178,169,0.5)",
        hovertemplate=f"%{{x}}: %{{y:.1f}}%<extra>{bm}</extra>",
    ))
fig_ann.update_layout(
    height=280, barmode="group",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=20, b=0),
    yaxis=dict(ticksuffix="%", gridcolor="rgba(128,128,128,0.1)",
               zeroline=True, zerolinecolor="rgba(128,128,128,0.25)"),
    xaxis=dict(showgrid=False),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_ann, use_container_width=True)

# ── Expanded views ─────────────────────────────────────────────────────────────
col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    with st.expander("Individual asset performance", expanded=False):
        fig_ind = go.Figure()
        for i, t in enumerate(available):
            t_cum = (1 + rets[t]).cumprod() * initial
            fig_ind.add_trace(go.Scatter(
                x=t_cum.index, y=t_cum.round(2),
                name=t, line=dict(color=COLORS[i % len(COLORS)], width=1.8),
                hovertemplate=f"%{{x|%d %b %Y}}  $%{{y:,.0f}}<extra>{t}</extra>",
            ))
        fig_ind.update_layout(
            height=280, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=8, b=0),
            yaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="rgba(128,128,128,0.1)"),
            xaxis=dict(showgrid=False),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ind, use_container_width=True)

with col_exp2:
    with st.expander("Correlation matrix", expanded=False):
        if len(available) > 1:
            corr = rets[available].corr().round(2)
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=list(corr.columns), y=list(corr.index),
                colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                text=corr.values.round(2), texttemplate="%{text}",
                colorbar=dict(thickness=10, len=0.8),
            ))
            fig_corr.update_layout(
                height=280, margin=dict(l=0, r=0, t=8, b=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Add more tickers to see correlations.")

# ── Allocation donut ───────────────────────────────────────────────────────────
with st.expander("Allocation breakdown", expanded=False):
    fig_pie = go.Figure(go.Pie(
        labels=list(nw.keys()),
        values=[round(w * 100, 1) for w in nw.values()],
        marker_colors=COLORS[:len(nw)],
        hole=0.45,
        textinfo="label+percent",
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ))
    fig_pie.update_layout(
        height=280,
        showlegend=False,
        margin=dict(l=0, r=0, t=8, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Stats table ────────────────────────────────────────────────────────────────
with st.expander("Per-asset stats table", expanded=False):
    rows = []
    for t in available:
        t_rets = rets[t]
        t_m = compute_metrics(t_rets, rf, initial)
        rows.append({
            "Ticker": t,
            "Weight": f"{nw[t]*100:.1f}%",
            "CAGR": f"{t_m['cagr']:.1f}%",
            "Total return": f"{t_m['total_ret']:.1f}%",
            "Max drawdown": f"{t_m['max_dd']:.1f}%",
            "Volatility": f"{t_m['ann_vol']:.1f}%",
            "Sharpe": f"{t_m['sharpe']:.2f}",
            "Best year": f"{t_m['best_yr']:.1f}%",
            "Worst year": f"{t_m['worst_yr']:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Data via Yahoo Finance (yfinance). Assumes daily rebalancing to target weights. "
    "Past performance does not guarantee future results."
)
