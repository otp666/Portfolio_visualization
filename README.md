# Portfolio Visualizer

An interactive dashboard for building ETF/stock portfolios, with real market data fetched via yfinance. Visualizes performance, drawdown, annual returns, correlations, and per-asset stats — all compared against a benchmark.

## Features

- Add any ETF or stock ticker (SPY, QQQ, TQQQ, AAPL, GLD, etc.)
- Allocation sliders — weights auto-normalize to 100%
- Quick presets (60/40 classic, TQQQ/BND, All-World, etc.)
- Benchmark comparison (SPY, QQQ, IWM, or none)
- Charts: portfolio growth, drawdown, annual returns, individual assets, correlation matrix
- Metrics: CAGR, total return, max drawdown, volatility, Sharpe, Calmar
- Per-asset stats table

---

## Run locally

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app opens at http://localhost:8501

---

## Deploy to Streamlit Community Cloud (free, shareable URL)

1. Push this folder to a **public GitHub repo** (or private — Streamlit Cloud supports both)

   ```bash
   git init
   git add app.py requirements.txt README.md
   git commit -m "initial portfolio visualizer"
   git remote add origin https://github.com/YOUR_USERNAME/portfolio-viz.git
   git push -u origin main
   ```

2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub

3. Click **New app**, select your repo and branch, set the main file path to `app.py`

4. Click **Deploy** — your app will be live at a URL like:
   `https://YOUR_USERNAME-portfolio-viz-app-HASH.streamlit.app`

5. Share that URL with anyone — no login required to view it.

---

## Project structure

```
portfolio_viz/
├── app.py            ← main Streamlit app
├── requirements.txt  ← Python dependencies
└── README.md         ← this file
```

---

## Notes

- Data is cached for 1 hour per ticker/date combination to avoid repeated API calls
- Assumes **daily rebalancing** to target weights (standard backtest assumption)
- Risk-free rate defaults to 4.5% (adjustable in sidebar)
- TQQQ and other leveraged ETFs will show extreme drawdowns — this is expected
