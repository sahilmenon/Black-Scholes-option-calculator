# Black-Scholes Option Pricing Calculator

Real-time options pricing in Python/Streamlit: Black-Scholes + Monte Carlo (GBM) pricing, full Greeks, 3D volatility surfaces, an implied volatility solver (Brent's method), and live market data from Yahoo Finance.

**Live app:** [black-scholes-option-visualiser.streamlit.app](https://black-scholes-option-visualiser.streamlit.app/)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://black-scholes-option-visualiser.streamlit.app/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What it does

| Feature | Detail |
|---|---|
| **Pricing** | Black-Scholes call & put (4-decimal precision); auto-fill spot + historical vol from any ticker |
| **Greeks** | Delta, Gamma, Theta, Vega, Rho for both sides; bar chart comparison |
| **Heatmaps** | Price, P&L, and all Greeks across a grid of spot × volatility |
| **Volatility surface** | Interactive 3D surface: option price vs spot price × time to maturity |
| **Monte Carlo** | GBM with configurable sim count and seed; BS vs MC side-by-side with 95% CI and sample paths |
| **IV solver** | Back-solve implied vol from any market price (Brent's method); IV sensitivity curve |
| **Live mispricing** | Fetch real options chains via Yahoo Finance; BS model price vs market price by strike |

---

## Quickstart

```bash
git clone https://github.com/sahilmenon/Black-Scholes-option-calculator.git
cd Black-Scholes-option-calculator
pip install -r requirements.txt
streamlit run main.py
```

Dependencies: `streamlit numpy pandas matplotlib seaborn scipy yfinance`

---

## Usage

1. **Sidebar — Option Parameters**: set spot, strike, expiry, risk-free rate, and vol manually, or enter a ticker to auto-fill from live data.
2. **Sidebar — Heatmap Range**: define the spot and vol ranges for the grid visualisations.
3. **Sidebar — P&L Settings**: set purchase price for payoff and P&L heatmap calculations.
4. **Sidebar — Monte Carlo**: choose simulation count and random seed.
5. Navigate the **8 tabs**: Pricing · Greeks · Heatmaps · Volatility Surface · Payoff · Monte Carlo · IV Solver · Live Market Data.

---

## What I'd do next

**American options** — Black-Scholes prices European-exercise options only. Adding a binomial tree (CRR) and Longstaff-Schwartz LSM Monte Carlo for American puts would cover most real equity options markets, where early exercise matters for deep in-the-money puts near ex-dividend dates.

**Volatility smile/skew fitting** — the model prices at a single flat implied vol, but real markets have a skew (equity puts trade at higher IV than calls) and term structure. Fitting the SABR or SVI parametric model to a live options chain and plotting the fitted surface next to the flat-vol price would show exactly where the model breaks down and by how much — which is what a desk quant actually cares about.

**Portfolio-level Greeks** — the tool currently prices one contract. A multi-leg builder (straddle, iron condor, butterfly, or custom spread) with aggregated delta, gamma, and vega would make it usable for strategy construction and not just single-leg analysis.

**Historical backtest** — given a hypothetical past trade ("buy a 30-day ATM call on AAPL on 2024-01-15"), simulate the actual P&L vs the day-zero BS estimate using Yahoo Finance historical prices. The divergence between GBM's assumptions and real price paths (fat tails, volatility clustering) is more instructive than any textbook derivation.

---

## License

MIT — see source for full text.
