# Black-Scholes Option Pricing Calculator

A comprehensive Streamlit app for pricing European options using the Black-Scholes model, with interactive visualizations, Greeks analysis, Monte Carlo simulation, implied volatility solving, and live market data integration.

**Live app:** https://black-scholes-option-visualiser.streamlit.app/

---

## Features

### Pricing
- Black-Scholes call and put prices with 4-decimal precision
- Auto-fill spot price and historical volatility from any ticker via Yahoo Finance

### Option Greeks
- Full Greeks table: **Delta, Gamma, Theta, Vega, Rho** for both call and put
- Bar chart comparison of call vs put Greeks

### Heatmaps
- **Price heatmaps** — option price across a grid of spot prices × volatilities
- **P&L heatmaps** — profit/loss relative to a user-defined purchase price
- **Greeks heatmaps** — Delta, Gamma, Vega, Theta across the same grid

### Volatility Surfaces
- Interactive 3D surfaces of option price vs spot price and time to maturity

### Payoff Diagram
- P&L at expiry for call and put positions
- Breakeven points and max loss clearly marked

### Monte Carlo Simulation
- Geometric Brownian Motion with configurable simulation count and seed
- Side-by-side comparison of Black-Scholes vs Monte Carlo prices
- 95% confidence intervals
- Sample path visualization

### Implied Volatility Solver
- Back-solves for IV from any market option price using Brent's method
- IV sensitivity curve showing how implied vol changes with market price

### Live Market Data & Mispricing
- Fetch real-time prices and options chains via Yahoo Finance
- Compare Black-Scholes model prices against market prices for each strike
- Mispricing bar charts (red = market overprices, green = market underprices)

---

## Installation

```bash
git clone https://github.com/sahilmenon/Black-Scholes-option-calculator.git
cd Black-Scholes-option-calculator
pip install -r requirements.txt
streamlit run main.py
```

### Dependencies
```
streamlit, numpy, pandas, matplotlib, seaborn, scipy, yfinance
```

---

## Usage

1. **Sidebar — Option Parameters**: set spot price, strike, time to expiry, risk-free rate, and volatility manually, or enter a ticker to auto-fill from live market data.
2. **Sidebar — Heatmap Range**: define the spot price and volatility ranges for grid visualizations.
3. **Sidebar — P&L Settings**: set the purchase price for payoff and P&L heatmap calculations.
4. **Sidebar — Monte Carlo**: choose number of simulations and random seed.
5. Navigate the **8 tabs** to explore different analyses.

---

## What's not included

- **American options** — Black-Scholes prices European options only; American options require binomial trees or finite differences
- **Multi-leg strategies** — no combined positions (spreads, straddles, condors)
- **Real-time streaming** — data refreshes on page reload, not live tick-by-tick

---

## License

MIT License — see source for full text.
