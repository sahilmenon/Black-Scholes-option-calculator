import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

st.set_page_config(page_title="Black-Scholes Calculator", layout="wide", page_icon="📈")

# ──────────────────────────── Core Math ──────────────────────────────────────

def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    if option_type == 'call':
        return S * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)
    return K * np.exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)

def compute_greeks(S, K, T, r, sigma, option_type='call'):
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d_1)
    sqrt_T = np.sqrt(T)
    exp_rT = np.exp(-r * T)

    delta = norm.cdf(d_1) if option_type == 'call' else norm.cdf(d_1) - 1
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T / 100  # per 1% vol move
    if option_type == 'call':
        theta = (-S * pdf_d1 * sigma / (2 * sqrt_T) - r * K * exp_rT * norm.cdf(d_2)) / 365
        rho = K * T * exp_rT * norm.cdf(d_2) / 100
    else:
        theta = (-S * pdf_d1 * sigma / (2 * sqrt_T) + r * K * exp_rT * norm.cdf(-d_2)) / 365
        rho = -K * T * exp_rT * norm.cdf(-d_2) / 100

    return {'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega, 'Rho': rho}

def implied_volatility(market_price, S, K, T, r, option_type='call'):
    intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    if market_price <= intrinsic:
        return None
    try:
        return brentq(
            lambda v: black_scholes(S, K, T, r, v, option_type) - market_price,
            1e-6, 10.0, xtol=1e-6, maxiter=500
        )
    except (ValueError, RuntimeError):
        return None

# ──────────────────────────── Cached Data Generation ─────────────────────────

@st.cache_data
def generate_price_heatmap(spot_min, spot_max, vol_min, vol_max, K, T, r, option_type, num_points):
    spots = np.linspace(spot_min, spot_max, num_points)
    vols = np.linspace(vol_min, vol_max, num_points)
    sg, vg = np.meshgrid(spots, vols)
    prices = np.vectorize(lambda s, v: black_scholes(s, K, T, r, v, option_type))(sg, vg)
    return sg, vg, prices

@st.cache_data
def generate_greek_heatmap(spot_min, spot_max, vol_min, vol_max, K, T, r, greek, option_type, num_points):
    spots = np.linspace(spot_min, spot_max, num_points)
    vols = np.linspace(vol_min, vol_max, num_points)
    sg, vg = np.meshgrid(spots, vols)
    values = np.vectorize(
        lambda s, v: compute_greeks(s, K, T, r, v, option_type)[greek]
    )(sg, vg)
    return sg, vg, values

@st.cache_data
def generate_vol_surface(spot_min, spot_max, T_max, K, r, sigma, option_type, num_points):
    spots = np.linspace(spot_min, spot_max, num_points)
    times = np.linspace(0.01, T_max, num_points)
    sg, tg = np.meshgrid(spots, times)
    prices = np.vectorize(lambda s, t: black_scholes(s, K, t, r, sigma, option_type))(sg, tg)
    return sg, tg, prices

@st.cache_data(ttl=300)
def fetch_live_data(ticker):
    if not YFINANCE_AVAILABLE:
        return None, None
    try:
        stock = yf.Ticker(ticker.upper())

        # fast_info is reliable in yfinance 0.2+
        price = None
        try:
            price = stock.fast_info.last_price
        except Exception:
            pass

        # fall back to recent history close
        hist = stock.history(period='1y')
        if not price and not hist.empty:
            price = float(hist['Close'].iloc[-1])

        # last resort: info dict (slow, fields vary by version)
        if not price:
            info = stock.info
            price = (info.get('currentPrice') or info.get('regularMarketPrice')
                     or info.get('previousClose'))

        hist_vol = None
        if not hist.empty and len(hist) > 20:
            returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
            hist_vol = float(returns.std() * np.sqrt(252))

        return (float(price) if price else None), hist_vol
    except Exception:
        return None, None

@st.cache_data(ttl=300)
def fetch_expirations(ticker):
    if not YFINANCE_AVAILABLE:
        return []
    try:
        return list(yf.Ticker(ticker.upper()).options)
    except Exception:
        return []

@st.cache_data(ttl=300)
def fetch_options_chain(ticker, expiry):
    if not YFINANCE_AVAILABLE:
        return None
    try:
        return yf.Ticker(ticker.upper()).option_chain(expiry)
    except Exception:
        return None

@st.cache_data
def run_monte_carlo(S, K, T, r, sigma, num_sims, seed):
    np.random.seed(seed)
    steps = max(int(T * 252), 30)
    dt = T / steps
    Z = np.random.standard_normal((steps, num_sims))
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    prices = S * np.cumprod(np.exp(log_returns), axis=0)
    final = prices[-1]
    disc = np.exp(-r * T)
    call_payoffs = np.maximum(final - K, 0)
    put_payoffs = np.maximum(K - final, 0)
    mc_call = disc * np.mean(call_payoffs)
    mc_put = disc * np.mean(put_payoffs)
    mc_call_se = disc * np.std(call_payoffs) / np.sqrt(num_sims)
    mc_put_se = disc * np.std(put_payoffs) / np.sqrt(num_sims)
    paths = np.vstack([np.full(num_sims, S), prices])[:, :100]
    time_ax = np.linspace(0, T, steps + 1)
    return mc_call, mc_put, mc_call_se, mc_put_se, paths, time_ax

# ──────────────────────────── Plot Functions ──────────────────────────────────

def _heatmap_labels(arr, n_labels=6):
    n = max(1, len(arr) // n_labels)
    return [f'{v:.1f}' if i % n == 0 else '' for i, v in enumerate(arr)]

def plot_heatmap(sg, vg, data, title, ax, center=None, cmap='RdBu_r'):
    x_labels = _heatmap_labels(sg[0])
    y_labels = _heatmap_labels(vg[:, 0])
    kwargs = dict(xticklabels=x_labels, yticklabels=y_labels,
                  cmap=cmap, annot=False, ax=ax)
    if center is not None:
        kwargs['center'] = center
    sns.heatmap(data, **kwargs)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Volatility")
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', rotation=0, labelsize=7)

def plot_vol_surface(sg, tg, prices, title):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(sg, tg, prices, cmap='jet',
                           edgecolor='black', linewidth=0.3, rstride=1, cstride=1)
    ax.view_init(elev=20, azim=-45)
    ax.xaxis.set_major_locator(plt.MaxNLocator(8))
    ax.yaxis.set_major_locator(plt.MaxNLocator(8))
    ax.zaxis.set_major_locator(plt.MaxNLocator(8))
    ax.grid(True, linestyle='-', alpha=0.4)
    ax.set_xlabel('Spot Price')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Option Price')
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()
    return fig

def plot_payoff(spot_min, spot_max, K, call_prem, put_prem):
    spots = np.linspace(spot_min * 0.7, spot_max * 1.3, 500)
    call_pnl = np.maximum(spots - K, 0) - call_prem
    put_pnl = np.maximum(K - spots, 0) - put_prem
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, pnl, color, label, be in [
        (ax1, call_pnl, '#2ECC71', 'Call', K + call_prem),
        (ax2, put_pnl, '#E74C3C', 'Put',  K - put_prem),
    ]:
        ax.plot(spots, pnl, color=color, linewidth=2)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.axvline(K, color='gray', linewidth=1, linestyle='--', label=f'Strike = {K:.2f}')
        ax.axvline(be, color=color, linewidth=1, linestyle=':', label=f'Breakeven = {be:.2f}')
        ax.fill_between(spots, pnl, 0, where=(pnl > 0), alpha=0.15, color='green')
        ax.fill_between(spots, pnl, 0, where=(pnl < 0), alpha=0.15, color='red')
        ax.set_title(f'{label} Payoff at Expiry')
        ax.set_xlabel('Spot Price at Expiry')
        ax.set_ylabel('P&L ($)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_mc(paths, time_ax, S, K, call_bs, put_bs, mc_call, mc_put):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i in range(paths.shape[1]):
        ax1.plot(time_ax, paths[:, i], alpha=0.1, linewidth=0.4, color='steelblue')
    ax1.axhline(K, color='red', linewidth=1.5, linestyle='--', label=f'Strike = {K}')
    ax1.axhline(S, color='black', linewidth=1, linestyle='--', label=f'S₀ = {S}')
    ax1.set_title(f'GBM Price Paths ({paths.shape[1]} shown)')
    ax1.set_xlabel('Time (Years)')
    ax1.set_ylabel('Asset Price')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    x = np.arange(2)
    w = 0.35
    b1 = ax2.bar(x - w/2, [call_bs, put_bs], w, label='Black-Scholes', color='steelblue', alpha=0.85)
    b2 = ax2.bar(x + w/2, [mc_call, mc_put], w, label='Monte Carlo', color='darkorange', alpha=0.85)
    for bars in [b1, b2]:
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Call', 'Put'])
    ax2.set_ylabel('Option Price ($)')
    ax2.set_title('Black-Scholes vs Monte Carlo')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig

# ──────────────────────────── Streamlit UI ───────────────────────────────────

st.title("Black-Scholes Option Pricing Calculator")

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("Real-Time Data (Optional)")
ticker = st.sidebar.text_input("Ticker Symbol", value="", placeholder="e.g. AAPL, MSFT")
live_price, live_vol = None, None
if ticker.strip():
    live_price, live_vol = fetch_live_data(ticker.strip())
    if live_price:
        msg = f"${live_price:.2f}"
        if live_vol:
            msg += f"  |  σ_hist = {live_vol*100:.1f}%"
        st.sidebar.success(msg)
    else:
        st.sidebar.error("Ticker not found or data unavailable.")

st.sidebar.header("Option Parameters")
default_spot = float(live_price) if live_price else 100.00
default_vol_pct = float(live_vol * 100) if live_vol else 20.00

S = st.sidebar.number_input("Current Asset Price", min_value=0.01, value=round(default_spot, 2), step=0.01)
K = st.sidebar.number_input("Strike Price", min_value=0.01, value=round(default_spot, 2), step=0.01)
T = st.sidebar.number_input("Time to Expiration (Years)", min_value=0.01, value=1.0, step=0.1)
r = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.01, max_value=100.0, value=5.00, step=0.01) / 100
sigma = st.sidebar.number_input("Volatility (%)", min_value=0.01, max_value=500.0,
                                 value=round(default_vol_pct, 2), step=0.01) / 100

st.sidebar.header("Heatmap Range")
spot_min = st.sidebar.number_input("Min Spot", min_value=0.01, value=round(S * 0.8, 2), step=0.01)
spot_max = st.sidebar.number_input("Max Spot", min_value=0.01, value=round(S * 1.2, 2), step=0.01)
vol_min = st.sidebar.number_input("Min Vol (%)", min_value=0.01, max_value=500.0, value=10.0, step=0.01) / 100
vol_max = st.sidebar.number_input("Max Vol (%)", min_value=0.01, max_value=500.0, value=50.0, step=0.01) / 100
num_points = st.sidebar.slider("Grid Points", min_value=10, max_value=50, value=20, step=5)

# Compute prices here so P&L defaults are sensible
call_price = black_scholes(S, K, T, r, sigma, 'call')
put_price  = black_scholes(S, K, T, r, sigma, 'put')

st.sidebar.header("P&L Settings")
call_prem = st.sidebar.number_input("Call Purchase Price ($)", min_value=0.0,
                                     value=round(call_price, 4), step=0.01)
put_prem  = st.sidebar.number_input("Put Purchase Price ($)", min_value=0.0,
                                     value=round(put_price, 4), step=0.01)

st.sidebar.header("Monte Carlo")
num_sims = st.sidebar.select_slider("Simulations", options=[1000, 5000, 10000, 50000], value=10000)
mc_seed  = st.sidebar.number_input("Seed", min_value=0, max_value=9999, value=42)

# Compute Greeks
call_g = compute_greeks(S, K, T, r, sigma, 'call')
put_g  = compute_greeks(S, K, T, r, sigma, 'put')

# ── Tabs ─────────────────────────────────────────────────────────────────────

(tab_overview, tab_heat, tab_greek_heat,
 tab_surf, tab_payoff, tab_mc, tab_iv, tab_market) = st.tabs([
    "Overview", "Price & P&L Heatmaps", "Greeks Heatmaps",
    "Vol Surfaces", "Payoff Diagram", "Monte Carlo",
    "IV Calculator", "Market Data"
])

# ─── Overview ────────────────────────────────────────────────────────────────

with tab_overview:
    st.table(pd.DataFrame({
        "Asset Price": [f"${S:.2f}"], "Strike": [f"${K:.2f}"],
        "Time": [f"{T:.2f}y"], "Volatility": [f"{sigma*100:.2f}%"],
        "Rate": [f"{r*100:.2f}%"],
    }))

    st.markdown(f"""
        <div style="display:flex;justify-content:space-around;margin:10px 0 24px;">
            <div style="background:#90EE90;padding:25px 20px;border-radius:20px;width:42%;
                        text-align:center;box-shadow:3px 3px 10px rgba(0,0,0,0.15);">
                <h3 style="color:#1a5c1a;margin-bottom:8px;">CALL Value</h3>
                <h1 style="color:#000;">${call_price:.4f}</h1>
            </div>
            <div style="background:#F4CCCC;padding:25px 20px;border-radius:20px;width:42%;
                        text-align:center;box-shadow:3px 3px 10px rgba(0,0,0,0.15);">
                <h3 style="color:#7a1a1a;margin-bottom:8px;">PUT Value</h3>
                <h1 style="color:#000;">${put_price:.4f}</h1>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("Option Greeks")
    greek_info = {
        'Delta': ('Δ', 'Price change per $1 spot move'),
        'Gamma': ('Γ', 'Delta change per $1 spot move'),
        'Theta': ('Θ', 'Price change per calendar day (time decay)'),
        'Vega':  ('ν', 'Price change per 1% volatility move'),
        'Rho':   ('ρ', 'Price change per 1% interest rate move'),
    }
    st.table(pd.DataFrame({
        'Greek':       [f"{v[0]} {k}" for k, v in greek_info.items()],
        'Description': [v[1] for v in greek_info.values()],
        'Call':        [f"{call_g[k]:+.6f}" for k in greek_info],
        'Put':         [f"{put_g[k]:+.6f}"  for k in greek_info],
    }))

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    for ax, (greek, (sym, _)) in zip(axes, greek_info.items()):
        bars = ax.bar(['Call', 'Put'], [call_g[greek], put_g[greek]],
                      color=['#4CAF50', '#F44336'], alpha=0.82)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f'{h:.4f}', ha='center',
                    va='bottom' if h >= 0 else 'top', fontsize=7)
        ax.set_title(f'{sym} {greek}', fontsize=10)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ─── Price & P&L Heatmaps ────────────────────────────────────────────────────

with tab_heat:
    st.subheader("Option Prices — Spot × Volatility")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(7, 6))
        sg, vg, cp = generate_price_heatmap(spot_min, spot_max, vol_min, vol_max,
                                             K, T, r, 'call', num_points)
        plot_heatmap(sg, vg, cp, "Call Prices ($)", ax)
        st.pyplot(fig); plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(7, 6))
        sg2, vg2, pp = generate_price_heatmap(spot_min, spot_max, vol_min, vol_max,
                                               K, T, r, 'put', num_points)
        plot_heatmap(sg2, vg2, pp, "Put Prices ($)", ax)
        st.pyplot(fig); plt.close()

    st.subheader("P&L Heatmaps — vs Purchase Price")
    st.caption(f"Call purchased at ${call_prem:.4f} | Put purchased at ${put_prem:.4f} — adjust in sidebar")
    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_heatmap(sg, vg, cp - call_prem, "Call P&L ($)", ax, center=0)
        st.pyplot(fig); plt.close()
    with col4:
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_heatmap(sg2, vg2, pp - put_prem, "Put P&L ($)", ax, center=0)
        st.pyplot(fig); plt.close()

# ─── Greeks Heatmaps ──────────────────────────────────────────────────────────

with tab_greek_heat:
    greeks_to_plot = {
        'Delta': (0,    'RdBu_r'),
        'Gamma': (None, 'YlOrRd'),
        'Vega':  (None, 'YlOrRd'),
        'Theta': (0,    'RdBu_r'),
    }
    for greek, (center, cmap) in greeks_to_plot.items():
        st.subheader(f"{greek} — Spot × Volatility")
        ca, cb = st.columns(2)
        with ca:
            fig, ax = plt.subplots(figsize=(7, 5))
            sg_g, vg_g, vals = generate_greek_heatmap(
                spot_min, spot_max, vol_min, vol_max, K, T, r, greek, 'call', num_points)
            plot_heatmap(sg_g, vg_g, vals, f"Call {greek}", ax, center=center, cmap=cmap)
            st.pyplot(fig); plt.close()
        with cb:
            fig, ax = plt.subplots(figsize=(7, 5))
            sg_g, vg_g, vals = generate_greek_heatmap(
                spot_min, spot_max, vol_min, vol_max, K, T, r, greek, 'put', num_points)
            plot_heatmap(sg_g, vg_g, vals, f"Put {greek}", ax, center=center, cmap=cmap)
            st.pyplot(fig); plt.close()

# ─── Volatility Surfaces ──────────────────────────────────────────────────────

with tab_surf:
    st.subheader("Option Price Surfaces — Spot × Time to Maturity")
    col5, col6 = st.columns(2)
    with col5:
        sg_s, tg_s, cp_s = generate_vol_surface(spot_min, spot_max, T, K, r, sigma, 'call', num_points)
        st.pyplot(plot_vol_surface(sg_s, tg_s, cp_s, "Call Price Surface"))
        plt.close()
    with col6:
        sg_s, tg_s, pp_s = generate_vol_surface(spot_min, spot_max, T, K, r, sigma, 'put', num_points)
        st.pyplot(plot_vol_surface(sg_s, tg_s, pp_s, "Put Price Surface"))
        plt.close()

# ─── Payoff Diagram ───────────────────────────────────────────────────────────

with tab_payoff:
    st.subheader("Payoff at Expiry")
    st.caption(f"Based on purchase prices set in sidebar — Call: ${call_prem:.4f} | Put: ${put_prem:.4f}")
    st.pyplot(plot_payoff(spot_min, spot_max, K, call_prem, put_prem))
    plt.close()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Call Breakeven", f"${K + call_prem:.2f}")
    c2.metric("Max Call Loss",  f"−${call_prem:.4f}")
    c3.metric("Put Breakeven",  f"${K - put_prem:.2f}")
    c4.metric("Max Put Loss",   f"−${put_prem:.4f}")

# ─── Monte Carlo ──────────────────────────────────────────────────────────────

with tab_mc:
    st.subheader("Monte Carlo Simulation — Geometric Brownian Motion")
    mc_call, mc_put, mc_call_se, mc_put_se, paths, time_ax = run_monte_carlo(
        S, K, T, r, sigma, int(num_sims), int(mc_seed)
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BS Call",  f"${call_price:.4f}")
    m2.metric("MC Call",  f"${mc_call:.4f}",  delta=f"{mc_call  - call_price:+.4f}")
    m3.metric("BS Put",   f"${put_price:.4f}")
    m4.metric("MC Put",   f"${mc_put:.4f}",   delta=f"{mc_put   - put_price:+.4f}")
    st.caption(
        f"95% CI — Call: [${mc_call - 1.96*mc_call_se:.4f}, ${mc_call + 1.96*mc_call_se:.4f}]"
        f" | Put: [${mc_put - 1.96*mc_put_se:.4f}, ${mc_put + 1.96*mc_put_se:.4f}]"
    )
    st.pyplot(plot_mc(paths, time_ax, S, K, call_price, put_price, mc_call, mc_put))
    plt.close()

# ─── IV Calculator ────────────────────────────────────────────────────────────

with tab_iv:
    st.subheader("Implied Volatility Solver")
    st.write("Enter a market option price to back-solve for implied volatility using Brent's method.")
    col_iv1, col_iv2 = st.columns([1, 2])
    with col_iv1:
        iv_type = st.selectbox("Option Type", ['call', 'put'])
        iv_mkt  = st.number_input("Market Option Price ($)", min_value=0.001,
                                   value=round(call_price, 4), step=0.01)
    with col_iv2:
        iv = implied_volatility(iv_mkt, S, K, T, r, iv_type)
        if iv is not None:
            i1, i2, i3 = st.columns(3)
            i1.metric("Implied Volatility", f"{iv*100:.4f}%")
            i2.metric("Model Volatility (σ)", f"{sigma*100:.2f}%")
            i3.metric("Difference", f"{(iv - sigma)*100:+.4f}%")
        else:
            st.error("No solution found — market price may be at or below intrinsic value.")

    if iv is not None:
        price_lo = max(0.001, iv_mkt * 0.1)
        price_hi = iv_mkt * 4
        price_range = np.linspace(price_lo, price_hi, 150)
        iv_curve = []
        for p in price_range:
            v = implied_volatility(p, S, K, T, r, iv_type)
            iv_curve.append(v * 100 if v is not None else np.nan)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(price_range, iv_curve, 'steelblue', linewidth=2)
        ax.axvline(iv_mkt, color='red', linestyle='--', label=f'Market = ${iv_mkt:.3f}')
        ax.axhline(iv * 100, color='green', linestyle='--', label=f'IV = {iv*100:.2f}%')
        ax.set_xlabel('Market Option Price ($)')
        ax.set_ylabel('Implied Volatility (%)')
        ax.set_title('Implied Volatility vs Market Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

# ─── Market Data ──────────────────────────────────────────────────────────────

with tab_market:
    st.subheader("Live Market Data & Mispricing")
    if not ticker.strip():
        st.info("Enter a ticker symbol in the sidebar to compare Black-Scholes model prices "
                "against live market option prices.")
    elif not YFINANCE_AVAILABLE:
        st.error("yfinance not installed.")
    elif not live_price:
        st.error(
            f"Could not load price data for **{ticker.upper()}**. "
            "Make sure it's a valid US-listed ticker (e.g. AAPL, MSFT, TSLA)."
        )
    else:
        info_label = f"**{ticker.upper()}** — ${live_price:.2f}"
        if live_vol:
            info_label += f" | 1Y historical vol: **{live_vol*100:.1f}%**"
        st.success(info_label)

        expirations = fetch_expirations(ticker.strip())
        if not expirations:
            st.warning(
                "No options chain found for this ticker. "
                "Options data is only available for US-listed equities with active options markets."
            )
        else:
            selected_exp = st.selectbox("Expiration Date", expirations)
            chain = fetch_options_chain(ticker.strip(), selected_exp)

            if chain is None:
                st.error("Could not load the options chain for this expiration.")
            else:
                exp_dt = datetime.strptime(selected_exp, '%Y-%m-%d')
                T_mkt = max((exp_dt - datetime.now()).days / 365, 0.001)
                st.caption(
                    f"Time to expiry: **{T_mkt:.3f} years** | "
                    f"Using σ = {sigma*100:.1f}% from sidebar"
                )

                for opt_label, df_raw, bs_type in [
                    ('Calls', chain.calls, 'call'),
                    ('Puts',  chain.puts,  'put'),
                ]:
                    df = df_raw[['strike', 'bid', 'ask', 'lastPrice',
                                 'impliedVolatility', 'volume', 'openInterest']].copy()
                    df = df[df['volume'].fillna(0) > 0].copy()
                    if df.empty:
                        continue

                    mid = (df['bid'] + df['ask']) / 2
                    df['marketPrice'] = np.where(mid > 0, mid, df['lastPrice'])
                    df['bsPrice'] = df['strike'].apply(
                        lambda k: black_scholes(live_price, k, T_mkt, r, sigma, bs_type)
                    )
                    df['mispricing']    = (df['marketPrice'] - df['bsPrice']).round(4)
                    df['mispricingPct'] = ((df['mispricing'] / df['bsPrice']) * 100).round(2)

                    st.subheader(f"{opt_label} — {selected_exp}")
                    display = df[['strike', 'marketPrice', 'bsPrice', 'mispricing',
                                  'mispricingPct', 'impliedVolatility', 'volume']].rename(columns={
                        'strike': 'Strike', 'marketPrice': 'Market Price',
                        'bsPrice': 'BS Price', 'mispricing': 'Δ ($)',
                        'mispricingPct': 'Δ (%)', 'impliedVolatility': 'Mkt IV',
                        'volume': 'Volume'
                    }).round(4)
                    st.dataframe(display, use_container_width=True)

                    step = df['strike'].diff().median()
                    bar_width = step * 0.8 if (not np.isnan(step) and step > 0) else 1.0
                    fig, ax = plt.subplots(figsize=(11, 4))
                    colors = ['#E74C3C' if m > 0 else '#2ECC71' for m in df['mispricing']]
                    ax.bar(df['strike'], df['mispricing'], color=colors, alpha=0.8, width=bar_width)
                    ax.axhline(0, color='black', linewidth=1)
                    ax.axvline(live_price, color='blue', linestyle='--', linewidth=1.5,
                               label=f'Spot = ${live_price:.2f}')
                    ax.set_xlabel('Strike Price')
                    ax.set_ylabel('Market − BS Price ($)')
                    ax.set_title(f'{ticker.upper()} {opt_label} Mispricing (σ = {sigma*100:.1f}%)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()

                st.caption(
                    "Red = market overprices vs model | Green = market underprices. "
                    "Adjust σ in sidebar to see how IV assumptions shift the mispricing."
                )
