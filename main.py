import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import norm
import seaborn as sns


# Function to calculate d1 and d2
def d1(S, X, T, r, sigma):
    return (np.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, X, T, r, sigma):
    return d1(S, X, T, r, sigma) - sigma * np.sqrt(T)

# Black-Scholes Formula for Call and Put options
def black_scholes(S, X, T, r, sigma, option_type='call'):
    d_1 = d1(S, X, T, r, sigma)
    d_2 = d2(S, X, T, r, sigma)
    
    if option_type == 'call':
        option_price = S * norm.cdf(d_1) - X * np.exp(-r * T) * norm.cdf(d_2)
    elif option_type == 'put':
        option_price = X * np.exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)
    else:
        raise ValueError("Option type must be 'call' or 'put'")
    
    return option_price

# Function to generate heatmap data
def generate_heatmap_data(spot_min, spot_max, vol_min, vol_max, X, T, r, option_type='call', num_points=50):
    spot_prices = np.linspace(spot_min, spot_max, num_points)
    volatilities = np.linspace(vol_min, vol_max, num_points)
    
    # Create a meshgrid of spot prices and volatilities
    spot_grid, vol_grid = np.meshgrid(spot_prices, volatilities)
    
    # Compute the option prices for each combination of spot price and volatility
    option_prices = np.zeros_like(spot_grid)
    for i in range(len(spot_prices)):
        for j in range(len(volatilities)):
            option_prices[j, i] = black_scholes(spot_prices[i], X, T, r, volatilities[j], option_type=option_type)
    
    return spot_grid, vol_grid, option_prices

# Function to plot the heatmap
def plot_heatmap(spot_grid, vol_grid, option_prices, title, ax):
    sns.heatmap(option_prices, xticklabels=np.round(spot_grid[0], 2), 
                yticklabels=np.round(vol_grid[:, 0], 2), cmap='RdYlGn', annot=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Volatility")

# Streamlit UI
st.title("Black-Scholes Option Pricing Heatmap")

# Sidebar Inputs
st.sidebar.header("Option Parameters")

X = st.sidebar.number_input("Strike Price", min_value=0.01, value=100.00, step=0.01)
T = st.sidebar.number_input("Time to Expiration (Years)", min_value=0.1,  value=1.0, step=0.1)
r = st.sidebar.number_input("Risk-Free Interest Rate (in %)", min_value=0.01, max_value=100.00, value=5.00, step=0.01) / 100
current_spot_price = st.sidebar.number_input("Current Asset Price", min_value=0.01, value=100.00, step=0.01)
current_volatility = st.sidebar.number_input("Current Volatility (in %)", min_value=0.01, max_value=100.00, value=20.00, step=0.01) / 100

st.sidebar.header("Spot Price and Volatility Range")

spot_min = st.sidebar.number_input("Minimum Spot Price", min_value=0.01, value=80.00, step=0.01)
spot_max = st.sidebar.number_input("Maximum Spot Price", min_value=0.01, value=120.00, step=0.01)
vol_min = st.sidebar.number_input("Minimum Volatility (in %)", min_value=0.01, max_value=100.00, value=10.00, step=0.01) / 100
vol_max = st.sidebar.number_input("Maximum Volatility (in %)", min_value=0.01, max_value=100.00, value=30.00, step=0.01) / 100

# New inputs for current asset price and volatility
st.sidebar.header("Grid")


num_points = st.sidebar.slider("Number of Points for Grid", min_value=10, max_value=100, value=25, step=1)

# Calculate the current option prices
call_price = black_scholes(current_spot_price, X, T, r, current_volatility, option_type='call')
put_price = black_scholes(current_spot_price, X, T, r, current_volatility, option_type='put')

# Summary Table
st.subheader("Option Parameters Summary")
summary_data = {
    "Current Spot Price": [f"{current_spot_price:.2f}"],
    "Strike Price": [f"{X:.2f}"],
    "Time to Maturity (Years)": [f"{T:.2f}"],
    "Volatility (σ)": [f"{current_volatility * 100:.2f}"],
    "Risk-Free Interest Rate (%)": [f"{r * 100:.2f}"],
}
st.table(summary_data)

# Display Call and Put Prices in smoother, rounded boxes
st.markdown(f"""
    <div style="display: flex; justify-content: space-around;">
        <div style="background-color: #90EE90; padding: 30px 20px; border-radius: 25px; width: 40%; text-align: center; box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.2);">
            <h3 style="color: #ffffff; font-weight: bold; margin-bottom: 10px;">CALL Value</h3>
            <h1 style="color: black; font-weight: bold;">${call_price:.2f}</h1>
        </div>
        <div style="background-color: #F4CCCC; padding: 30px 20px; border-radius: 25px; width: 40%; text-align: center; box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.2);">
            <h3 style="color: #ffffff; font-weight: bold; margin-bottom: 10px;">PUT Value</h3>
            <h1 style="color: black; font-weight: bold;">${put_price:.2f}</h1>
        </div>
    </div>
""", unsafe_allow_html=True)


# Generate heatmap data and plots automatically on startup
col1, col2 = st.columns([1,1], gap="small")

with col1:
    st.subheader("Call Option Prices Heatmap")
    fig, ax = plt.subplots(figsize=(6, 5))
    spot_grid_call, vol_grid_call, call_prices = generate_heatmap_data(spot_min, spot_max, vol_min, vol_max, X, T, r, option_type='call', num_points=num_points)
    plot_heatmap(spot_grid_call, vol_grid_call, call_prices, title="Call Option Prices", ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Put Option Prices Heatmap")
    fig, ax = plt.subplots(figsize=(6, 5))
    spot_grid_put, vol_grid_put, put_prices = generate_heatmap_data(spot_min, spot_max, vol_min, vol_max, X, T, r, option_type='put', num_points=num_points)
    plot_heatmap(spot_grid_put, vol_grid_put, put_prices, title="Put Option Prices", ax=ax)
    st.pyplot(fig)
