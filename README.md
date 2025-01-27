# Black-Scholes Option Pricing Heatmap Application

This application is a Python-based Streamlit app that calculates and visualizes option prices using the Black-Scholes formula. It generates interactive heatmaps for call and put options based on user-defined parameters.

## Features

- **Black-Scholes Model**: Calculate call and put option prices based on the Black-Scholes formula.
- **Interactive Heatmaps**: Visualize how option prices vary with changes in spot price and volatility.
- **Customizable Parameters**: Adjust key option pricing parameters such as strike price, time to expiration, risk-free rate, and volatility.
- **Real-Time Calculations**: Display call and put option prices based on current input values.

## Requirements

To run this application, you need the following Python libraries:

- `streamlit`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy`

## Installation

1. Clone this repository or download the script file `main.py`.
2. Install the required dependencies:
   ```bash
   pip install streamlit numpy pandas matplotlib seaborn scipy
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run main.py
   ```

## Usage

1. Launch the app by running the `streamlit` command mentioned above.
2. Use the sidebar to input the following parameters:
   - **Current Asset Price**: Spot price of the asset.
   - **Strike Price**: Strike price of the option.
   - **Time to Expiration (Years)**: Time until the option expires.
   - **Risk-Free Interest Rate (%)**: Annualized risk-free interest rate.
   - **Current Volatility (%)**: Implied volatility of the asset.
   - **Spot Price and Volatility Ranges**: Define the range for the heatmap axes.
   - **Grid Resolution**: Number of points for the heatmap grid.
3. View the calculated option prices and the corresponding heatmaps for call and put options.

## Example Output

- **Option Prices Summary**: Displays the current calculated call and put option prices.
- **Heatmaps**: Interactive visualizations showing how option prices vary with spot price and volatility.

## Code Structure

1. **`black_scholes`**: Computes the call and put option prices using the Black-Scholes formula.
2. **`generate_heatmap_data`**: Creates data for heatmap visualization.
3. **`plot_heatmap`**: Renders the heatmap using Seaborn.
4. **Streamlit UI**: Implements an interactive interface for parameter input and visualization.

## Screenshots

- **Summary Table and Option Prices**: Displays calculated values for the selected parameters.
- **Heatmaps**: Visual representation of call and put option prices over varying spot prices and volatilities.

## License

This project is licensed under the MIT License. Feel free to use and modify it as needed.

