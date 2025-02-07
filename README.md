BLACK-SCHOLES OPTION PRICING HEATMAP APPLICATION

DESCRIPTION
-----------
A Python-based Streamlit app that calculates and visualizes option prices using the Black-Scholes formula. The application generates interactive heatmaps and 3D surfaces for call and put options based on user-defined parameters.

FEATURES
--------
- Black-Scholes Model calculations for call and put options
- Interactive heatmaps showing price sensitivity to spot price and volatility
- 3D volatility surfaces for comprehensive price visualization
- Real-time option price calculations
- Customizable parameters via sidebar controls
- Grid resolution adjustment for detailed analysis

REQUIREMENTS
------------
Python libraries:
- streamlit
- numpy
- pandas
- matplotlib
- seaborn
- scipy

INSTALLATION
------------
1. Clone the repository or download the script files
2. Install dependencies:
   pip install streamlit numpy pandas matplotlib seaborn scipy
3. Run the application:
   streamlit run main.py

USAGE
-----
1. Launch the app using the streamlit command
2. Adjust parameters in the sidebar:
   - Current Asset Price
   - Strike Price
   - Time to Expiration (Years)
   - Risk-Free Interest Rate (%)
   - Current Volatility (%)
   - Spot Price and Volatility Ranges
   - Grid Resolution
3. View the calculated option prices, heatmaps, and 3D surfaces

OUTPUT
------
- Current call and put option prices
- Price sensitivity heatmaps
- 3D volatility surfaces
- Parameter summary table

LICENSE
-------
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

CONTACT
-------
For questions or issues, please open an issue on the project repository. 