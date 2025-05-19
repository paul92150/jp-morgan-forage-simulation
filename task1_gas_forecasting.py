import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from scipy.interpolate import CubicHermiteSpline

# --------------------------------------------------
# 1. Load the monthly data from CSV
# --------------------------------------------------
# The CSV file "nat_gas.csv" must have two columns: 'Dates' and 'Prices'
df = pd.read_csv('data/nat_gas.csv')
df['Dates'] = pd.to_datetime(df['Dates'], format="%m/%d/%y")
df = df.sort_values('Dates')

# Convert dates to ordinal numbers (for regression and interpolation)
df['Ordinal'] = df['Dates'].apply(lambda d: d.toordinal())
x_monthly = df['Ordinal'].values
y_monthly = df['Prices'].values

# --------------------------------------------------
# 2. Fit a linear regression and compute residuals
# --------------------------------------------------
X_reg = x_monthly.reshape(-1, 1)
reg = LinearRegression().fit(X_reg, y_monthly)

# Calculate the trend (predicted value) and the residual for each month
df['Trend'] = reg.predict(X_reg)
df['Residual'] = df['Prices'] - df['Trend']

# Compute average residual per calendar month (1 to 12)
df['Month'] = df['Dates'].dt.month
avg_residual_by_month = df.groupby('Month')['Residual'].mean()

# --------------------------------------------------
# 3. Extrapolate for the next 18 months using regression and seasonal residual
# --------------------------------------------------
last_date = df['Dates'].max()
forecast_dates = []
forecast_prices = []

for i in range(1, 19):  # next 18 months
    # Generate forecast date by adding i months to the last historical date
    forecast_date = last_date + pd.DateOffset(months=i)
    # Optionally adjust to month-end (if desired)
    forecast_date = forecast_date + pd.offsets.MonthEnd(0)
    forecast_dates.append(forecast_date)
    
    # Convert forecast date to ordinal
    forecast_ordinal = forecast_date.toordinal()
    # Compute the trend value for this date using the regression
    trend_val = reg.predict(np.array([[forecast_ordinal]]))[0]
    # Get the seasonal adjustment: average residual for this month
    month_key = forecast_date.month
    seasonal_adjustment = avg_residual_by_month.get(month_key, df['Residual'].mean())
    # Forecast price = trend + seasonal adjustment
    forecast_price = trend_val + seasonal_adjustment
    forecast_prices.append(forecast_price)

forecast_df = pd.DataFrame({
    'Dates': pd.to_datetime(forecast_dates),
    'Prices': forecast_prices
})
forecast_df['Ordinal'] = forecast_df['Dates'].apply(lambda d: d.toordinal())

# --------------------------------------------------
# 4. Combine historical monthly data with forecast monthly data
# --------------------------------------------------
combined_df = pd.concat([df[['Dates', 'Prices', 'Ordinal']], forecast_df], ignore_index=True)
combined_df = combined_df.sort_values('Dates').reset_index(drop=True)

# --------------------------------------------------
# 5. Compute derivatives and create Hermite cubic spline interpolation
# --------------------------------------------------
combined_x = combined_df['Ordinal'].values
combined_y = combined_df['Prices'].values
n_points = len(combined_x)
dy = np.zeros_like(combined_y)

# For the first point, use the slope between the first and second points
dy[0] = (combined_y[1] - combined_y[0]) / (combined_x[1] - combined_x[0])
# For the last point, use the slope between the last and second-to-last points
dy[-1] = (combined_y[-1] - combined_y[-2]) / (combined_x[-1] - combined_x[-2])
# For internal points, use the slope between the previous and next points
for i in range(1, n_points - 1):
    dy[i] = (combined_y[i + 1] - combined_y[i - 1]) / (combined_x[i + 1] - combined_x[i - 1])

# Build the Hermite cubic spline interpolation using the combined monthly data
hermite_spline = CubicHermiteSpline(combined_x, combined_y, dy)

# --------------------------------------------------
# 6. Define the final estimate_price function
# --------------------------------------------------
def estimate_price(date_input):
    """
    Estimate the natural gas price for a given date using the final Hermite interpolation.
    
    Args:
        date_input (str or datetime): The date for which to estimate the price.
    
    Returns:
        float: The estimated price.
    """
    if not isinstance(date_input, datetime):
        date_input = pd.to_datetime(date_input)
    ordinal = date_input.toordinal()
    return float(hermite_spline(ordinal))

# --------------------------------------------------
# 7. Visualization
# --------------------------------------------------
# Generate a daily date range from the earliest historical date to the last forecast date
final_start_date = combined_df['Dates'].min()
final_end_date = combined_df['Dates'].max()
daily_dates = pd.date_range(start=final_start_date, end=final_end_date, freq='D')
daily_ordinals = daily_dates.map(lambda d: d.toordinal())
final_estimated_prices = hermite_spline(daily_ordinals)

plt.figure(figsize=(14, 7))
# Plot historical monthly data (real data)
plt.plot(df['Dates'], df['Prices'], 'o', label='Historical Monthly Data', color='blue')
# Plot forecast monthly points
plt.plot(forecast_df['Dates'], forecast_df['Prices'], 'o', label='Forecast Monthly Data', color='red')
# Plot the final continuous estimate price function from Hermite interpolation
plt.plot(daily_dates, final_estimated_prices, label='Final Estimated Price Curve')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Final Estimated Price Function Using Monthly Extrapolation and Hermite Interpolation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
