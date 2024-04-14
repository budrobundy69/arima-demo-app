# Importing necessary libraries for the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Creating a function to simulate time series data for Umsatzentwicklung (Sales development)
def generate_time_series_data(start_date, num_points, initial_value, trend, volatility):
    np.random.seed(42)

    dates = pd.date_range(start=start_date, periods=num_points)
    values = np.zeros(num_points)
    values[0] = initial_value
    for t in range(1, num_points):
        values[t] = values[t - 1] + trend + np.random.normal(0, volatility)
    return pd.Series(data=values, index=dates)

# Generating synthetic data for sales, price, and insolvency developments
sales_data = generate_time_series_data('2020-01-01', 365, 1000, 5, 50)
price_data = generate_time_series_data('2020-01-01', 365, 50, 0.2, 1)
insolvency_data = generate_time_series_data('2020-01-01', 365, 2, 0.05, 0.5)

# Checking for stationarity of the time series data
def test_stationarity(timeseries):
    result = adfuller(timeseries)
    return pd.Series(result[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

# Display the test results
sales_stationarity = test_stationarity(sales_data)
price_stationarity = test_stationarity(price_data)
insolvency_stationarity = test_stationarity(insolvency_data)

# Plotting the time series data
plt.figure(figsize=(12, 8))
plt.subplot(311)
plt.plot(sales_data, label='Sales Development')
plt.title('Sales Development')
plt.legend()

plt.subplot(312)
plt.plot(price_data, label='Price Development')
plt.title('Price Development')
plt.legend()

plt.subplot(313)
plt.plot(insolvency_data, label='Insolvency Development')
plt.title('Insolvency Development')
plt.legend()

plt.tight_layout()
plt.show()

# Using ARIMA models for each time series
# This is a basic ARIMA model, further analysis needed to find optimal parameters
def fit_arima_model(data, order):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit

# Fitting the models
sales_model = fit_arima_model(sales_data, (1,1,1))
price_model = fit_arima_model(price_data, (1,1,1))
insolvency_model = fit_arima_model(insolvency_data, (1,1,1))

# Forecasting the next 30 days
sales_forecast = sales_model.get_forecast(steps=30)
price_forecast = price_model.get_forecast(steps=30)
insolvency_forecast = insolvency_model.get_forecast(steps=30)

# Plotting the forecasts
plt.figure(figsize=(12, 8))
plt.subplot(311)
plt.plot(sales_data, label='Actual Sales')
plt.plot(pd.date_range(sales_data.index[-1], periods=31, freq='D')[1:], sales_forecast.predicted_mean, label='Forecasted Sales')
plt.title('Sales Forecast')
plt.legend()

plt.subplot(312)
plt.plot(price_data, label='Actual Prices')
plt.plot(pd.date_range(price_data.index[-1], periods=31, freq='D')[1:], price_forecast.predicted_mean, label='Forecasted Prices')
plt.title('Price Forecast')
plt.legend()

plt.subplot(313)
plt.plot(insolvency_data, label='Actual Insolvencies')
plt.plot(pd.date_range(insolvency_data.index[-1], periods=31, freq='D')[1:], insolvency_forecast.predicted_mean, label='Forecasted Insolvencies')
plt.title('Insolvency Forecast')
plt.legend()

plt.tight_layout()
plt.show()

# Output the summaries of the fitted models
print("Sales ARIMA Model Summary:")
print(sales_model.summary())
print("\nPrice ARIMA Model Summary:")
print(price_model.summary())
print("\nInsolvency ARIMA Model Summary:")
print(insolvency_model.summary())
