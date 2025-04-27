import pandas as pd import matplotlib.pyplot as plt
import seaborn as sns import statsmodels.api as sm from statsmodels.tsa.arima.model 
import ARIMA from statsmodels.tsa.stattools 
import adfuller from sklearn.metrics import mean_squared_error import numpy as np 
data = pd.read_csv("/AAPL.csv", parse_dates=["Date"], index_col="Date") 
data = data[['Close']]  # Use only the 'Close' price column data.dropna(inplace=True)  # Drop missing values data.head() 
plt.figure(figsize=(12, 6)) plt.plot(data['Close'],label='Close Price history') plt.title("stock price of AAPL over historys") plt.xlabel("year") plt.ylabel("Close Price") plt.legend() 
plt.show() 
  
plt.plot(data['Close'],label='Close Price history') 
fitted_values = model_fit.predict()   
 
# Calculate the residuals residuals = data['Close'] - fitted_values   # Plot the residuals plt.figure(figsize=(12, 6)) plt.plot(residuals) plt.title('Residuals of ARIMA Model') plt.xlabel('Date') plt.ylabel('Residuals') plt.show() 
forecast_steps = 120 forecast = model_fit.forecast(steps=forecast_steps) plt.figure(figsize=(12,6)) plt.plot(data['Close'], label='historical Price') plt.plot(pd.date_range(data.index[-1], periods=forecast_steps + 1, freq='D')[1:], forecast, label='Forecasted Price') plt.title("Stock Price Forecast") 
plt.xlabel("year") plt.ylabel(" stock Price") plt.legend() 

MSFT (STOCK PRICE FORCASTING)

import pandas as pd import matplotlib.pyplot as plt import seaborn as sns import statsmodels.api as sm from statsmodels.tsa.arima.model import ARIMA from statsmodels.tsa.stattools import adfuller from sklearn.metrics import mean_squared_error import numpy as np
data = pd.read_csv("/content/MSFT.csv", parse_dates=["Date"], index_col="Date") data = data[['Close']]  # Use only the 'Close' price column data.dropna(inplace=True)  # Drop missing values data.head() 
 data = pd.read_csv("/content/MSFT.csv", parse_dates=["Date"], index_col="Date") 

 plt.figure(figsize=(10, 6)) plt.plot(data['Close'],label='Close Price history') plt.title("stock price of MSFT over historys") plt.xlabel("year ") plt.ylabel("Close Price") plt.legend() 
plt.show() 
from statsmodels.tsa.seasonal import seasonal_decompose  
# Assuming 'data' is your DataFrame and 'Close' is the column to decompose decomposition = seasonal_decompose(data['Close'], model='additive', period=30)  # Adjust period if needed 
 
# Access the components trend = decomposition.trend seasonal = decomposition.seasonal residuals = decomposition.resid 
 
# Plot the components decomposition.plot() plt.show() 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf plot_acf(data['Close'], lags=30) plot_pacf(data['Close'], lags=30) plt.show() 
 
def stationary_test(series):     result = adfuller(series) 
    print('ADF Statistic: %f' % result[0])     print('p-value: %f' % result[1]) 
    print('Critical Values:')     for key, value in result[4].items():         print('\t%s: %.3f' % (key, value)) 
fitted_values = model_fit.predict() 
 
# Calculate the residuals residuals = data['Close'] - fitted_values 
 
# Plot the residuals plt.figure(figsize=(12, 6)) plt.plot(residuals) plt.title('Residuals of ARIMA Model AAPL') 
 
plt.xlabel('YEAR') plt.ylabel('Residuals') plt.show() 
odel = ARIMA(data['Close'], order=(2, 1, 0))  # Replace (5, 1, 0) with your desired order model_fit = model.fit() 
 
# Get the fitted values and residuals fitted_values = model_fit.predict() residuals = data['Close'] - fitted_values 
 Plot the original time series and fitted values plt.figure(figsize=(10, 6)) plt.plot(data['Close'], label='Original Data') plt.plot(fitted_values, label='Fitted Values', color='green') plt.title('ARIMA Model Fit MSFT') plt.xlabel('YEAR') plt.ylabel('Close Price') plt.legend() plt.show() 
 plt.figure(figsize=(12, 6)) plt.plot(residuals) plt.title('Residuals of ARIMA Model MSFT') plt.xlabel('YEAR') plt.ylabel('Residuals') plt.show() 
forecast_steps = 757 forecast = model_fit.forecast(steps=forecast_steps) plt.figure(figsize=(10,6)) plt.plot(data['Close'], label='historical Price') plt.plot(pd.date_range(data.index[-1], periods=forecast_steps + 1, freq='D')[1:], forecast, label='Forecasted Price') plt.title("Stock Price Forecast MSFT FOR 3 YRS ") 
