import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')
def arima(n):
    stock_name = yf.Ticker('TSLA')
    history = stock_name.history(period='2y', interval='1d')
    closing_prices = history['Close']
    history['Date'] = history.index
    dates = history['Date']
    stock_data = pd.DataFrame({'Date': dates, 'Close': closing_prices})
    stock_data['Log_Close'] = np.log(stock_data['Close'])
    stock_data = stock_data.dropna()
    lags = n
    for lag in range(1, lags + 1):
        stock_data[f'lag_{lag}'] = stock_data['Log_Close'].shift(lag)
    stock_data = stock_data.dropna()
    X = stock_data[[f'lag_{lag}' for lag in range(1, lags + 1)]]
    y = stock_data['Log_Close']
    split_index = int(len(stock_data))
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    model = LinearRegression()
    model.fit(X_train, y_train)
    #print(model.coef_)
    #print(model.intercept_)
    #adf_result = adfuller(stock_data['Log_Close'])
    #print(f"ADF Statistic: {adf_result[0]}")
    #print(f"p-value: {adf_result[1]}")
    #print("The R^2 accuracy of the model is: ",acc*100,"%")
    def future(n):
        latest_data = y_train[-n:] 
        coefficients=model.coef_
        intercept=model.intercept_
        sum=intercept
        for i in range (len(latest_data)):
            sum= sum+coefficients[i]*latest_data[i]
        print(np.exp(sum))
    future(n)
print(arima(n))
