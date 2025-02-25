import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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
    print(len(stock_data))
    lags = n
    for lag in range(1, lags + 1):
        stock_data[f'lag_{lag}'] = stock_data['Log_Close'].shift(lag)
    stock_data = stock_data.dropna()
    X = stock_data[[f'lag_{lag}' for lag in range(1, lags + 1)]]
    y = stock_data['Log_Close']
    split_index = int(len(stock_data) * 0.9)
    X_test = X.iloc[-1].values.reshape(1, -1)  # Ensure 2D
    y_test = y.iloc[-1]  # Scalar, no .values needed
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)[0]
    diff=np.exp(y_test)-np.exp(y_pred)
    acc=abs(diff)*100/np.exp(y_test)
    return acc