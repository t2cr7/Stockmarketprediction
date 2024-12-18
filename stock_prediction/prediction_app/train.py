import yfinance as yf
import pandas as pd
import numpy as np
import os
import math
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.linear_model import LinearRegression
import joblib  # For saving models
from datetime import datetime

# Directory where trained models will be saved
MODEL_DIR = "trained_models"

# Ensure the directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# List of stock tickers to pre-train
tickers = [
        'ADANIPORTS.NS', 
        'ASIANPAINT.NS',
        'AXISBANK.NS', 
        'DIVISLAB.NS',
        'DRREDDY.NS',
        'EICHERMOT.NS', 
        'GRASIM.NS', 
        'HCLTECH.NS',
        'HDFCBANK.NS',  
        'HINDUNILVR.NS',
        'ICICIBANK.NS',
        'IOC.NS', 
        'INFY.NS', 
        'JSWSTEEL.NS',   
        'M&M.NS', 
        'POWERGRID.NS',  
        'SBILIFE.NS',
        'SBIN.NS',
        'SUNPHARMA.NS',
        'TATACONSUM.NS', 
        'TATAMOTORS.NS', 
        'TECHM.NS',  
        'UPL.NS',
        'ZEEL.NS'
]
  # Add more tickers as needed

def get_historical_data(ticker):
    """Fetch historical stock data."""
    end = datetime.now()
    start = datetime(end.year - 2, end.month, end.day)
    data = yf.download(ticker, start=start, end=end)
    data = data.reset_index()
    return data

def train_arima_model(data, ticker):
    """Train an ARIMA model on the stock data and save it."""
    data['Price'] = data['Close']
    data = data[['Date', 'Price']]
    data.index = pd.to_datetime(data['Date'])
    data = data['Price'].values

    size = int(len(data) * 0.80)
    train, test = data[0:size], data[size:len(data)]

    history = [x for x in train]
    predictions = []
    
    for t in range(len(test)):
        model = ARIMA(history, order=(6, 1, 0))
        model_fit = model.fit()
        history.append(test[t])

    # Save ARIMA model
    arima_model_path = os.path.join(MODEL_DIR, f"{ticker}_arima.pkl")
    joblib.dump(model_fit, arima_model_path)
    print(f"ARIMA model for {ticker} saved at {arima_model_path}")

def train_lstm_model(data, ticker):
    """Train an LSTM model on stock data and save it."""
    training_set = data[['Close']].values
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    
    X_train = []
    y_train = []
    
    for i in range(7, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-7:i, 0])
        y_train.append(training_set_scaled[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build LSTM model
    regressor = Sequential()
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.1))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    regressor.fit(X_train, y_train, epochs=25, batch_size=32)
    
    # Save LSTM model
    lstm_model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.h5")
    regressor.save(lstm_model_path)
    print(f"LSTM model for {ticker} saved at {lstm_model_path}")

def train_linear_regression_model(data, ticker):
    """Train a Linear Regression model on stock data and save it."""
    forecast_out = 7  # Forecast for the next 7 days
    data['Close after n days'] = data['Close'].shift(-forecast_out)

    df_new = data[['Close', 'Close after n days']].dropna()
    X = np.array(df_new['Close']).reshape(-1, 1)
    y = np.array(df_new['Close after n days']).reshape(-1, 1)

    X_train = X[:int(0.8 * len(X))]
    y_train = y[:int(0.8 * len(y))]

    # Train the linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Save the Linear Regression model
    lr_model_path = os.path.join(MODEL_DIR, f"{ticker}_linear_regression.pkl")
    joblib.dump(lr, lr_model_path)
    print(f"Linear Regression model for {ticker} saved at {lr_model_path}")

def train_all_models():
    """Fetch data and train all models for the tickers."""
    for ticker in tickers:
        print(f"Training models for {ticker}...")
        data = get_historical_data(ticker)
        train_arima_model(data, ticker)
        train_lstm_model(data, ticker)
        train_linear_regression_model(data, ticker)

if __name__ == "__main__":
    train_all_models()
