import os
import csv
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for rendering plots
from django.shortcuts import render, HttpResponse
import yfinance as yf
from sklearn.metrics import mean_squared_error
import numpy as np
import math
from statsmodels.tsa.arima.model import ARIMA
from keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Define the index view
def index(request):
    return render(request, 'index.html')

# Load pre-trained models
def load_arima_model(ticker):
    arima_model_path = os.path.join('trained_models', f'{ticker}_arima.pkl')
    return joblib.load(arima_model_path)

def load_lstm_model(ticker):
    lstm_model_path = os.path.join('trained_models', f'{ticker}_lstm.h5')
    return load_model(lstm_model_path)

def load_linear_regression_model(ticker):
    lr_model_path = os.path.join('trained_models', f'{ticker}_linear_regression.pkl')
    return joblib.load(lr_model_path)

def get_historical_data(ticker):
    """Fetch historical stock data and information for the given ticker."""
    stock = yf.Ticker(ticker)
    data = stock.history(period="2y")  # Get historical data for the past 2 years
    
    stock_info = {
        'name': stock.info.get('longName', ticker),
        'country': stock.info.get('country', 'Unknown'),
        'currency': stock.info.get('currency', 'USD'),
        'open': data['Open'].iloc[-1] if not data.empty else 'N/A',
        'high': data['High'].iloc[-1] if not data.empty else 'N/A',
        'low': data['Low'].iloc[-1] if not data.empty else 'N/A',
        'close': data['Close'].iloc[-1] if not data.empty else 'N/A',
        'volume': data['Volume'].iloc[-1] if not data.empty else 'N/A',
    }
    
    return data, stock_info

def get_arima_prediction(ticker):
    """Get ARIMA predictions for the stock ticker."""
    df, _ = get_historical_data(ticker)
    data = df['Close'].values
    size = int(len(data) * 0.80)
    train, test = data[0:size], data[size:len(data)]
    
    history = [x for x in train]
    predictions = []
    for t in range(len(test)):
        model = ARIMA(history, order=(6, 1, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        predictions.append(output[0])
        history.append(test[t])

    error_arima = math.sqrt(mean_squared_error(test, predictions))
    return predictions, test, error_arima

def get_lstm_prediction(ticker):
    """Get LSTM predictions for the stock ticker."""
    df, _ = get_historical_data(ticker)
    lstm_model = load_lstm_model(ticker)
    
    sequence_length = 60
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    predicted_scaled = lstm_model.predict(X)
    
    predicted_original = scaler.inverse_transform(predicted_scaled)
    y_original = scaler.inverse_transform(np.reshape(y, (-1, 1)))
    
    error_lstm = math.sqrt(mean_squared_error(y_original, predicted_original))
    
    return predicted_original, y_original, error_lstm

def get_lr_prediction(ticker):
    """Get Linear Regression predictions for the stock ticker."""
    df, _ = get_historical_data(ticker)
    lr_model = load_linear_regression_model(ticker)
    
    X = np.array(df[['Close']])
    y = np.array(df['Close'])
    
    predicted_lr = lr_model.predict(X)
    
    error_lr = math.sqrt(mean_squared_error(y, predicted_lr))
    return predicted_lr, y, error_lr

def ensure_static_dir():
    """Ensure the static directory exists."""
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    return static_dir

# Plotting Functions for ARIMA, LSTM, and Linear Regression Models
def plot_arima_vs_actual(test, predictions, filename):
    static_dir = ensure_static_dir()
    plt.figure(figsize=(10, 6))
    plt.plot(test, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title('ARIMA Model: Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(os.path.join(static_dir, filename))
    plt.close()

def plot_lstm_vs_actual(test, predictions, filename):
    static_dir = ensure_static_dir()
    plt.figure(figsize=(10, 6))
    plt.plot(test, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title('LSTM Model: Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(os.path.join(static_dir, filename))
    plt.close()

def plot_lr_vs_actual(test, predictions, filename):
    static_dir = ensure_static_dir()
    plt.figure(figsize=(10, 6))
    plt.plot(test, label='Actual Price')
    plt.plot(predictions, label='Predicted Price')
    plt.title('Linear Regression Model: Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(os.path.join(static_dir, filename))
    plt.close()

# View for handling stock predictions and plotting
def insertintotable(request):
    if request.method == 'POST':
        quote = request.POST.get('nm')
        if not quote:
            return HttpResponse("Stock ticker symbol is required", status=400)

        try:
            _, stock_info = get_historical_data(quote)
        except Exception as e:
            return HttpResponse(f"Error fetching stock data: {str(e)}", status=500)

        try:
            arima_pred, arima_test, error_arima = get_arima_prediction(quote)
            lstm_pred, lstm_test, error_lstm = get_lstm_prediction(quote)
            lr_pred, lr_test, error_lr = get_lr_prediction(quote)
        except Exception as e:
            return HttpResponse(f"Error during model prediction: {str(e)}", status=500)

        arima_pred = [float(p) for p in arima_pred]
        lstm_pred = [float(p) for p in lstm_pred]
        lr_pred = [float(p) for p in lr_pred]

        plot_arima_vs_actual(arima_test, arima_pred, 'ARIMA.png')
        plot_lstm_vs_actual(lstm_test, lstm_pred, 'LSTM.png')
        plot_lr_vs_actual(lr_test, lr_pred, 'LR.png')

        return render(request, 'results.html', {
            'quote': quote,
            'arima_pred': round(arima_pred[-1], 2),
            'lstm_pred': round(lstm_pred[-1], 2),
            'lr_pred': round(lr_pred[-1], 2),
            'error_arima': round(error_arima, 2),
            'error_lstm': round(error_lstm, 2),
            'error_lr': round(error_lr, 2),
            'stock_info': stock_info
        })

# About Us view
def about(request):
    return render(request, 'about.html')

# Tickers Info view
def tickers_info(request):
    """Reads tickers from a CSV file and displays their information."""
    csv_file_path = os.path.join(ensure_static_dir(), 'tickers.csv')

    if not os.path.exists(csv_file_path):
        return HttpResponse("Ticker CSV file not found.", status=404)

    headers = ['Ticker', 'Name', 'Country', 'Currency', 'Open', 'High', 'Low', 'Close', 'Volume']
    rows = []

    try:
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row if the CSV file has one
            for row in reader:
                ticker = row[0]  # Assuming the first column is the ticker symbol
                try:
                    _, stock_info = get_historical_data(ticker)
                    rows.append([
                        ticker,
                        stock_info['name'],
                        stock_info['country'],
                        stock_info['currency'],
                        stock_info['open'],
                        stock_info['high'],
                        stock_info['low'],
                        stock_info['close'],
                        stock_info['volume']
                    ])
                except Exception as e:
                    # If fetching data for a ticker fails, add an error message
                    rows.append([ticker, 'Data not available', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])
    except Exception as e:
        return HttpResponse(f"Error reading the CSV file: {str(e)}", status=500)

    return render(request, 'tickers_info.html', {
        'headers': headers,
        'rows': rows
    })
