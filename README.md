# Stockmarketprediction
This project is a Stock Price Prediction System that uses historical stock data to predict future prices with pre-trained ARIMA, LSTM, and Linear Regression models. The system provides textual and graphical representations of predicted prices and errors for user-defined stock tickers.

# Features
>Fetch historical stock data using yfinance API.
>Predict stock prices using ARIMA, LSTM, and Linear Regression models.
>Display predictions and errors in both textual and graphical formats.
>User-friendly web interface built with Django.

# Tools and Technologies
>Python
>Django (Web framework)
>yfinance (API for fetching stock data)
>ARIMA, LSTM, Linear Regression (Prediction models)
>Joblib (Model serialization)
>Matplotlib (Data visualization)

# Prerequisites
>Python 3.8+
>Django 4.0+
>Required Python packages (see requirements.txt)

# Installation
1.Clone the repository:
git clone https://github.com/t2cr7/Stockmarketprediction.git  
cd stock-price-prediction
2.Install dependencies:
pip install -r requirements.txt  

# HOW TO RUN
1.Start the Django server:
 python manage.py runserver  
2.Access the web app:
 Open your browser and navigate to http://127.0.0.1:8000/.
3.Usage:
 >Enter a stock ticker symbol (e.g., AAPL, TSLA) in the input field.
 >View predictions and graphs for ARIMA, LSTM, and Linear Regression models.

# Project structure
stock-price-prediction/  
├── prediction_app/          # Django app folder  
│   ├── views.py             # Main logic for handling requests  
│   ├── models.py            # Placeholder for Django models (if used)  
│   ├── templates/           # HTML templates  
│   ├── static/              # Static assets (CSS, JS, images)  
├── requirements.txt         # Dependencies  
├── manage.py                # Django management script  
├── README.md                # Project documentation  
└── trained_models/          # Serialized pre-trained models (ARIMA, LSTM, LR)  

Feel free to fork, modify, and contribute to this project!







