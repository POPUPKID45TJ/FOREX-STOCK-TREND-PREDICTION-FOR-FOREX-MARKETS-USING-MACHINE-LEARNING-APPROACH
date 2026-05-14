import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# -----------------------------
# Utility Functions
# -----------------------------

def prepare_stock_data(ticker, start_date, end_date):
    """Download stock data and flatten MultiIndex if needed."""
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.error(f"No data found for {ticker}.")
        return pd.DataFrame()

    # Flatten MultiIndex if multiple tickers
    if isinstance(df.columns, pd.MultiIndex):
        df = df.stack(level=1).reset_index()
    else:
        df = df.reset_index()

    return df

def load_news_data(ticker, start_date, end_date):
    """Placeholder news sentiment data."""
    dates = pd.date_range(start_date, end_date)
    news_data = pd.DataFrame({
        'Date': dates,
        'Sentiment': np.random.randn(len(dates))
    })
    return news_data

def safe_merge(stock_data, news_data, key='Date'):
    """Ensure both DataFrames have key as a column before merging."""
    stock_data = stock_data.reset_index(drop=False)
    news_data = news_data.reset_index(drop=False)
    return pd.merge(stock_data, news_data, on=key, how='left')

def preprocess_data(df):
    """Add % Change column based on Adj Close or Close."""
    col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    df['% Change'] = df[col].pct_change()
    df.dropna(inplace=True)
    return df

def create_sequences(data, seq_length):
    """Return X, y sequences for LSTM."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length-1])
        y.append(data[i+seq_length-1])
    return np.array(X), np.array(y)

def build_model(input_shape):
    """Build and compile LSTM model."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# -----------------------------
# Streamlit UI
# -----------------------------

st.sidebar.title("Configuration")
tickers = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN']
ticker = st.sidebar.selectbox('Select Ticker', tickers, index=0)
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2023-12-31'))
seq_length = st.sidebar.slider("Sequence Length", min_value=10, max_value=100, value=50)

st.title("Stock Price Prediction with News Sentiment")

# Load data
stock_data = prepare_stock_data(ticker, start_date, end_date)
news_data = load_news_data(ticker, start_date, end_date)

if not stock_data.empty:
    # Merge safely
    merged_data = safe_merge(stock_data, news_data)

    # Preprocess
    merged_data = preprocess_data(merged_data)
    price_col = 'Adj Close' if 'Adj Close' in merged_data.columns else 'Close'

    # Moving averages plot
    st.header('Moving Averages Indicator')
    fig_ma, ax_ma = plt.subplots()
    ax_ma.plot(merged_data['Date'], merged_data[price_col], label='Actual Price')
    ax_ma.plot(merged_data['Date'], merged_data[price_col].rolling(window=100).mean(), label='100-day MA')
    ax_ma.plot(merged_data['Date'], merged_data[price_col].rolling(window=200).mean(), label='200-day MA')
    ax_ma.set_title('100-day and 200-day Moving Averages')
    ax_ma.set_xlabel('Date')
    ax_ma.set_ylabel('Price')
    ax_ma.legend()
    st.pyplot(fig_ma)

    # Scale and sequence
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(merged_data[price_col]).reshape(-1, 1))
    X, y = create_sequences(scaled_data, seq_length)

    # Train/test split
    train_size = int(len(X) * 0.7)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Reshape for LSTM
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Build and train
    model = build_model(input_shape)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Evaluate
    loss = model.evaluate(X_test, y_test, verbose=0)
    st.write(f"Model Loss: {loss:.4f}")

    # Predict
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Plot actual vs predicted
    st.header("Actual vs Predicted Prices")
    fig, ax = plt.subplots()
    ax.plot(merged_data['Date'][train_size + seq_length:], merged_data[price_col].values[train_size + seq_length:], label='Actual Price')
    ax.plot(merged_data['Date'][train_size + seq_length:], predictions, label='Predicted Price')
    ax.set_title('Actual vs. Predicted Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)