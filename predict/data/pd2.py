import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Function to load data
start = '2010-01-01'
end = '2023-12-31'
default_ticker = "TSLA"
ticker = st.text_input('Ticker', default_ticker)
data = yf.download(ticker, start=start, end=end)
def load_data(ticker, start_date, end_date):
     return data
# Preprocessing data
data2 = data.copy()
data2['% Change'] = data2['Adj Close'] / data2['Adj Close'].shift(1) - 1 
data2.dropna(inplace=True)
annual_return = data2['% Change'].mean() * 252 * 100

st.header('Price Movements')
st.write(data2)
st.write('Annual Return is ', annual_return, '%')

st.subheader('Stock Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(data.Close)
st.pyplot(fig)

    # Plot 100-day and 200-day moving averages
st.header('Moving Averages Indicator')
fig_ma, ax_ma = plt.subplots()
ax_ma.plot(data['Adj Close'], label='Actual Price')
ax_ma.plot(data['Adj Close'].rolling(window=100).mean(), label='100-day MA')
ax_ma.plot(data['Adj Close'].rolling(window=200).mean(), label='200-day MA')
ax_ma.set_title('100-day and 200-day Moving Averages')
ax_ma.set_xlabel('Date')
ax_ma.set_ylabel('Price')
ax_ma.legend()
st.pyplot(fig_ma)

# Function to preprocess data
def preprocess_data(data):
    data['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1 
    data.dropna(inplace=True)
    return data

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

# Function to build LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main function
def main():
    st.title("Stock Price Prediction")
    
    # User input for stock ticker and date range
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2020-01-01'))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2023-12-31'))
    seq_length = st.sidebar.slider("Sequence Length", min_value=10, max_value=100, value=50)

    # Load and preprocess data
    data = load_data(ticker, start_date, end_date)
    data = preprocess_data(data)
    
    # Create sequences for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(data['Adj Close']).reshape(-1,1))
    sequences = create_sequences(scaled_data, seq_length)
    
    # Split data into train and test sets
    train_size = int(len(sequences) * 0.7)
    train_data = sequences[:train_size]
    test_data = sequences[train_size:]

    # Prepare input and target variables
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]
    
    # Reshape input data for LSTM
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Build and train LSTM model
    model = build_model(input_shape)
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    # Evaluate model on test data
    loss = model.evaluate(X_test, y_test)
    st.write(f"Model Loss: {loss}")

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Plot actual vs. predicted prices
    fig, ax = plt.subplots()
    ax.plot(data.index[train_size + seq_length:], data['Adj Close'].values[train_size + seq_length:], label='Actual Price')
    ax.plot(data.index[train_size + seq_length:], predictions, label='Predicted Price')
    ax.set_title('Actual vs. Predicted Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
