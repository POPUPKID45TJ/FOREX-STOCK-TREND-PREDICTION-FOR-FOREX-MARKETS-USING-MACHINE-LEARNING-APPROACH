import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews

# --- Defaults ---
default_ticker = "TSLA"
default_start_date = pd.to_datetime("2020-01-01")
default_end_date = pd.Timestamp.now().date()

st.title('📈 Stock Dashboard')

# --- Sidebar ---
tickers = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN']
ticker = st.sidebar.selectbox('Select Ticker', tickers, index=0)
start_date = st.sidebar.date_input('Start Date', default_start_date)
end_date = st.sidebar.date_input('End Date', default_end_date)

# --- Download data ---
data = yf.download(ticker, start=start_date, end=end_date)

# Safely choose column
y_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
y_series = data[y_col]
if isinstance(y_series, pd.DataFrame):
    y_series = y_series.squeeze()

# --- Chart ---
fig = px.line(data, x=data.index, y=y_series, title=f"{ticker} Price History")
st.plotly_chart(fig)

# --- Tabs ---
pricing_data, fundamental_data, news = st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News"])

# --- Pricing Data ---
with pricing_data:
    st.header('Price Movements')
    data2 = data.copy()
    data2['% Change'] = y_series / y_series.shift(1) - 1
    data2.dropna(inplace=True)
    st.write(data2)

    annual_return = data2['% Change'].mean() * 252 * 100
    st.write('Annual Return is ', round(annual_return, 2), '%')

# --- Fundamental Data ---
with fundamental_data:
    st.header("Fundamentals")

    # Create sub-tabs
    av_tab, yf_tab = st.tabs(["Alpha Vantage", "Yahoo Finance"])

    # --- Alpha Vantage ---
    with av_tab:
        key = '9C2OWRXPCKQHEBVS'
        fd = FundamentalData(key, output_format='pandas')

        def safe_fd_call(func, ticker, label):
            try:
                data = func(ticker)
                if data and len(data) > 0:
                    df = data[0].T[2:]
                    df.columns = list(data[0].T.iloc[0])
                    st.subheader(label)
                    st.write(df)
                else:
                    st.warning(f"No {label} data returned for {ticker} (Alpha Vantage)")
            except Exception as e:
                st.error(f"Error fetching {label}: {e}")

        safe_fd_call(fd.get_balance_sheet_annual, ticker, "Balance Sheet")
        safe_fd_call(fd.get_income_statement_annual, ticker, "Income Statement")
        safe_fd_call(fd.get_cash_flow_annual, ticker, "Cash Flow Statement")

    # --- Yahoo Finance ---
    with yf_tab:
        st.subheader("Yahoo Finance Fundamentals")
        yf_ticker = yf.Ticker(ticker)

        try:
            st.write("Balance Sheet")
            st.write(yf_ticker.balance_sheet)
        except Exception as e:
            st.warning(f"Yahoo Finance balance sheet unavailable: {e}")

        try:
            st.write("Income Statement")
            st.write(yf_ticker.financials)
        except Exception as e:
            st.warning(f"Yahoo Finance income statement unavailable: {e}")

        try:
            st.write("Cash Flow")
            st.write(yf_ticker.cashflow)
        except Exception as e:
            st.warning(f"Yahoo Finance cash flow unavailable: {e}")

        # --- Extra: Key Ratios Snapshot ---
        try:
            st.write("📊 Key Ratios & Info")
            info = yf_ticker.info
            st.write({
                "Market Cap": info.get("marketCap"),
                "PE Ratio": info.get("trailingPE"),
                "EPS": info.get("trailingEps"),
                "Dividend Yield": info.get("dividendYield"),
                "52 Week High": info.get("fiftyTwoWeekHigh"),
                "52 Week Low": info.get("fiftyTwoWeekLow")
            })
        except Exception as e:
            st.warning(f"Yahoo Finance ratios unavailable: {e}")

# --- News ---
with news:
    st.header(f'📰 News for {ticker}')
    sn = StockNews(ticker, save_news=False)
    df_news = sn.read_rss()

    for i in range(min(10, len(df_news))):
        st.subheader(f'News {i + 1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        st.write(f"Title Sentiment: {df_news['sentiment_title'][i]}")
        st.write(f"Summary Sentiment: {df_news['sentiment_summary'][i]}")