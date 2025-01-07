from login_module import login_ui
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import ta
from streamlit_autorefresh import st_autorefresh
import time
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import sqlite3

import requests

import smtplib
from email.mime.text import MIMEText

# Mapping of stock tickers to names
# Ensure login session is set up
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Display login UI if not logged in
if not st.session_state.logged_in:
    login_ui()
else:
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Page 1"  # Default page

# Function to navigate between pages
    def navigate_to(page):
        st.session_state["current_page"] = page
        
 

    stock_names = {
                'AAPL': 'Apple Inc.',
                'GOOGL': 'Alphabet Inc.',
                'MSFT': 'Microsoft Corporation',
                'AMZN': 'Amazon.com Inc.',
                'TSLA': 'Tesla Inc.',
                'NFLX': 'Netflix Inc.',
                'NVDA': 'NVIDIA Corporation',
                'META': 'Meta Platforms, Inc.',
                'BABA': 'Alibaba Group Holding Ltd.',
                'ADBE': 'Adobe Inc.',
                'CRM': 'Salesforce, Inc.',
                'COKE': 'The Coca-Cola Company',
                'PEP': 'PepsiCo, Inc.',
                'SPOT': 'Spotify Technology S.A.',
                'INTC': 'Intel Corporation',
                'RELIANCE.NS': 'Reliance Industries Limited',
                'TCS.NS': 'Tata Consultancy Services',
                'TATASTEEL.NS': 'Tata Steel Ltd.',
                'OLA.NS': 'Ola Electric Mobility Pvt. Ltd.',
                'ADANIGREEN.NS': 'Adani Green Energy Ltd.',
            }

            # Fetch stock data based on the ticker, period, and interval
    def fetch_stock_data(ticker, period, interval):
                try:
                    end_date = datetime.now()
                    if period == '1wk':
                        start_date = end_date - timedelta(days=7)
                        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
                    elif period == '1d':
                        data = yf.download(ticker, period='1d', interval=interval)
                    else:
                        data = yf.download(ticker, period=period, interval=interval)
                    if data.empty:
                        st.warning(f"No data found for {ticker}. It may be delisted or there is no price data available.")
                        return None
                    return data
                except Exception as e:
                    st.error(f"Error fetching data for {ticker}: {e}")
                    return None

            # Process data to ensure it is timezone-aware and has the correct format
    def process_data(data):
        try:
                if data.index.tzinfo is None:
                    data.index = data.index.tz_localize('UTC')
                data.index = data.index.tz_convert('US/Eastern')
                data.reset_index(inplace=True)
                data.rename(columns={'Date': 'Datetime'}, inplace=True)
                return data
        except Exception as e:
            st.warning(f"Error processing data: {e}")
            return data  

            # Calculate basic metrics from the stock data
    def calculate_metrics(data):
                last_close = data['Close'].iloc[-1]
                prev_close = data['Close'].iloc[0]
                change = last_close - prev_close
                pct_change = (change / prev_close) * 100
                high = data['High'].max()
                low = data['Low'].min()
                volume = data['Volume'].sum()
                return last_close, change, pct_change, high, low, volume

            # Add simple moving average (SMA) and exponential moving average (EMA) indicators
    def add_technical_indicators(data):
                data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
                data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
                return data

            # Adding RSI and MACD
    def add_more_indicators(data):
                data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
                data['MACD'] = ta.trend.macd(data['Close'])
                data['MACD_Signal'] = ta.trend.macd_signal(data['Close'])
                return data

            # Fetch stock news using the News API
    def fetch_stock_news(ticker):
        
                url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey=8536f96646b340e4b4d4e85e1d70e2f3'
                response = requests.get(url)
                news = response.json()
                if news['status'] == 'ok' and 'articles' in news:
                    return news['articles'][:5]  # Limit to 5 latest articles
                return []

            # Set up Streamlit page layout
    def predict_stock_price(ticker, period='1mo', interval='1d'):
    # Fetch stock data for prediction
        data = fetch_stock_data(ticker, period, interval)
        if data is None:
            return None

        # Preprocess the data for model prediction (using 'Close' prices)
        data = process_data(data)
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data['Date_Ordinal'] = data['Datetime'].apply(lambda x: x.toordinal())  # Convert date to ordinal number for regression

        # Train a Linear Regression model to predict 'Close' prices
        X = data[['Date_Ordinal']]  # Independent variable (date)
        y = data['Close']  # Dependent variable (price)
        
        model = LinearRegression()
        model.fit(X, y)
        
        
        # Make prediction for the next day
        next_day_ordinal = data['Date_Ordinal'].max() + 1  # Predict for the next day
        predicted_price = model.predict([[next_day_ordinal]])[0]
        return predicted_price
    
    def send_email_alert(stock, price, user_email):
        try:
            msg = MIMEText(f"The stock {stock} has reached your target price of {price}.")
            msg['Subject'] = f"{stock} Price Alert"
            msg['From'] = 'your_email@example.com'
            msg['To'] = user_email

            # Set up the server and send the email
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login("anshul.saini1507@gmail.com", "bsyurzezqoxpmeeh")  # Use app password for Gmail
                server.sendmail('anshul.saini1507@gmail.com', user_email, msg.as_string())
            return True
        except Exception as e:
            return False

    # Function to set price alert for a stock
    def set_price_alert(ticker, alert_price, user_email):
        data = fetch_stock_data(ticker, period='1d', interval='1d')  # Fetch most recent stock data
        if data is not None:
            last_close = data['Close'].iloc[-1]
            if last_close >= alert_price:
                if send_email_alert(ticker, last_close, user_email):
                    st.success(f"Price Alert: {ticker} has reached {last_close}, alert sent to {user_email}.")
                else:
                    st.error("Failed to send email alert. Please try again.")
            else:
                st.warning(f"{ticker} has not reached the target price of {alert_price}.")
        else:
            st.error(f"Failed to fetch data for {ticker}. Please try again.")
    def page_1():
            st.set_page_config(layout="wide")
            st.title('Real Time Stock Market Dashboard')

            # Sidebar for user input parameters
            st.sidebar.header('Chart Parameters')
            ticker = st.sidebar.selectbox('Select Ticker', list(stock_names.keys()), index=0)
            time_period = st.sidebar.selectbox('Time Period', ['1d', '1wk', '1mo', '1y', 'max'])

            # Enhanced Interval Selection
            custom_intervals = ['1m', '5m', '15m', '30m', '1d', '1wk']
            interval = st.sidebar.selectbox('Data Interval', custom_intervals)
            chart_type = st.sidebar.selectbox('Chart Type', ['Candlestick', 'Line'])
            indicators = st.sidebar.multiselect('Technical Indicators', ['SMA 20', 'EMA 20', 'RSI', 'MACD'])
            
            
            
            
            if st.sidebar.button("Prediction"):
                navigate_to("Page 2") 
                





            # Sidebar option for stock comparison
            st.sidebar.subheader("Compare Stocks")
            compare_ticker = st.sidebar.selectbox("Select Ticker to Compare", list(stock_names.keys()))

            # Sidebar option for stock news

            # Auto-refresh logic (refresh every 10 seconds) and countdown
            refresh_interval = 20
            st_autorefresh(interval=refresh_interval * 1000, key="datarefresh")
            last_refresh_time = time.time()
            time_since_refresh = int(time.time() - last_refresh_time)
            countdown = refresh_interval - time_since_refresh
            st.sidebar.write(f"Auto-refresh in {countdown} seconds")

            # MAIN CONTENT AREA
            data = fetch_stock_data(ticker, time_period, interval)
            if data is not None:
                data = process_data(data)
                data = add_technical_indicators(data)
                data = add_more_indicators(data)
                last_close, change, pct_change, high, low, volume = calculate_metrics(data)

                # Display main metrics
                currency = 'INR' if ticker.endswith('.NS') else 'USD'
                st.metric(label=f"{stock_names[ticker]} Last Price", value=f"{last_close:.2f} {currency}", delta=f"{change:.2f} ({pct_change:.2f}%)")
                coll, col2, col3 = st.columns(3)
                coll.metric("High", f"{high:.2f} {currency}")
                col2.metric("Low", f"{low:.2f} {currency}")
                col3.metric("Volume", f"{volume:,}")

                # Plot the stock price chart
                fig = go.Figure()
                if chart_type == 'Candlestick':
                    fig.add_trace(go.Candlestick(
                        x=data['Datetime'],
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close']
                    ))
                else:
                    fig.add_trace(go.Scatter(x=data['Datetime'], y=data['Close'], mode='lines', name='Close Price'))

                # Add selected technical indicators to the chart
                for indicator in indicators:
                    if indicator == 'SMA 20':
                        fig.add_trace(go.Scatter(x=data['Datetime'], y=data['SMA_20'], name='SMA 20'))
                    elif indicator == 'EMA 20':
                        fig.add_trace(go.Scatter(x=data['Datetime'], y=data['EMA_20'], name='EMA 20'))
                    elif indicator == 'RSI':
                        st.subheader('RSI Indicator')
                        st.line_chart(data['RSI'])
                    elif indicator == 'MACD':
                        st.subheader('MACD Indicator')
                        st.line_chart(data[['MACD', 'MACD_Signal']])

                # Format graph
                fig.update_layout(title=f'{stock_names[ticker]} {time_period.upper()} Chart',
                                xaxis_title='Time',
                                yaxis_title=f'Price ({currency})',
                                height=680)
                st.plotly_chart(fig, use_container_width=True)

                # Display historical data and technical indicators
                st.subheader('Historical Data')
                st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']])
                #st.subheader('Technical Indicators')
                #st.dataframe(data[['Datetime', 'SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal']])

                # Download data as CSV
                st.download_button(
                    label="Download Data as CSV",
                    data=data.to_csv().encode('utf-8'),
                    file_name=f'{ticker}_data.csv',
                    mime='text/csv'
                )

            # Stock comparison chart
            compare_data = fetch_stock_data(compare_ticker, time_period, interval)
            if compare_data is not None:
                compare_data = process_data(compare_data)
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Scatter(x=data['Datetime'], y=data['Close'], name=f"{ticker} Close Price"))
                fig_compare.add_trace(go.Scatter(x=compare_data['Datetime'], y=compare_data['Close'], name=f"{compare_ticker} Close Price"))
                fig_compare.update_layout(title='Stock Comparison',
                                        xaxis_title='Time',
                                        yaxis_title='Price (USD)',
                                        height=680)
                st.plotly_chart(fig_compare)

            # Stock News Section
            
    def page_2():
            st.title("Prediction")
            
            # Navigation button to Page 1
            if st.sidebar.button("Dashboard"):
                navigate_to("Page 1")
            st.sidebar.subheader("Analyze Stocks")   
            
            #New Changes made here
            st.sidebar.subheader("Select Stock for Prediction")
            predict_ticker = st.sidebar.selectbox("Select Ticker", list(stock_names.keys()))
            
            # Call the prediction function and display the result
            predicted_price = predict_stock_price(predict_ticker)

            
            if predicted_price is not None:
                #st.subheader(f"Predicted Price for {stock_names[predict_ticker]}")
                

                st.write(f"# The predicted closing price for {stock_names[predict_ticker]} on the next day is: {predicted_price:.2f}")

            else:
                st.write(f"Could not fetch data for {stock_names[predict_ticker]}. Please try again later.") 
            
            if "email" in st.session_state:
                user_email = st.session_state.email

            # Sidebar for setting price alert
            st.sidebar.subheader("Set Price Alert")
            #alert_ticker = st.sidebar.selectbox("Select Stock for Alert", list(stock_names.keys()))
            alert_price = st.sidebar.number_input("Enter Target Price", min_value=0.0, value=1000.0, step=0.1)

            if st.sidebar.button("Set Alert"):
                set_price_alert(predict_ticker, alert_price, user_email)
                
            show_news = st.sidebar.checkbox("Show News Related to Selected Stock")
            
            if show_news:
                st.subheader(f"Latest News for {stock_names[predict_ticker]}")
                articles = fetch_stock_news(predict_ticker)
                if articles:
                    for article in articles:
                        st.write(f"**{article['title']}**")
                        st.write(article['description'])
                        st.write(f"[Read more]({article['url']})")
                else:
                    st.write("No news available for this stock at the moment.")



   

        # Additional sidebar content (if needed)
    if st.session_state["current_page"] == "Page 1":
        page_1()
    elif st.session_state["current_page"] == "Page 2":
        page_2()
        
    


