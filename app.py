import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

start = '2011-01-01'
end = '2024-01-05'
st.title('Stock Price Prediction')
user_input = st.text_input('Enter your Stock Ticker', 'AAPL')

df = yf.download(user_input, start=start, end=end)
st.subheader('Data from 2011 to 2023')  # Corrected line
st.write(df.describe())


st.subheader('Closing price vs Time chart with 100 & 200 days moving average')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 8))
background_color = '#e0ffff'
plt.axhspan(min(df.Close), max(df.Close), facecolor=background_color)
plt.plot(df.Close, label='Closing Price', c='#002366')
plt.plot(ma100, label='100 days MA', c='#ffa500')
plt.plot(ma200, label='200 days MA', c='#2f847c')
plt.legend(loc='upper left')
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.7):int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Load the saved model
model = tf.keras.models.load_model('keras_model.h5')

# Prepare the data for forecasting
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predict future stock prices for the next 100 data points
y_predicted = model.predict(x_test)
scale_factor = 1/scaler.scale_[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

# ...

# Create a DataFrame with the correct index
df_forecast = pd.DataFrame({'Actual Prices': y_test, 'Forecasted Prices': y_predicted.flatten()}, index=final_df.index[-len(y_test):])

# Plot the actual and forecasted prices with specified colors
fig, ax = plt.subplots(figsize=(12, 8))
df_forecast.plot(ax=ax, color=['navy', 'red'])
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Actual vs Forecasted Prices')
plt.legend(loc='upper left')
st.pyplot(fig)


