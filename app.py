import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime
import pandas_datareader.data as web
import joblib   # needed for loading scaler

# Load the trained model & scaler
model = load_model("StockPredictionModel.keras", compile=False)
scaler = joblib.load("scaler.save")

# Streamlit app
st.header('Stock Market Predictor:')

# User input for stock symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')

# Dates
start_date = st.date_input("Enter start date", datetime.date(2010, 1, 1))
end_date = st.date_input("Enter end date", datetime.date(2023, 12, 31))

# Convert to string for DataReader
start = start_date.strftime("%Y-%m-%d")
end = end_date.strftime("%Y-%m-%d")

# Download stock data from Stooq
# For US stocks like Google, Stooq requires ".US"
try:
    data = web.DataReader(f"{stock}.US", "stooq", start, end)
    data = data.sort_index()
except Exception as e:
    st.error(f"⚠️ Could not fetch data: {e}")
    st.stop()

if data.empty:
    st.error("⚠️ No stock data found for the given symbol and date range. Try another input.")
    st.stop()

# Show dataframe in app
st.subheader(f"{stock} Stock Data ({start} → {end})")
st.write(data)

# --------------------
# DATA PRE_PROCESSING
# --------------------
data_train, data_test = train_test_split(data, test_size=0.2, shuffle=False)
data_train = pd.DataFrame(data_train.Close)
data_test = pd.DataFrame(data_test.Close)

# Scale
data_train_scale = scaler.fit_transform(data_train)

# Concatenating past 100 days data to test
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test])
data_test_scale = scaler.transform(data_test)

# Preparing x_test and y_test
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# Predictions
y_predict = model.predict(x)

# Inverse transform (get back original prices)
y_predict = scaler.inverse_transform(y_predict)
y = scaler.inverse_transform(y.reshape(-1, 1))

# Final Graph: Predicted vs Original
st.subheader('Predicted Price vs Original Price')

dates = data_test.index[100:]

fig2 = plt.figure(figsize=(10, 6))
plt.plot(dates, y, 'g', label='Original Price')
plt.plot(dates, y_predict, 'r', label='Predicted Price')
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.xticks(rotation=45)
st.pyplot(fig2)

# ------------------------
# Next Day Price Prediction
# ------------------------
last_100_days = data_test_scale[-100:]
last_100_days = np.array(last_100_days).reshape(1, 100, 1)

# Predict the next day's price
next_day_price = model.predict(last_100_days)
next_day_price = float(scaler.inverse_transform(next_day_price)[0][0])  # ensure scalar

# Get the actual last price from data
last_price = float(data.Close.iloc[-1])

# Simple probability assignment
if next_day_price > last_price:
    prob_increase = 0.7
    prob_decrease = 0.3
else:
    prob_increase = 0.3
    prob_decrease = 0.7

# Show result in Streamlit
st.subheader("Next Day Prediction")
st.write(f"Last Closing Price: {last_price:.2f}")
st.write(f"Predicted Next Day Price: {next_day_price:.2f}")
st.write(f"Probability of Increase: {prob_increase*100:.2f}%")
st.write(f"Probability of Decrease: {prob_decrease*100:.2f}%")













