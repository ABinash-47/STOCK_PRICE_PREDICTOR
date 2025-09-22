import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime
import joblib
from alpha_vantage.timeseries import TimeSeries

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

# Convert to string
start = start_date.strftime("%Y-%m-%d")
end = end_date.strftime("%Y-%m-%d")

# Alpha Vantage API key
api_key = "2P5VFXBO27RMLQIM"

# Fetch data from Alpha Vantage
try:
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta = ts.get_daily(symbol=stock, outputsize='full')
    data = data.sort_index()  # ascending order by date
    data = data.loc[start:end]  # filter by selected dates
except Exception as e:
    st.error(f"⚠️ Could not fetch data: {e}")
    st.stop()

#if data.empty:
 #   st.error("⚠️ No stock data found for the given symbol and date range. Try another input.")
  #  st.stop()

# Show dataframe
st.subheader(f"{stock} Stock Data ({start} → {end})")
st.write(data)

# --------------------
# DATA PRE_PROCESSING
# --------------------
data_train, data_test = train_test_split(data, test_size=0.2, shuffle=False)
data_train = pd.DataFrame(data_train['4. close'])  # Alpha Vantage column name
data_test = pd.DataFrame(data_test['4. close'])

# Scale
data_train_scale = scaler.fit_transform(data_train)

# Concatenate past 100 days
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

# Inverse transform
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

next_day_price = model.predict(last_100_days)
next_day_price = float(scaler.inverse_transform(next_day_price)[0][0])

last_price = float(data['4. close'].iloc[-1])  # last real price

if next_day_price > last_price:
    prob_increase = 0.7
    prob_decrease = 0.3
else:
    prob_increase = 0.3
    prob_decrease = 0.7

# Show result
st.subheader("Next Day Prediction")
st.write(f"Last Closing Price: {last_price:.2f}")
st.write(f"Predicted Next Day Price: {next_day_price:.2f}")
st.write(f"Probability of Increase: {prob_increase*100:.2f}%")
st.write(f"Probability of Decrease: {prob_decrease*100:.2f}%")















