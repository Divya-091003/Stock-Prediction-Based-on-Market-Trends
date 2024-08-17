import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model


# Input date range
start = '2010-01-01'
end = '2023-12-31'

# Streamlit UI
st.title('Stock Trend Prediction')
user_input = st.text_input('Enter stock ticker', 'TSLA')

# Download data
df = yf.download(user_input, start=start, end=end)

# Display first few rows and data summary
st.subheader('Data from 2010-2023')
st.write(df.describe())

# Plot closing price vs time
st.subheader('Closing Price vs Time chart')
fig_close = plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'])
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Closing Price vs Time')
st.pyplot(fig_close)

# Calculate and plot 100-day moving average
st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df['Close'].rolling(window=100).mean()
fig_ma100 = plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Closing price')
plt.plot(df.index, ma100, label='100-day-MA', linestyle='--', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Closing price vs Time with 100-day Moving Average')
plt.legend()
st.pyplot(fig_ma100)

# Calculate and plot 100-day and 200-day moving averages
st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma200 = df['Close'].rolling(window=200).mean()
fig_ma200 = plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Closing price')
plt.plot(df.index, ma100, label='100-day-MA', linestyle='--', color='red')
plt.plot(df.index, ma200, label='200-day-MA', linestyle='--', color='green')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Closing Price vs Time with 100-day and 200-day Moving Averages')
plt.legend()
st.pyplot(fig_ma200)



data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(data_training)


#load the model
model=load_model('keras_model.h5')

#testing part
past_100_days=data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data=scaler.fit_transform(final_df)
x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)
y_predicted=model.predict(x_test)
scaler=scaler.scale_

scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor


#final graph

st.subheader('Predictions Vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

  