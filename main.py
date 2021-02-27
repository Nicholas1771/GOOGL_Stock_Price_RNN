import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# data preprocessing

# import the dataset
dataset = pd.read_csv('data/GOOGL_Stock_Price_Train.csv')
training_set = dataset.iloc[:, 1:2].values

# feature scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# data structure
X_train = []
y_train = []
for i in range(120, 2314):
    X_train.append(training_set_scaled[i - 120:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# initializing the rnn
regressor = Sequential()

# add four LSTM layers
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(rate=0.2))

regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(rate=0.2))

regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(rate=0.2))

regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(rate=0.2))

regressor.add(LSTM(units=100))
regressor.add(Dropout(rate=0.2))

# add output layer, compile, and fit rnn
regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(x=X_train, y=y_train, batch_size=32, epochs=100)

# making predictions

# get real stock price of 2017 January
dataset_test = pd.read_csv('data/GOOGL_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# concatenate training and test set
dataset_total = pd.concat((dataset['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 120:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

# format the test set
X_test = []
for i in range(120, 141):
    X_test.append(inputs[i - 120:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# predict results
predictions = regressor.predict(X_test)
predictions = sc.inverse_transform(predictions)

# plot the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predictions, color='green', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock price')
plt.legend()
plt.show()
