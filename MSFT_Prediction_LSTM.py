import pandas as pd 
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RepeatedKFold
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# Multivariate time series forcasting --> where there is more than one observation for each time step.

# Stock Data extraction
dataset = pd.read_csv('msft_Stock.csv')
timeset = dataset.iloc[:,0].values
training = dataset.iloc[:, 2:6].values

# Normalization
sc = MinMaxScaler(feature_range=(0,1))
training_scaled = sc.fit_transform(training)
#print(training_scaled[0 : 60])


# Fea
n = len(dataset) # number of samples (rows)
lag = 60 # number of lines (time steps)
features = 4
feature_len = features * lag # number of features m (column) in the input X set
X_set = np.empty([n-feature_len, lag, features])
y_set = np.empty([n-feature_len, features])  # Multivariate (>1 output feature) time
Timset = np.empty([n-feature_len, 1])

# Make dataset X_set and y_set, and timestep,
for i in range(0, n-feature_len):
    feature = training_scaled[i:i+lag, :]
    X_set[i,:,:]= feature
    y_set[i,:] = training_scaled[i+lag, :]
    Timset[i,] = timeset[i+lag,]

# Multivariate LSTM, Multiple parallel series of data  (https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/)

# define model
model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(lag, features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(features))
model.compile(optimizer='adam', loss='mse')
# fit model
history = model.fit(X_set, y_set, epochs=40, verbose=0)


# demonstrate prediction
#print(model.test_on_batch(X_set, y=y_set))


yhat = model.predict(X_set, verbose=0)

#Undo th scaling
y_predict= sc.inverse_transform(yhat)
y_data = sc.inverse_transform(y_set)

fig, (ax2, ax3) = plt.subplots(2)
fig.suptitle('MSFT Stock prediction LSTM')

'''
ax1.set_title('Open')
ax1.plot(Timset, y_data[:, 0], label='Actual')
ax1.plot(Timset, y_predict[:, 0], label='Prediction')
ax1.legend(loc='upper left')
'''

ax2.set_title('High')
ax2.plot(Timset, y_data[:, 1], label='Actual')
ax2.plot(Timset, y_predict[:, 1], label='Prediction')
ax2.legend(loc='upper left')
ax2.set_ylabel('Price ($)')

ax3.set_title('Low')
ax3.plot(Timset, y_data[:, 2], label='Actual')
ax3.plot(Timset, y_predict[:, 2], label='Prediction')
ax3.legend(loc='upper left')
ax3.set_xlabel('Time step (daily)')
ax3.set_ylabel('Price ($)')
'''
ax4.set_title('Close')
ax4.plot(Timset, y_data[:, 3], label='Actual')
ax4.plot(Timset, y_predict[:, 3], label='Prediction')
ax4.legend(loc='upper left')
'''

#plt.legend()
plt.show()



