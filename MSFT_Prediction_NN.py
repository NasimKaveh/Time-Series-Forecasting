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
training = dataset.iloc[:, 2:6].values
timeset = dataset.iloc[:,0].values

# Normalization
sc = MinMaxScaler(feature_range=(0,1))
training_scaled = sc.fit_transform(training)
#print(training_scaled[0 : 60])

# Feature preparation 
n = len(dataset) # number of samples (rows)
lag = 60 # number of lines to use from stock dataset in each X_set row (time step)
features = 4 # # columns
feature_len = features * lag # number of features in the X set

# Initialize matrix size
X_set = np.empty([n-feature_len, feature_len])
y_set = np.empty([n-feature_len, features])  # Multivariate (>1 output feature) time
Timset = np.empty([n-feature_len, 1])

# Allocate data into X_set and y_set, can also use the pandas shift() function
for i in range(0, n-feature_len):
    feature = training_scaled[i:i+lag, :]
    feature_expand = feature.reshape(-1,) # transpose the column into row
    X_set[i,:]= feature_expand # allocate the row into X_set by keeping the same column features
    y_set[i,:] = training_scaled[i+lag, :]
    Timset[i,] = timeset[i+lag,]
    
#print('Timeset', Timset)
#print('y_set', y_set[1])
#print(training_scaled)


# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, random_state = 0, test_size = 0.15, shuffle = True)

# Define model
# Standard Neural Network using Keras for Supervised Learning: fully connected with three layers (Dense)
model = Sequential()
model.add(Dense(12, input_dim=feature_len, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='sigmoid'))
opt= SGD(lr=0.01, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

# fit model
history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 200, verbose = 0)

yhat = model.predict(X_set, verbose=0)

#Undo th scaling
y_predict= sc.inverse_transform(yhat)
y_data = sc.inverse_transform(y_set)


'''
model.summary()

# evaluate the model and plot
train_mse = model.evaluate(X_train, y_train, verbose = 0)
test_mse = model.evaluate(X_test, y_test, verbose = 0)
print("train:", train_mse, " test:",test_mse)
plt.title('Loss/Mean Squared Error')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
'''

fig, (ax2, ax3) = plt.subplots(2)
fig.suptitle('MSFT Stock prediction NN')

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




