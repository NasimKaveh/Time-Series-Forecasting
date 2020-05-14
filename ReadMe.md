# ReadMe

## Objective
The aim of this project is to forecast multivariate time series stock market data using two different methodes: Standard Neural Network(NN) and Long Short-Term Memory (LSTM).

## Reading Data from Web
pandas_datareader.data is used to extract the stock market data and datetime module is used to specify the time period. Data are then stored to local csv files. In this project: 

start = datetime(2000, 9, 1)

end = datetime(2020, 3, 10)

## Data preprocessing

The time series forecasting is first re-framed as a supervised learning problem, by transforming the sequence of data to input (X_set) and output data (y_set).

The input files are Gold and MSFT data from year 2000 to 2020 and saved respectively into csv dataset files. Data are extracted into dataframe called dataset herein using pandas read csv function and four columns are extracted as numpy arrays from csv for the multivariate time series problem: ('open', 'low', 'high', 'close'). Another variable called timestep retries the time for plotting purpose after the training. 

```
dataset = pd.read_csv('Gold_Stock.csv')
training = dataset.iloc[:, 2:6].values
timeset = dataset.iloc[:,0].values
```

Dataset is first normalized and scaled between 0 and 1 using the MinMaxScaler module from ScikitLearn.

 X_set and y_set, it was decided to select a lag of 60


In this problem, 60 lines from the main csv dataset is used 
# Feature preparation 
n = len(dataset) # number of samples (rows)
lag = 60 # number of lines to use from stock dataset in each X_set row (time step)
features = 4 # # columns
feature_len = features * lag # number of features in the X set

# Initialize matrix size
X_set = np.empty([n-feature_len, feature_len])
y_set = np.empty([n-feature_len, features])  # Multivariate (>1 output feature) time
Timset = np.empty([n-feature_len, 1])









