# pip install pyrcn

import numpy as np
import pandas as pd
from pyrcn.echo_state_network import ESNRegressor
from sklearn.preprocessing import MinMaxScaler

def root_mean_squared_error(act, pred):

   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()
   rmse_val = np.sqrt(mean_diff)
   return rmse_val
 
df = pd.read_csv('Beijing_PM.csv')
dataset = df[["pm2.5"]]
dataset.fillna(0, inplace=True)
dataset = dataset[24:]
timeseries = dataset.values.astype('float32')

scaler = MinMaxScaler(feature_range=(-1, 1))
timeseries = scaler.fit_transform(timeseries)

train_size = int(len(timeseries) * 0.67)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_sequence(seq, obs):
    x = []
    y = []
    for i in range(len(obs)-seq):
        window = obs[i:(i+seq)]
        after_window = obs[i+seq]
        window = [[x] for x in window]
        x.append(window)
        y.append(after_window)
    return np.array(x), np.array(y)
    
size_sample = 2
x_train, y_train = create_sequence(size_sample, train)
x_test, y_test = create_sequence(size_sample, test)
x_train = np.squeeze(x_train)
x_test = np.squeeze(x_test)

esn = ESNRegressor()
esn.fit(x_train, y_train)
y_pred = esn.predict(x_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

print(root_mean_squared_error(y_test, y_pred))