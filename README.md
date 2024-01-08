# esn-for-beijing-air-pollution
An echo state network model to predict the air pollution in Beijing.
I wanted to use an ESN to predict the air pollution in Beijing and to compare the results with the ones of an LSTM. I used PyRCN library to build my ESN model. My best RMSE with this model was 25.5611, whereas with an LSTM I've had less than 23. Maybe it's due to the fact that this time series data is not so chaotic to be best predicted by a reservoir computing model:-).
