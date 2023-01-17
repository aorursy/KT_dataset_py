### Using multivariate multi-step LSTM Encoder Decoder method as used below, all credits to Jason Brownlee

### https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/

### Data is 96 timesteps every day (15 min traffic count samples)

from math import sqrt

from numpy import split

from numpy import array

from numpy import ndarray

from pandas import read_csv

from sklearn.metrics import mean_squared_error

from matplotlib import pyplot

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

#from keras.layers import LSTM

from keras.layers import CuDNNLSTM

from keras.layers import RepeatVector

from keras.layers import TimeDistributed
### split a univariate dataset into train/test sets

def split_dataset(data):

    # split into days

    train, test = data[0:139584], data[139584:]

    ### restructure into windows of daily data

    train = array(split(train, len(train)/96))

    test = array(split(test, len(test)/96))

    return train, test



### evaluate one or more daily forecasts against expected values

def evaluate_forecasts(actual, predicted):

    scores = list()

    ### calculate an RMSE score for each sample

    for i in range(actual.shape[1]):

        # calculate mse

        mse = mean_squared_error(actual[:, i], predicted[:, i])

        ### calculate rmse

        rmse = sqrt(mse)

        ### store

        scores.append(rmse)

    ### calculate overall RMSE

    s = 0

    for row in range(actual.shape[0]):

        for col in range(actual.shape[1]):

            s += (actual[row, col] - predicted[row, col])**2

    score = sqrt(s / (actual.shape[0] * actual.shape[1]))

    return score, scores



### summarize scores

def summarize_scores(name, score, scores):

    s_scores = ', '.join(['%.1f' % s for s in scores])

    print('%s: [%.3f] %s' % (name, score, s_scores))



### convert history into inputs and outputs

def to_supervised(train, n_input, n_out=96):

    # flatten data

    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))

    X, y = list(), list()

    in_start = 0

    ### step over the entire history one day at a time

    for _ in range(len(data)):

        # define the end of the input sequence

        in_end = in_start + n_input

        out_end = in_end + n_out

        ### ensure we have enough data for this instance

        if out_end < len(data):

            x_input = data[in_start:in_end, 0]

            x_input = x_input.reshape((len(x_input), 1))

            X.append(x_input)

            y.append(data[in_end:out_end, 0])

        ### move along one time step

        in_start += 1

    return array(X), array(y)



### train the model, batch size and number of epochs can be played around with!

def build_model(train, n_input):

    # prepare data

    train_x, train_y = to_supervised(train, n_input)

    ### define parameters

    verbose, epochs, batch_size = 2, 10, 128

    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    ### reshape output into [samples, timesteps, features]

    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

    ### define model

    model = Sequential()

    model.add(CuDNNLSTM(400, input_shape=(n_timesteps, n_features)))

    model.add(RepeatVector(n_outputs))

    model.add(CuDNNLSTM(600, return_sequences=True))

    model.add(CuDNNLSTM(400, return_sequences=True))

    model.add(TimeDistributed(Dense(200, activation='relu')))

    model.add(TimeDistributed(Dense(1)))

    model.compile(loss='mse', optimizer='adam')

    ### fit network

    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    return model



### make a forecast

def forecast(model, history, n_input):

    # flatten data

    data = array(history)

    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))

    ### retrieve last observations for input data

    input_x = data[-n_input:, :]

    ### reshape into [1, n_input, n]

    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))

    ### forecast the next week

    yhat = model.predict(input_x, verbose=0)

    ### we only want the vector forecast

    yhat = yhat[0]

    return yhat
dataset = read_csv('../input/DataFinal.csv', header=0, infer_datetime_format=True, parse_dates=['Index'], index_col=['Index'])

# split into train and test

train, test = split_dataset(dataset.values)

n_input = 96
model = build_model(train, n_input)
history = [x for x in train]

### walk-forward validation over each week

predictions = list()

for i in range(len(test)):

    # predict the week

    yhat_sequence = forecast(model, history, n_input)

    ### store the predictions

    predictions.append(yhat_sequence)

    ### get real observation and add to history for predicting the next week

    history.append(test[i, :])

### evaluate predictions days for each week

#predictions = scalar.inverse_transform(predictions[:,:,0])

predictions = array(predictions)
score, scores = evaluate_forecasts(test[:, :, 0], predictions)

summarize_scores('lstm', score, scores)
from pylab import rcParams

rcParams['figure.figsize'] = 24, 6

# plot scores (per sample)

days = list(range(0,96))

pyplot.plot(days, scores, marker='o', label='lstm')

pyplot.show()

# Morning hours seems to be hardest to forecast accurately. This could be due to factors like seasons, day light savings, events etc.
# Plotting the 7th day ahead in time, the large variation here on this day could be due to it being 31st Dec 2017, and a Sunday! :p

pyplot.plot(range(0,96), dataset[140160:], marker='o', label='test')

pyplot.plot(range(0,96), predictions[-1,:,:], marker='o', label='test')

pyplot.show()
### Plotting a full week prediction(orange) compared to the actual week's traffic data(blue)

pyplot.plot(range(0,672), dataset[139584:], marker='o', label='test')

pyplot.plot(range(0,672), ndarray.flatten(predictions[:,:,:]), marker='o', label='test')

pyplot.show()