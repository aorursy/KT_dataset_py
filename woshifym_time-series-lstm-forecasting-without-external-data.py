import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import sys

import os

import time

from tqdm import tqdm

from tqdm.keras import TqdmCallback

from datetime import timedelta

from copy import deepcopy



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf

# import keras

# import keras.backend as K

# from keras.models import Sequential

# from keras.layers import Input, LSTM, Dense, Activation, Dropout

# from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Input, LSTM, Dense, Activation, Dropout

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# from tensorflow.compat.v1.keras.layers import CuDNNLSTM
train_path = '../input/covid19-global-forecasting-week-4/train.csv'

test_path = '../input/covid19-global-forecasting-week-4/test.csv'

sub_path = '../input/covid19-global-forecasting-week-4/submission.csv'
df = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

df.tail()
def load_csv(path):

    df = pd.read_csv(path)

    df.fillna('None', inplace=True)

    return df
# metrics



def root_mean_squared_log_error(y_true, y_pred, smooth=1):

    '''



    :param y_true: (-1,) (np ndarray) actual values

    :param y_pred: (-1,) (np ndarray) predicted values

    :return: (float) root mean squared log error, value range[0, inf)

    '''

    return np.sqrt((1 / len(y_true)) * np.sum((np.log(y_true + smooth) - np.log(y_pred + smooth)) ** 2))



def r2_score(y_true, y_pred):

    '''



    :param y_true: (-1,) (np ndarray) actual values

    :param y_pred: (-1,) (np ndarray) predicted values

    :return: (float) r2 score, value range(-inf, 1]

    '''

    sse = np.sum((y_true - y_pred) ** 2)

    var = np.sum((y_true - np.mean(y_true)) ** 2)

    if var == 0:

        r2 = 0

    else:

        r2 = 1 - sse / var

    return r2
# build LSTM model



def timesteps(data, steps):

    results = []

    for i in range(len(data) - steps):

      results.append(data[i:i+steps+1].values.tolist())

    return np.array(results)



def input_reshape(data, shape):

    return data.reshape(shape)



def slide1_window(data, value):

    data = data.reshape(-1, 2).tolist()

    new_data = data[1:]

    new_data.append(value)

    return np.array(new_data)



# def check_data(*datas):

#     results = []

#     for data in datas:

#         if isinstance(data, np.ndarray):

#           data = data.reshape(-1,)

#         elif isinstance(data, pd.core.series.Series):

#           data = data.values.reshape(-1,)

#         else:

#             data = np.array(data).reshape(-1,)

#         results.append(data.astype(np.float64))

#     return tuple(results)



def fit_lstm(train, vali, lstm_input_shape=None, batch_size=1, epochs=1, verbose=1):

    X_train, y_train = train[:, :-1], train[:, -1]

    X_train = input_reshape(X_train, lstm_input_shape)



    vali_data = (input_reshape(vali[:, :-1], lstm_input_shape), vali[:, -1])



    model = Sequential()

    model.add(LSTM(128, input_shape=(lstm_input_shape[1:])))

    model.add(Dropout(0.2))

    model.add(Dense(lstm_input_shape[-1]))

    

    early_stopping = EarlyStopping(monitor='val_loss', patience=20)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=0, factor=0.5)

    callbacks = [early_stopping, reduce_lr]

    if verbose != -1:

        callbacks.append(TqdmCallback(verbose=verbose))



    model.compile(optimizer='adam', loss='mse')



    model.fit(

      X_train,

      y_train,

      batch_size=batch_size,

      epochs=epochs,

      verbose=0,

      validation_data=vali_data,

      callbacks=callbacks

      )

    # for i in range(epochs):

    #   model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=verbose)

    #   model.reset_states()

    return model
# help functions for forecast



def load_cases(df, feat):

    cases = pd.DataFrame(df[feat].values, index=df['Date'])

    return cases



# scale train data to [0, 1] 

def scale_fit(data):

    scaler = MinMaxScaler()

    scaler.fit(scale_reshape(data))

    return scaler



def scale_transform(scaler, data):

    return scaler.transform(scale_reshape(data)).reshape(data.shape)



def scale_reshape(data):

    results = np.array(data).reshape(-1, 2)

    return results



# apply difference to the data 

class diff_node:

    def __init__(self, first=None, v=None, level=0):

        self.first = first

        self.v = v

        self.level = level

        self.prev = None

        self.next = None



def difference(X, degree):



    diff = X

    if degree == 0:

        root = diff_node(v=diff)

    else:

        root = diff_node(first=diff.iloc[0].values.reshape(-1,2))

        for i in range(degree):

            diff = diff.diff()[1:].reset_index(drop=True)

            if i + 1 >= degree:

                root.next = diff_node(v=diff, level=i+1)

            else:

                root.next = diff_node(first=diff.iloc[0].values.reshape(-1,2), level=i+1)

            root.next.prev = root

            root = root.next

    return root



def inverse_difference(node):

    results = node.v

    if node.level <= 1:

        results = np.where(results >= 0.0, results, 0.0) 

    while node.prev is not None:

        prev = node.prev

        first = prev.first

        tmp = np.cumsum(results, axis=0) + first

        results = np.concatenate((first, tmp), axis=0)

        if prev.level <= 1:

            results = np.where(results >= 0.0, results, 0.0)

        node = prev

    return np.array(results)
# construct a class for easily debug



class covid19_forecaster:

    def __init__(self, feat, Country, Province, random_state=0):

        self.scalar = None

        self.model = None

        self.model_input_shape = None

        self.history = None

        self.last_timestep = None

        self.train_date = None

        self.difference_node = None

        self.feat = feat

        self.Country = Country

        self.Province = Province

        self.random_state = random_state





    def fit(self, df, time_steps=10, diff_degrees=0, vali_size=0.2, batch_size=1, epochs=1, verbose=1

            #, fit_evaluation=False, if_plot=False

           ):

        # print('running on fit method')

        train_stepped = self._prepare_data(df, time_steps, diff_degrees)



        train_stepped_scaled = scale_transform(self.scalar, train_stepped)



        train, vali = train_test_split(train_stepped_scaled, test_size=vali_size, random_state=self.random_state)

        self.model_input_shape = -1, time_steps, len(self.feat)



        # print('starting fitting')

        #print('  Feature: %s'%(self.feat))

        # print('  Country_Region: %s, Province_State: %s'%(self.Country, self.Province))

        self.model = fit_lstm(train, vali,self.model_input_shape, batch_size, epochs, verbose)

        # print('fitting completed\n')

        self.last_timestep = train_stepped_scaled[-1, :]



#         if fit_evaluation:

#             y_true = pd.Series(self.history[time_steps:], index=self.train_date)

#             y_pred = pd.Series(

#                 self._prediction_postprocess(

#                     self.scalar,

#                     self.model.predict(input_reshape(train_stepped_scaled[:, :-1], self.model_input_shape)),

#                     self.difference_node,

#                     append=False

#                 ).reshape(-1, ),

#                 index=self.train_date

#             )

#             self.evaluation(y_true, y_pred, if_plot)





    def forecast(self, forecast_steps):

        predictions_scaled = []

        X_test, y_test = self.last_timestep[:-1], self.last_timestep[-1]

        for i in range(forecast_steps):

            X_test = slide1_window(X_test, y_test)

            y_test = self.model.predict(X_test.reshape(self.model_input_shape))[0, :] * 1.03 # multiply a compensation

            predictions_scaled.append(y_test)

        predictions_scaled = np.array(predictions_scaled)

        # predictions = prediction_inverse_transform(self.scalar, predictions_scaled)

        predictions = self._prediction_postprocess(self.scalar, predictions_scaled, self.difference_node)



        start_date = pd.to_datetime(self.train_date[-1], format='%Y-%m-%d')

        pred_dates = self._generate_dates(start_date, forecast_steps)

        return pd.DataFrame(predictions, index=pred_dates, columns=self.feat)



#     def evaluation(self, y_true, y_pred, if_plot=False):

#         self._evaluation(y_true, y_pred, if_plot)



    def _prepare_data(self, df, time_steps, diff_degrees):

        cases = load_cases(df, self.feat)

        self.history = cases



        self.difference_node = difference(cases, diff_degrees)

        difference_value = self.difference_node.v



        self.scalar = scale_fit(difference_value)

        train_stepped = timesteps(difference_value, time_steps)

        self.train_date = cases.index[time_steps:]

        return train_stepped



    def _prediction_postprocess(self, scaler, predictions, node, append=True):

        preds = scaler.inverse_transform(predictions.reshape(-1, len(self.feat)))

        preds = preds.reshape(-1, len(self.feat))



        tmp_node = deepcopy(node)

        if append:

            tmp_node.v = tmp_node.v.append(pd.DataFrame(preds), ignore_index=True)

            preds = inverse_difference(tmp_node)[-len(predictions):]

        else:

            tmp_node.v = preds

            preds = inverse_difference(tmp_node)

        return np.array(preds)



#     def _plot(self, y, h, fig_size=(6.4, 4.8), metrics=None):

#         plt.figure(figsize=fig_size)

#         y, h = self._indextodatetime(y, h)

#         y.plot(color='blue', label='actual')

#         h.plot(color='red', label='predicted')

#         plt.xlabel('Date')

#         plt.ylabel(self.feat)

#         i = 0

#         if metrics:

#             for name, metric in metrics.items():

#                 plt.text(0.25, 0.8 - i / 25, '%s: %.8f' % (name, metric), transform=plt.gca().transAxes)

#                 i += 1



#         plt.title('Country_Region: %s\nProvince_State: %s'%(self.Country, self.Province))

#         plt.legend(loc='best')

#         plt.show()



#     def _evaluation(self, y_true, y_pred, if_plot=False, plot_size=(6.4, 4.8)):

#         y, h = check_data(y_true, y_pred)

#         metrics = {}

#         metrics['r2_score'] = r2_score(y, h)

#         metrics['rmsle'] = root_mean_squared_log_error(y, h)

#         if if_plot:

#             self._plot(y_true, y_pred, plot_size, metrics)

#         else:

#             for key, value in metrics.items():

#                 print('%s: %.8f' % (key, value))



    def _generate_dates(self, start_date, length):

        dates = []

        for i in range(length):

            new_date = start_date + timedelta(days=int(i + 1))

            dates.append(new_date.strftime('%Y-%m-%d'))

        return dates



    def _indextodatetime(self, *datas):

        results = []

        for data in datas:

            dates = data.index.to_series().apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d')).values

            values = data.values

            new_data = pd.Series(values, index=dates)

            results.append(new_data)

        return tuple(results)
# the main function for forecasting



def forecast(train_path, test_path):

    df_train = load_csv(train_path)

    df_test = load_csv(test_path)

    cp = []

    for i, row in df_test.iterrows():

        v = row['Country_Region'], row['Province_State']

        if v not in cp:

            cp.append(v)



    feats = ['ConfirmedCases', 'Fatalities']

    train_start = 40 

    train_end = 84 

    time_steps = 10

    diff_degrees = 1

    vali_size = 0.15

    batch_size = 8

    epochs = 200

    verbose = -1

    forecast_steps = 30 # predictions from 2020-04-15 to 2020-05-14

    predictions = []# {feats[0]:[], feats[1]:[]}

    for Country, Province in tqdm(cp):

        df = df_train[(df_train['Country_Region']==Country)&(df_train['Province_State']==Province)]

        train = df[train_start:train_end]

        random_state = np.random.randint(0, 20, 1)[0]

        

        forecaster = covid19_forecaster(feats, Country, Province, random_state=random_state)

        forecaster.fit(train, time_steps=time_steps, diff_degrees=diff_degrees, vali_size=vali_size,

                       batch_size=batch_size, epochs=epochs, verbose=verbose)

        preds = forecaster.forecast(forecast_steps)

        preds = preds.values.tolist()



        # add data (from 04-02 to 04-14) to predictions; works for public leaderboard

        for v in train[-13:][feats].values.tolist():

            predictions.append(v)

        # add predictions (from 04-15 to 05-14); works for private leaderboard

        for p in preds:

            predictions.append(p)

    return predictions
predictions = forecast(train_path, test_path)
# generate submission file

df_sub = pd.read_csv(sub_path)

feats = ['ConfirmedCases', 'Fatalities']

df_sub[feats] = predictions

df_sub.to_csv('submission.csv', index=False)