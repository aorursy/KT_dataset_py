# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import numpy as np

import statsmodels.api as sm

import warnings

warnings.filterwarnings('ignore')



# Function for representing data in a linear form

def split_ap(values, period):

    l = list(values)

    split_list = lambda n: zip(*[iter(l + [None] * ((n - len(l) % n) % n))] * n)

    l = list(split_list(period))

    ap = []

    for i in l:

        ap.append(sum(i))

    temp = [0]

    for i in ap:

        b = temp[-1] + i

        temp.append(b)

    return temp[1:]



# Loadind the dataset

predata = pd.read_csv('../input/corn_OHLC2013-2017.txt', delimiter=',', header=0, names=['date', 'open', 'high', 'low', 'close'])

predata.index = pd.bdate_range(start=predata.date.values[0], periods=len(predata), freq='7D')

predata.drop('date', axis=1, inplace=True)



data = pd.DataFrame()

data['open'] = split_ap(predata.open.values,1)

data['high'] = split_ap(predata.high.values,1)

data['low'] = split_ap(predata.low.values,1)

data['close'] = split_ap(predata.high.values,1)

data.index = predata.index



# parameter selection for SARIMA

ps = range(0, 3)

d = [1,2]

qs = range(0, 2)

Ps = range(0, 2)

D = [1,2]

Qs = range(0, 3)



from itertools import product



parameters = product(ps, d, qs, Ps, D, Qs)

parameters_list = list(parameters)

len(parameters_list)



period = 53



best_mape = float("inf")



for param in tqdm(parameters_list):



    # try except is needed, because on some sets of parameters the model is not trained

    try:

        model = sm.tsa.statespace.SARIMAX(data.open[:-period], order=(param[0], param[1], param[2]),

                                          seasonal_order=(param[3], param[4], param[5], 12)).fit(disp=-1)

    

    except ValueError:

        continue

    except np.linalg.linalg.LinAlgError:

        continue

        

    forecast = model.forecast(steps=period)

    y_true, y_pred = data.values[-1], forecast.values[-1]

    mape = round((np.mean(np.abs((y_true - y_pred) / y_true)) * 100), 2)

    # save the best model, aic, parameters

    if mape < best_mape:

        best_model = model

        best_mape = mape

        best_param = param

    



warnings.filterwarnings('default')



# make a forecast, visualize the result

forecast = best_model.get_prediction(start=pd.to_datetime(data.index.values[0]), end=pd.to_datetime(data.index.values[-1]), dynamic=False)

forecast = forecast.predicted_mean

predict = pd.DataFrame()

predict['Real Open'] = predata.open

predict["Predict Open"] = forecast - forecast.shift(1)

predict.dropna(inplace=True)



ax = predict.plot(title='SARIMA. Price Predict')

ax.set_xlabel('Date')

ax.set_ylabel('Price, $')