import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O 

import matplotlib.pyplot as plt

import re

import seaborn as sb

%matplotlib inline
# missing value processing

train = pd.read_csv('../input/train_1.csv').fillna(0)

train.head()
# extract web page language information

def get_language(page):

    res = re.search('[a-z][a-z].wikipedia.org',page)

    if res:

        return res.group()[:2]     # result fo the match converted to a str obj

    return 'na'
# add language column 

train['language'] = train['Page'].map(get_language)
# statistics of different languages

sb.countplot(train['language'])
# 　use dict cave different language DF

lang_sets = {}

lang_sets['en'] = train[train.language=='en'].iloc[:,0:-1]

lang_sets['ja'] = train[train.language=='ja'].iloc[:,0:-1]

lang_sets['de'] = train[train.language=='de'].iloc[:,0:-1]

lang_sets['na'] = train[train.language=='na'].iloc[:,0:-1]

lang_sets['fr'] = train[train.language=='fr'].iloc[:,0:-1]

lang_sets['zh'] = train[train.language=='zh'].iloc[:,0:-1]

lang_sets['ru'] = train[train.language=='ru'].iloc[:,0:-1]

lang_sets['es'] = train[train.language=='es'].iloc[:,0:-1]
# daily average pageviews for each language 

sums = {}

for key in lang_sets:

    sums[key] = lang_sets[key].iloc[:,1:].sum(axis=0) / lang_sets[key].shape[0]
# DataFrame

traffic_sum = pd.DataFrame(sums) 

traffic_sum.columns=['German','English','Spanish','French',

                     'Japanese','Nan','Russian','Chinese'] 

traffic_sum.plot(figsize=(12,6),grid=True)
# import dependent libraries

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

# Keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM
# build model

def LSTM_Model(train_x,train_y,batch_size=10,epochs=100,verbose=0):

    regressor = Sequential()

    # add LSTM layer

    regressor.add(LSTM(units= 8, activation='relu', input_shape = (None,1)))

    # add Dense layer

    regressor.add(Dense(units=1))

    # compile 

    regressor.compile(optimizer = 'rmsprop',loss='mean_squared_error')

    # fitting data

    regressor.fit(train_x, train_y, batch_size = batch_size, epochs=epochs, verbose = verbose)

    return regressor
# scaling data(daily pageviews for each language )

sc = MinMaxScaler()

scaling_sum = sc.fit_transform(traffic_sum)

scale_sum = pd.DataFrame(scaling_sum)

scale_sum.columns = traffic_sum.columns

scale_sum.head()
models = {}    # save different models

for language in scale_sum.columns:

    X = np.array(scale_sum[language])[0:549]

    Y = np.array(scale_sum[language])[1:550]

    

    # splitting dataset

    train_x, test_x , train_y, test_y = train_test_split(X, Y, test_size = 0.3, random_state=0)

    train_x = np.reshape(train_x,(384, 1, 1))

    train_y = np.reshape(train_y, (-1, 1))

    test_x = np.reshape(test_x, (165, 1, 1))

    test_y = np.reshape(test_y, (-1, 1))

    # training data

    model = LSTM_Model(train_x, train_y,10, 100)

    # save model

    models[language] = model

    # test result

    predict = model.predict(test_x)

    

    # show test result

    plt.figure(figsize=(12,6))

    plt.plot(test_y, c='r', label='Real web view')

    plt.plot(predict, c='g', label='Predicted view')

    plt.title(language + ' wiki page view forecasting')

    plt.xlabel('days')

    plt.legend()

    plt.grid(True)

    plt.show()
def plot_cruve(real, pred, page):

    plt.figure(figsize=(10,5))

    plt.plot(real, c='r', label='Real web view')

    plt.plot(pred, c='g', label='Predicted view')

    plt.title(page)

    plt.xlabel('days')

    plt.legend()

    plt.grid(True)

    plt.show()
# mapping dictionary

lang_dict={'English':'en','Chinese':'zh','German':'de','Nan':'na',

           'Japanese':'ja','French':'fr','Spanish':'es','Russian':'ru'}
for language in models.keys():

    # random number

    index = np.random.randint(10000)

    # randomly selected different language webpage test model

    data = lang_sets[lang_dict[language]].iloc[index, 1:]

    page = lang_sets[lang_dict[language]].iloc[index, 0]

    real_data = np.array(data)[1:550].reshape(-1,1)

    test_data = np.array(data)[0:549].reshape(549, 1, 1)

    # predict

    pred = models[language].predict(test_data)

    # show 

    plot_cruve(real_data, pred, page  )