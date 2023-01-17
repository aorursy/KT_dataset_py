import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train=pd.read_csv("../input/electricity-consumption/train.csv")

test=pd.read_csv("../input/electricity-consumption/test.csv")



print(train.head())



def preprocess(data):

    data['temperature'] = data['temperature'] - data['temperature'].min()

    data['temperature'] = data['temperature'] / data['temperature'].max()    



    data['pressure'] = data['pressure'] - data['pressure'].min()

    data['pressure'] = data['pressure'] / data['pressure'].max()    



    data['windspeed'] = data['windspeed'] - data['windspeed'].min()

    data['windspeed'] = data['windspeed'] / data['windspeed'].max()    



    data['var1'] = data['var1'] - data['var1'].min()

    data['var1'] = data['var1'] / data['var1'].max()    

    

    data['A'] = (data['var2'] == 'A').astype(int)

    data['B'] = (data['var2'] == 'B').astype(int)

    data['C'] = (data['var2'] == 'C').astype(int)

    

    data['datetime'] = pd.to_datetime(data['datetime'])



    data['year'] = data['datetime'].dt.year

    for y in set(data['year']):

        data['y' + str(y)] = (data['year'] == y).astype(int)

    

    data['weekday'] = data['datetime'].dt.weekday

    for d in set(data['weekday']):

        data['wd' + str(d)] = (data['weekday'] == d).astype(int)



    data['month'] = data['datetime'].dt.month

    for m in set(data['month']):

        data['m' + str(m)] = (data['month'] == m).astype(int)



    data['day'] = data['datetime'].dt.day

    data['dsin'] = np.sin(2 * np.pi * (data['day']-1) / 31)

    data['dcos'] = np.cos(2 * np.pi * (data['day']-1) / 31)



    data['hour'] = 60 * data['datetime'].dt.hour + data['datetime'].dt.minute

    data['hsin'] = np.sin(2 * np.pi * data['hour'] / (60*24))

    data['hcos'] = np.cos(2 * np.pi * data['hour'] / (60*24))

        

    y = data['electricity_consumption']

    X = data.drop(['electricity_consumption', 'ID', 'datetime', 'var2', 'month', 'year', 'weekday', 'hour', 'day' ], axis='columns')

    return X, y



X_train, y_train = preprocess(train)

print(X_train.head())



def scoref(model, X_test, y_test):

    p_test = model.predict(X_test)

    return (sum([ (p-y)**2 for p, y in zip(p_test, y_test) ]) / len(p_test)) ** 0.5



from sklearn.model_selection import cross_val_score

from sklearn.utils import shuffle



X_train, y_train = shuffle(X_train, y_train)
from sklearn.neighbors import KNeighborsRegressor



model = KNeighborsRegressor(n_neighbors=3)

err = cross_val_score(model, X_train, y_train, cv=5, scoring=scoref)

print('mean cv error', sum(err) / 5)

# mean cv error 35.495417212506204
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor



def neural_net():

    # create model

    model = Sequential()

    model.add(Dense(4000, input_dim=35, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1000, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model



import datetime



current_time1 = datetime.datetime.now()

model = KerasRegressor(build_fn=neural_net, epochs=500, batch_size=128, verbose=0)

err = cross_val_score(model, X_train, y_train, cv=5, scoring=scoref)

current_time2 = datetime.datetime.now()

print('time', current_time2 - current_time1, 'mean cv error', sum(err) / 5)
