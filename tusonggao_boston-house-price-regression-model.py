import os

import time

import math

import numpy as np

import pandas as pd



import tensorflow as tf

import lightgbm as lgb



import keras.backend as K

from keras.models import Sequential

from keras.layers import Dense, BatchNormalization



from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split
housing = load_boston()

X_data, y_data = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)



print('X_train.shape is ', X_train.shape,'y_train.shape is ', y_train.shape,

      'X_test.shape is ', X_test.shape,'y_test.shape is ', y_test.shape)
def batcher(X_data, y_data, batch_size=-1, random_seed=None):

    if batch_size == -1:

        batch_size = len(X_data)

    if batch_size < 1:

       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    

    if random_seed is not None:

        np.random.seed(random_seed)



    rnd_idx = np.random.permutation(len(X_data))

    n_batches = len(X_data) // batch_size

    for batch_idx in np.array_split(rnd_idx, n_batches):

        X_batch, y_batch = X_data[batch_idx], y_data[batch_idx]

        yield X_batch, y_batch



def root_mean_squared_error(y_true, y_pred):

    return K.sqrt(K.mean(K.square(y_pred - y_true)))



def create_dnn_model():

    model = Sequential()

    model.add(Dense(400, kernel_initializer='normal', activation='relu'))

    model.add(Dense(350, kernel_initializer='normal', activation='relu'))

    model.add(Dense(300, kernel_initializer='normal', activation='relu'))

    model.add(Dense(250, kernel_initializer='normal', activation='relu'))

    model.add(Dense(200, kernel_initializer='normal', activation='relu'))

    model.add(Dense(150, kernel_initializer='normal', activation='relu'))

    model.add(Dense(100, kernel_initializer='normal', activation='relu'))

    model.add(Dense(50, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss=root_mean_squared_error, optimizer='adam')

    return model



def regression_keras(X_train, y_train, X_test, y_test):

    X_merged = np.r_[X_train, X_test]

    X_merged = StandardScaler().fit_transform(X_merged)      # preprocessing data

    X_merged = np.c_[np.ones((len(X_merged), 1)), X_merged]  # add a column of 1,to learn the bias value

    X_train = X_merged[:len(X_train)]

    X_test = X_merged[len(X_train):]

    y_train = y_train.reshape(-1, 1)

    y_test = y_test.reshape(-1, 1)

    

    model = create_dnn_model()

    total_iter_num = 0

    max_epochs = 1000

    batch_size = 4

    for epoch in range(max_epochs):

        epoch_start_t = time.time()

        for batch_X, batch_y in batcher(X_train, y_train, batch_size):

            total_iter_num += 1

            model.train_on_batch(batch_X, batch_y)

            if total_iter_num % 2000 == 0:

                train_score = model.evaluate(x=X_train, y=y_train, verbose=0)

                test_score = model.evaluate(x=X_test, y=y_test, verbose=0)

                print('total_iter_num:{:7d} train_loss {:.9f} test_loss {:.9f}'.format(total_iter_num, train_score, test_score))

    rmse_test = model.evaluate(x=X_test, y=y_test, verbose=0)

    print('final rmse for test data is ', rmse_test)

    

np.random.seed(1001)

regression_keras(X_train, y_train, X_test, y_test)
def rmse_tsg(y_true, y_pred):

    return np.sqrt(np.mean(np.square(y_true - y_pred)))



def regression_lightGBM(X_train, y_train, X_test, y_test):

    print('in regression_lightGBM')

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=42)



    lgbm_param = {'n_estimators': 10000, 'n_jobs': -1, 'learning_rate': 0.008,

                  'random_state': 42, 'max_depth': 5, 'min_child_samples': 3,

                  'num_leaves': 51, 'subsample': 0.9, 'colsample_bytree': 0.9,

                  'silent': -1, 'verbose': -1}

    lgbm = lgb.LGBMRegressor(**lgbm_param)

    lgbm.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],

             eval_metric='rmse', verbose=100, early_stopping_rounds=500)



    y_test_predict = lgbm.predict(X_test)

#     rmse_test = np.sqrt(np.mean(np.square(y_test_predict, y_test)))  # final rmse for test data is  22.716238670565314

    rmse_test = rmse_tsg(y_test_predict, y_test)  # final rmse for test data is  645.829464212127

    print('final rmse for test data is ', rmse_test)



regression_lightGBM(X_train, y_train, X_test, y_test)