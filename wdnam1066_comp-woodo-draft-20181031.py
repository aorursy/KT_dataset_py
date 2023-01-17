# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from __future__ import print_function
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
KAGGLE_DIR = '../input/'
tr_train = pd.read_csv(KAGGLE_DIR + 'train_V2.csv')
tr_test = pd.read_csv(KAGGLE_DIR + 'test_V2.csv')
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
tr_train = reduce_mem_usage(tr_train)
tr_test = reduce_mem_usage(tr_test)
tr_train = tr_train.dropna()
tr_train_sample = tr_train.sample(n=100000, random_state = 76)
tr_test_sample = tr_test.sample(n=100000, random_state = 76)
XX = tr_train_sample.drop(['Id', 'groupId', 'matchId','matchType', 'winPlacePerc'], axis = 1)
YY = tr_train_sample['winPlacePerc']
XX_test = tr_test.drop(['Id', 'groupId', 'matchId','matchType'], axis = 1)
from sklearn.preprocessing import scale
X_scaled = scale(XX)
X_test_scaled = scale(XX_test)
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

n_cols = X_scaled.shape[1]
early_stopping_monitor = EarlyStopping(patience=2)
model = Sequential()
model.add(Dense(100, activation = 'relu', input_shape = (n_cols,)))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
model.fit(X_scaled, YY, validation_split = 0.3, epochs = 20, callbacks = [early_stopping_monitor])
y_predict = model.predict(X_test_scaled).reshape(-1,1)
y_predict
predictions = pd.DataFrame(y_predict, index=tr_test["Id"]).rename(columns={0:'winPlacePerc'})
predictions.head(100)
predictions.to_csv('submission.csv')
