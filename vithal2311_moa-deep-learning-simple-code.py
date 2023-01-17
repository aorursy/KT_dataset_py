# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Loading the required libraries

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow import keras

from keras.layers import Dense # Dense layers are "fully connected" layers

from keras.models import Sequential # Documentation: https://keras.io/models/sequential/

from keras.layers import  Flatten

from keras.utils.np_utils import to_categorical

from keras.optimizers import SGD

from keras.callbacks import EarlyStopping





























# Load the data

df = pd.read_csv('../input/lish-moa/train_features.csv')

dt = pd.read_csv('../input/lish-moa/test_features.csv')

target = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
# SOME Feature engineering, Replacing the categorical variable to numerical

df['cp_type'].replace(['ctl_vehicle','trt_cp'],[0,1],inplace=True)

df['cp_dose'].replace(['D1','D2'],[0,1],inplace=True)



dt['cp_type'].replace(['ctl_vehicle','trt_cp'],[0,1],inplace=True)

dt['cp_dose'].replace(['D1','D2'],[0,1],inplace=True)
dtrain=df.drop(['sig_id'],axis = 1)

x_train = dtrain.values

#print(x_train)

dtest=dt.drop(['sig_id'],axis = 1)

x_test = dtest.values

target_train = target.drop(['sig_id'],axis = 1)

y_train= target_train.values
#Creating simple deep learing model



n_cols = x_train.shape[1]

model = Sequential()



# Add the first hidden layer

model.add(Dense(1400, activation='relu',input_shape = (n_cols,)))





# Add the second hidden layer

model.add(Dense(588, activation='relu'))

model.add(Dense(470, activation='relu'))

#model.add(Flatten())

# Add the output layer

model.add(Dense(206,activation='sigmoid'))

#print("input shape ",model.input_shape)

#print("output shape ",model.output_shape)



# Compile the model with learnign reate 0.1 ,here early stopping is applied too

opt = keras.optimizers.Adam(lr=0.1)

model.compile(optimizer=opt, loss='binary_crossentropy' , metrics=['accuracy'])

early_stopping_monitor = EarlyStopping(patience=3)

# Fit the model

model.fit(x_train,y_train, 

    epochs=10,callbacks=[early_stopping_monitor])



pred = model.predict(x_test, verbose=0)
submission = pd.read_csv('../input/lish-moa/sample_submission.csv')



columns = list(submission.columns)

columns.remove('sig_id')



for i in range(len(columns)):

    submission[columns[i]] = pred[:, i]



submission.to_csv('submission.csv', index=False)