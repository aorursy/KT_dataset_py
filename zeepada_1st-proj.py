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
df = pd.read_csv('../input/wine_dataset.csv')

df.head()
col=df.columns

col[:-1]

x_data=df[col[:-1]]

y_data=df[col[-1]]

x_data.head()

y_data.head() 

# 나중에 shuffle 해줄 것!
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2,random_state = 42)



print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
from keras.models import Sequential
model=Sequential()

from keras.layers import Dense
model.add(Dense(64, activation='relu', input_shape=(12,)))

model.add(Dense(32, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(2, activation='softmax'))
model.summary()
from keras import backend as K



def recall(y_target, y_pred):

    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다

    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다



    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    count_true_positive_false_negative = K.sum(y_target_yn)



    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())



    return recall





def precision(y_target, y_pred):

    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다



    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    count_true_positive_false_positive = K.sum(y_pred_yn)



    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())



    return precision





def f1score(y_target, y_pred):

    _recall = recall(y_target, y_pred)

    _precision = precision(y_target, y_pred)

    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다

    _f1score = ( 2 * _recall * _precision) / (_recall + _precision+ K.epsilon())

    

    return _f1score

from keras.optimizers import Adam



model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=[f1score]) # imbalance data => F1 score 사용

       

y_train=pd.get_dummies(y_train) # one hot 인코딩



model.fit(x=x_train, y=y_train, epochs=40, batch_size=64, shuffle=True)
import keras



y_test = pd.get_dummies(y_test) # one hot 인코딩



loss_and_metrics = model.evaluate(x=x_test, y=y_test, batch_size=64, verbose=0)

print("test셋 결과")

print(loss_and_metrics) # 위에 metrics f1score 쓰는 것?