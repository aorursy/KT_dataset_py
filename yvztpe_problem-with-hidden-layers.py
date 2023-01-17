# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.keras.layers import  MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dropout
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
x_l = np.load('../input/Sign-language-digits-dataset/X.npy')
Y_l = np.load('../input/Sign-language-digits-dataset/Y.npy')
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0)
Y = Y.reshape(X.shape[0],1)
from sklearn.model_selection import train_test_split
X, _, Y, _ = train_test_split(X, Y, test_size=0.95, random_state=0, stratify=Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0, stratify=Y)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]
X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])#(348, 64x64)
X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])
x_train = X_train_flatten
x_test = X_test_flatten
y_train = Y_train
y_test = Y_test
from sklearn.preprocessing import OneHotEncoder
sicak=OneHotEncoder(categories='auto')
y_train=sicak.fit_transform(y_train).toarray()
sicak2=OneHotEncoder(categories='auto')
y_test=sicak2.fit_transform(y_test).toarray()
len(y_train)
model = Sequential()

model.add(Dense(2048,activation="softmax",input_dim=4096))#L1

model.add(Dense(2,activation="softmax"))
optimizer = Adam(lr=1e-3)#0.001
model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])
tarih=model.fit(x_train,y_train,epochs=200)
plt.plot(tarih.history['acc'])
plt.plot(tarih.history['loss'])
result = model.evaluate(x=x_test, y=y_test)
result#loss,accuracy
model.predict(x_test)
y_test
model = Sequential()

model.add(Dense(2048,activation="softmax",input_dim=4096))#L1
model.add(Dense(1024,activation="softmax"))#L2
model.add(Dense(512,activation="softmax"))#L2

model.add(Dense(2,activation="softmax"))
optimizer = Adam(lr=1e-3)#0.001
model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])
tarih=model.fit(x_train,y_train,epochs=200)
plt.plot(tarih.history['acc'])
plt.plot(tarih.history['loss'])
plt.plot(tarih.history['acc'])
plt.plot(tarih.history['loss'])