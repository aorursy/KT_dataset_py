import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import ensemble

import random



%matplotlib inline



plt.style.use('bmh')
df_final = pd.read_csv("../input/ctrain01/CTrain01.csv")

print(f"df_final : {df_final.shape}")



df_train = df_final[df_final.SalePrice.notna()]

df_test = df_final[df_final.SalePrice.isna()]

print(f"df_train : {df_train.shape}")

print(f"df_test : {df_test.shape}")
non_features = ['Id', 'SalePrice']



xkeys = [key for key in df_final.keys() if key not in non_features]



X = df_train[xkeys].values

y = df_train['SalePrice'].values



print(f"X : {X.shape}")

print(f"y : {y.shape}")




X = df_train[xkeys].values

y = df_train['SalePrice'].values



print("X    ", X.shape)

print("y    ", y.shape)
import numpy as np

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.datasets import load_iris

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2
np.random.seed(7)

for k in [1, 5, 10, 30, 100, 200]:

    Xkb = SelectKBest(chi2, k=k).fit_transform(X, y)

    model = Sequential()

    model.add(Dense(50, input_dim=k, kernel_initializer='normal', activation='relu'))

    model.add(Dense(25, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adadelta())



    hist = model.fit(Xkb, y, epochs=100, batch_size=10, verbose=0)

    print(f"k = {k:5d} -> {hist.history['loss'][-1]/1000000:10.2f}")

np.random.seed(7)

structs = [

    ("a S", 300, [200, 100, 50, 25, 1]),

    ("a M", 400, [300, 200, 100, 50, 25, 1]),

    ("a L", 500, [400, 300, 200, 100, 50, 25, 1]),

    ("a XL", 1000, [700, 500, 300, 200, 100, 50, 25, 1]),

    ("a XXL", 5000, [1000, 700, 500, 300, 200, 100, 50, 25, 1]),

]



k = 30

Xkb = SelectKBest(chi2, k=k).fit_transform(X, y)

for name, d1, ds in structs:

    model = Sequential()

    model.add(Dense(d1, input_dim=k, kernel_initializer='normal', activation='relu'))

    for d in ds:

        model.add(Dense(d, kernel_initializer='normal', activation='relu'))

    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adadelta())



    hist = model.fit(Xkb, y, epochs=100, batch_size=10, verbose=0)

    print(f"{name:7s} -> {hist.history['loss'][-1]/1000000:10.2f}")

np.random.seed(7)

k = 30

Xkb = SelectKBest(chi2, k=k).fit_transform(X, y)

name, d1, ds = ("a XL", 1000, [700, 500, 300, 200, 100, 50, 25, 1])

for bs in [1, 5, 10, 29, 97, 730]:

    model = Sequential()

    model.add(Dense(d1, input_dim=k, kernel_initializer='normal', activation='relu'))

    for d in ds:

        model.add(Dense(d, kernel_initializer='normal', activation='relu'))

    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adadelta())



    hist = model.fit(Xkb, y, epochs=100, batch_size=bs, verbose=0)

    print(f"{bs:7d} {name:7s} -> {hist.history['loss'][-1]/1000000:10.2f}")
