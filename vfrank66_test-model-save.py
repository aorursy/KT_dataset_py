# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
import tensorflow as tf
set_random_seed(42)
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
import seaborn as sns
from matplotlib import pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def load_data():
    TRAIN_DATA = "../input/ch2_train.csv"
    df = pd.read_csv(TRAIN_DATA)
    y = df.iloc[:, 10]
    X = df.iloc[:, 0:10]
    print(y[0:5])
    print(X[0:5])
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
    X_val, X_test, y_val, y_test = train_test_split(X,y, test_size = 0.5)
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_network(input_features=None):
    # first we specify an input layer, with a shape == features
    inputs = Input(shape=(input_features,), name="input")
    
    # One or more layers should go here !!!!
    x = Dense(32, activation='relu', name="hidden1")(inputs)
    x = Dense(32, activation='relu', name="hidden2")(x)
    #x = Dense(32, activation='relu', name="hidden3")(x)
    # for regression we will use a single neuron with linear (no) activation
    prediction = Dense(1, activation='linear', name="final")(x)

    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
    return model
X_train, y_train, X_val, y_val, X_test, y_test = load_data()
input_features = X_train.shape[1]
model = build_network(input_features=input_features)
# Fit your MLP
# You may choose to adjust the number of epochs, batch size, or model to get a better result.
model.fit(x=X_train.values, y=y_train, batch_size=32, epochs=50, verbose=1, validation_data=(X_val.values, y_val))
# summarize history for loss
plt.figure(figsize=(10,4))
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.yscale('log')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.clf()
model.save("ch2_model.h5")