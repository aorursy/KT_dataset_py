# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



filename = "../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv"

df = pd.read_csv(filename)

df.head()
#seperating the data and labels

X = df.iloc[:, :12]

Y = df.iloc[:, [12]]

#import the required packages

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras
#converting the data into numpy arrays and splitting the data into train and validation set

X = np.array(X)

Y = np.array(Y)

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y)
#defining the model architecture

model = keras.models.Sequential([keras.layers.BatchNormalization(),

                               keras.layers.Dense(70, activation = "selu", kernel_initializer="lecun_normal", input_shape=X_train.shape[1:]),

                               keras.layers.Dense(70, activation = "selu", kernel_initializer = "lecun_normal", kernel_regularizer = keras.regularizers.l2(0.03)),

                               keras.layers.Dense(20, activation = "selu", kernel_initializer = "lecun_normal", kernel_regularizer = keras.regularizers.l2(0.03)),

                               keras.layers.Dense(1, activation = "sigmoid")])
#setting the learning algorithm and compiling the model

optimizer = keras.optimizers.Adam(lr = 0.005)

model.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])

model_checkpoint = keras.callbacks.ModelCheckpoint("heart_failure_model.h5")

early_stopping = keras.callbacks.EarlyStopping(patience=10)
#training the model

history = model.fit(X_train, Y_train, epochs=100, validation_data = (X_valid, Y_valid), callbacks=[model_checkpoint, early_stopping])
#plotting the results

import matplotlib.pyplot as plt



pd.DataFrame(history.history).plot(figsize=(8, 5))

plt.grid(True)

plt.gca().set_ylim(0,1)

plt.show()