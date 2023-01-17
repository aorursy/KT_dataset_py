import numpy as np

import pandas as pd

import math

from keras.models import Sequential

from keras.layers import Dense
data=pd.read_csv("../input/foresta/forest_data.csv")

datatest = pd.read_csv("../input/florestateste/forest_data_teste.csv")

x_train = data.drop('Label', 1)

y_train = pd.get_dummies(data['Label'])



x_test = datatest.drop('Label', 1)

y_test = pd.get_dummies(datatest['Label'])
model = Sequential()

model.add(Dense(32, input_dim=5, activation='relu'))

model.add(Dense(16, activation='relu'))

model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(x_train, y_train, epochs=200, batch_size=30)
scores = model.evaluate(x_test, y_test)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))