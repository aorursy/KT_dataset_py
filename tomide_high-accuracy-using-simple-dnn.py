import pandas as pd
import missingno as msn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

data = pd.read_csv('../input/train.csv', nrows=2000000)
data.head()
new_data = data.dropna(axis = 1)
x = new_data.drop(['is_attributed', 'click_time'], axis = 1)
print (x.head())
y = new_data['is_attributed']
print (y.head())
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=101)
scaler = MinMaxScaler()
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)
model_1 = Sequential()
model_1.add(Dropout(0.2, input_shape=(5,)))
model_1.add(Dense(256, activation='relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(128, activation='relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(64, activation='relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(32, activation='relu'))
model_1.add(Dropout(0.2))
model_1.add(Dense(units = 1, activation='sigmoid'))
model_1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
model_1.fit(scaled_x_train, y_train, epochs=5, validation_data=(scaled_x_test, y_test), verbose = 2)
