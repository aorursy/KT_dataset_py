import numpy as np

import pandas as pd 

from keras.models import Sequential

from keras.layers import Dense

from sklearn.model_selection import train_test_split
dataset = pd.read_csv('../input/diabetes.csv')
dataset.head()
dataset.info()
outcome = dataset['Outcome']

data = dataset[dataset.columns[:8]]

train,test = train_test_split(dataset, test_size = 0.25, random_state = 0, stratify = dataset['Outcome'])

train_X = train[train.columns[:8]]

test_X = test[test.columns[:8]]

train_Y = train['Outcome']

test_Y = test['Outcome']
train_X.head(2)
gildong = Sequential()
gildong.add(Dense(12, input_dim = 8, activation = 'relu'))

gildong.add(Dense(8, activation = 'relu'))

gildong.add(Dense(1, activation = 'relu'))
gildong.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
gildong.fit(train_X, train_Y, epochs = 200, batch_size = 32, validation_data = (test_X, test_Y))
scores = gildong.evaluate(test_X, test_Y)
print("%s: %.2f%%" %(gildong.metrics_names[1], scores[1]*100))