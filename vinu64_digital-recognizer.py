# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test =  pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

format_ = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
print(train.columns)

print('-'*100)

print(test.columns)

print('-'*100)

print(format_.columns)
X_train = train.iloc[:,1:]

y_train = pd.get_dummies(train['label'])

print(X_train.shape,y_train.shape)
y_train
from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import EarlyStopping



X_train = X_train/255

test = test/255

def get_new_model(shape):

    model = Sequential()

    model.add(Dense(200,activation = 'relu', input_shape = (shape,)))

    model.add(Dense(200, activation='relu'))

    model.add(Dense(200, activation='relu'))

    model.add(Dense(200, activation='relu'))

    model.add(Dense(200, activation='relu'))

    model.add(Dense(200, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',

                  loss='categorical_crossentropy',

                  metrics=['accuracy'])

    return model

early_stopping_monitor = EarlyStopping(patience=4)

model = get_new_model(784)

model.fit(X_train,y_train, validation_split=0.3,epochs = 30, callbacks=[early_stopping_monitor])

predictions = model.predict(test)
predictions = predictions.argmax(axis=1)
format_['Label'] = predictions

format_.to_csv('submission.csv', index = False)