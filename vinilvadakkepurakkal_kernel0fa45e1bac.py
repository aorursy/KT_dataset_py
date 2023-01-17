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
import numpy as np

import pandas as pd

import keras

from keras.layers import Dense, Dropout

from keras.optimizers import Adam

from keras.models import Sequential

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
df_train=pd.read_csv("../input/train.csv")

df_test=pd.read_csv("../input/test.csv")
df_features = df_train.iloc[:, 1:785]

df_label = df_train.iloc[:, 0]



X_test = df_test.iloc[:, 0:784]



print(X_test.shape)
X_train, X_cv, y_train, y_cv = train_test_split(df_features, df_label,test_size = 0.2,random_state = 1212)



X_train = X_train.as_matrix().reshape(33600, 784) #(33600, 784)

X_cv = X_cv.as_matrix().reshape(8400, 784) #(8400, 784)



X_test = X_test.as_matrix().reshape(28000, 784)
print(min(X_train[1]))

print(max(X_train[1]))
X_train = X_train.astype('float32')

X_cv= X_cv.astype('float32') 

X_test = X_test.astype('float32')
X_train= X_train/255 

X_cv= X_cv/255

X_test= X_test/255

y_train = keras.utils.to_categorical(y_train, 10)

y_cv = keras.utils.to_categorical(y_cv, 10)
model = Sequential()

model.add(Dense(512,activation='relu', input_shape=[784,]))
#model.add(Dropout(0.1))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(300, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(Adam(lr=0.01), loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history=model.fit(X_train,y_train, batch_size=100, epochs=20, validation_data=(X_cv,y_cv), verbose=1)
test_pred = pd.DataFrame(model.predict(X_test, batch_size=200))

test_pred = pd.DataFrame(test_pred.idxmax(axis = 1))

test_pred.index.name = 'ImageId'

test_pred = test_pred.rename(columns = {0: 'Label'}).reset_index()

test_pred['ImageId'] = test_pred['ImageId'] + 1



test_pred.head()
test_pred.to_csv('mnist_submission.csv', index = False)