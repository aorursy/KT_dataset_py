# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.optimizers import SGD

import numpy as np



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

sub=pd.read_csv('../input/sample_submission.csv')
if train.isna().sum().all()==0:

  print("yes")
if test.isna().sum().all()==0:

  print("yes")
x=train.drop(['label'],axis=1)

x_test=test.copy()
y=train['label']
x.shape

y.shape
y=pd.Categorical(y)
y=pd.get_dummies(y)
y.head()
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
x_val.shape
y_train.shape
y_val.shape
x_train=x_train.values

x_val=x_val.values

y_train=y_train.values

y_val=y_val.values

x_test=x_test.values
scale=np.max(x_train)

x_train=x_train/scale

x_val=x_val/scale

x_test=x_test/scale
mean=np.mean(x_train)

x_train=x_train-mean

x_val=x_val-mean

x_test=x_test-mean
model=Sequential()

model.add(Dense(128,activation='relu',input_dim=784))

model.add(Dropout(0.15))

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.15))

model.add(Dense(10, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',

              optimizer=sgd,

              metrics=['accuracy'])
model.fit(x_train, y_train,

          epochs=25,

          batch_size=16)

score = model.evaluate(x_val, y_val, batch_size=16)
score
pred=model.predict(x_test,verbose=0)

new_pred = [np.argmax(y, axis=None, out=None) for y in pred]

output=pd.DataFrame({'ImageId':sub['ImageId'],'Label':new_pred})

output.to_csv('Digit_recognizer.csv', index=False)