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
import pandas as pd

Dig_MNIST = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")

sample_submission = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

test = pd.read_csv("../input/Kannada-MNIST/test.csv")

train = pd.read_csv("../input/Kannada-MNIST/train.csv")
import tensorflow as tf

from keras.models import Sequential

from keras.layers import Flatten,Conv2D,MaxPooling2D

from keras.layers import Dense



mymodel = Sequential()

#conv layers



mymodel.add(Conv2D(32,(3,3),input_shape = (28,28,1), activation = 'relu'))

mymodel.add(MaxPooling2D(pool_size=(2, 2)))



mymodel.add(Conv2D(32,(3,3), activation = 'relu'))

mymodel.add(MaxPooling2D(pool_size=(2, 2)))



mymodel.add(Conv2D(32,(3,3), activation = 'relu'))

mymodel.add(MaxPooling2D(pool_size=(2, 2)))









mymodel.add(Flatten())

mymodel.add(Dense(64,activation='relu'))

mymodel.add(Dense(units=10,activation='softmax'))



mymodel.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
mymodel.summary()
fultr=train.merge(Dig_MNIST,how='outer')

fux=fultr.drop('label',1).values/255.0

fuxr=fux.reshape(-1,28,28,1)

fuy=fultr['label'].values
mymodel.fit(fuxr,fuy,epochs=10)
mymodel.evaluate(fuxr,fuy)
labels=mymodel.predict((test.drop('id',1).values/255.0).reshape(-1,28,28,1))

label=labels.argmax(1)

label
sample_submission['label']=label

sample_submission.to_csv('submission.csv',index=False)
sample_submission