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
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

print(train.shape)

train.head()
print(test.shape)

test.head()
y_train=train['label']

X_train=train.drop('label',1)

X_test=test

X_train=np.asarray(X_train).reshape(42000,784)/255.0

X_test=np.asarray(X_test).reshape(28000,784)/255.0

y_train=np.asarray(y_train)
import tensorflow as tf

from tensorflow import keras
model = keras.Sequential([

    keras.layers.Dense(1568,activation=tf.nn.relu,input_shape=(784,)),    

    keras.layers.Dense(10, activation=tf.nn.softmax)

])



model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.0001),

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=30)
predictions = model.predict(X_test)

y_predictions=[]

for i in range(0,test.shape[0]):

    y_predictions.append(np.argmax(predictions[i]))



tf_submission=pd.DataFrame({'ImageId':range(1,test.shape[0]+1),'Label':y_predictions})

tf_submission.to_csv('tf_submission.csv',index=False)