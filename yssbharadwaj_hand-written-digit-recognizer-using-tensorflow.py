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
from tensorflow import keras

from keras.layers import Dense, Flatten

from keras.layers.convolutional import Conv2D

from keras.models import Sequential

from keras.utils import to_categorical

import matplotlib.pyplot as plt
train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

X_train = train_data.drop('label',axis=1)

y_train = train_data['label']

X_test = test_data
X_train = X_train/255

X_test = X_test/255
X_test.shape
# for convolution

X_train = X_train.values.reshape(X_train.shape[0],28,28,1)

X_test = X_test.values.reshape(X_test.shape[0],28,28,1)
## Checking out the shapes involved in dataset

print ("Shape of X_train: {}".format(X_train.shape))

print ("Shape of y_train: {}".format(y_train.shape))

print ("Shape of X_test: {}".format(X_test.shape))
# for categorical_crossentropy loss function

y_train = to_categorical(y_train)
plt.imshow(X_train[0][:,:,0])
# to avoid overfitting

class handwriting_acc_callback(keras.callbacks.Callback):

    def on_epoch_end(self,epoch,logs={}):

        if(logs.get('acc')>=0.98):

            print("\n Model has reached 90% accuracy! Congratulations !!!!!")

            self.model.stop_training = True
model = keras.models.Sequential([

                                 keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),

                                 keras.layers.MaxPool2D(2,2),

                                 keras.layers.Conv2D(64,(3,3),activation='relu'),

                                 keras.layers.MaxPool2D(2,2),

                                 keras.layers.Flatten(),

                                 keras.layers.Dense(128,activation='relu'),

                                 keras.layers.Dense(10,activation='softmax')

])
model.summary()
model.compile(optimizer='adam',metrics=['acc'],loss='categorical_crossentropy')
hacallback = handwriting_acc_callback()

# model.fit(X_train, y_train, epochs=50,callbacks=[hacallback]) # Using Callback

model.fit(X_train, y_train, epochs=10)
sample = X_test[9]

prediction = model.predict(sample.reshape(1, 28, 28, 1))



print ("\n\n--------- Prediction --------- \n\n")

plt.imshow(sample.reshape(28, 28), cmap="gray")

plt.title("Predicted Value:{}".format(np.argmax(prediction)))

plt.show()
# predict results

results = model.predict(X_test)

results = np.argmax(results,axis=1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,len(results)+1),name = "ImageId"),results],axis = 1)



submission.to_csv("Submission.csv",index=False)