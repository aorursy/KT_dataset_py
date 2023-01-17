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
from keras.models import Sequential

from keras.layers import Dense,Activation,Flatten, Dropout

from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import SGD

from keras.utils import np_utils

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample = pd.read_csv('../input/sample_submission.csv')
y = train.label

X = train.drop('label', axis=1)
X_train = X.values/255

X_train = X_train.reshape(-1,28,28,1)     # 这地方一定要用（-1，28，28，1）

y_train = np_utils.to_categorical(y)
X_test = test.values / 255

X_test = X_test.reshape(-1, 28, 28, 1)
def myModel(weights_path = None):

    model = Sequential()

    

    model.add(ZeroPadding2D((1,1), input_shape=(28,28,1)))

    model.add(Conv2D(64, (3,3),padding='same', activation='relu'))

    model.add(ZeroPadding2D((1,1)))

    model.add(Conv2D(64, (3,3),padding='same', activation='relu'))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    

    model.add(ZeroPadding2D((1,1)))

    model.add(Conv2D(128, kernel_size=3, activation='relu'))

    model.add(ZeroPadding2D((1,1)))

    model.add(Conv2D(128, kernel_size=3, activation='relu'))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    

    model.add(ZeroPadding2D((1,1)))

    model.add(Conv2D(256, kernel_size=3, activation='relu'))

    model.add(ZeroPadding2D((1,1)))

    model.add(Conv2D(256, kernel_size=3, activation='relu'))

    model.add(ZeroPadding2D((1,1)))

    model.add(Conv2D(256, kernel_size=3, activation='relu'))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    

    model.add(ZeroPadding2D((1,1)))

    model.add(Conv2D(512, kernel_size=3, activation='relu'))

    model.add(ZeroPadding2D((1,1)))

    model.add(Conv2D(512, kernel_size=3, activation='relu'))

    model.add(ZeroPadding2D((1,1)))

    model.add(Conv2D(512, kernel_size=3, activation='relu'))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(4096, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    

    return model
VGG = myModel()

optimizer = SGD()

VGG.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history = VGG.fit(X_train,y_train,batch_size=64, epochs=15, validation_split=0.2,verbose=1)
# show the accuracy and loss 

print(history.history.keys())

# accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



# loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
ret = VGG.predict(X_test)
results = np.argmax(ret,axis = 1)
df = {'ImageId':sample['ImageId'],

     'Label':results }



submission = pd.DataFrame(df)
submission.to_csv('submission.csv', index=False)