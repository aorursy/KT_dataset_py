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
import pandas as pd
pd_data = pd.read_csv('../input/train.csv')
pd_data.head()
train_label = pd_data['label']
del pd_data['label']
pd_data = pd_data.astype(np.float)
pd_data = np.multiply(pd_data, 1.0/255.0)
train_label = np_utils.to_categorical(train_label)
train_label[0]
validation_images = pd_data[:2000]
validation_labels = train_label[:2000]
train_images = pd_data[2000:]
train_labels = train_label[2000:]

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.core import Reshape
def dig_model():
    model = Sequential()
    model.add(Reshape((28,28,1), input_shape=(784, )))
    model.add(Convolution2D(6, 5, 5, border_mode='valid', input_shape=(28,28,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Convolution2D(16, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Convolution2D(120, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = dig_model()
model.fit(train_images, train_labels, validation_data = (validation_images, validation_labels), nb_epoch=200, batch_size=200, verbose=2)
test_images = pd.read_csv('../input/test.csv')
test_images = test_images.astype(np.float).values
test_images = np.multiply(test_images, 1.0 / 255.0)
predictions = model.predict_classes(test_images, verbose=0)
submissions=pd.DataFrame({"ImageId":list(range(1,len(predictions)+1)),
                         "Label":predictions})
submissions.to_csv("DR.csv",index=False, header=True)
