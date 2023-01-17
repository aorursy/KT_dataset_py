from tensorflow.python.client import device_lib

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split # sklearn, train_test_split

import matplotlib.pyplot as plt # matplotlib, plt

from keras.utils.np_utils import to_categorical # categorical - one-hot-encoding



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# CNN libraries

from sklearn.metrics import confusion_matrix

import itertools

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator



# Any results you write to the current directory are saved as output.
print(os.listdir("../input"))

directory_train = "../input/Kannada-MNIST/train.csv"

directory_test = "../input/Kannada-MNIST/test.csv"
test_data = pd.read_csv(directory_test)

test_data.head(10)
train_data = pd.read_csv(directory_train)

train_data.head(10)
print('Test Shape:',test_data.shape)

print('Train Shape:',train_data.shape)
y_train = train_data["label"]

x_train = train_data.drop(['label'],axis=1)

y_train = to_categorical(y_train,num_classes=10)



x_test = test_data.drop(['id'], axis=1)
print(y_train.shape[0])

print(x_train.shape[0])

print(test_data.shape[0])


x_train /= 255.0

x_test /= 255.0

x_train = x_train.values.reshape(x_train.shape[0],28, 28,1)

x_test = x_test.values.reshape(x_test.shape[0],28, 28,1)

def separating_data():

    return train_test_split(x_train, 

                            y_train, 

                            test_size = 0.05,

                            random_state = 42

                           )



x_train, x_validation, y_train, y_validation = train_test_split(x_train, 

                                                                y_train, 

                                                                test_size = 0.05,

                                                                random_state = 42)
max_epochs = 2

model = Sequential()



activations = ['relu', 'softmax'] 



model.add(Conv2D(filters = 32, 

                 kernel_size = (5,5),

                 padding = 'Same', 

                 activation = activations[0], 

                 input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, 

                 kernel_size = (5,5),

                 padding = 'Same', 

                 activation = activations[0]))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, 

                 kernel_size = (3,3),

                 padding = 'Same', 

                 activation = activations[0]))

model.add(Conv2D(filters = 64, 

                 kernel_size = (3,3),

                 padding = 'Same', 

                 activation = activations[0]))



model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = activations[0]))

model.add(Dropout(0.5))

model.add(Dense(10, activation = activations[1]))



model.compile(loss='categorical_crossentropy', 

              optimizer='adam', 

              metrics=["accuracy"])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
generator = ImageDataGenerator()

batches = generator.flow(x_train, 

                         y_train, 

                         batch_size=32)

batches_value = generator.flow(x_validation, 

                               y_validation, 

                               batch_size=32)



history = model.fit_generator(generator=batches, 

                              steps_per_epoch=batches.n, 

                              epochs=max_epochs, 

                              validation_data=batches_value, 

                              validation_steps=batches_value.n)
my_predictions = model.predict_classes(x_test)

submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = my_predictions

submission.to_csv("submission.csv",index=False)