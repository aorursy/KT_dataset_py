# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#load data

x_ = np.load("../input/sign-language-digits-dataset/Sign-language-digits-dataset/X.npy")

y_ = np.load("../input/sign-language-digits-dataset/Sign-language-digits-dataset/Y.npy")



print("x_ shape:",x_.shape)

print("y_ shape:",y_.shape)
#visualize

plt.subplot(2, 2, 1)

plt.imshow(x_[260])

plt.axis('off')

plt.subplot(2, 2, 2)

plt.imshow(x_[500])

plt.axis('off')

plt.subplot(2, 2, 3)

plt.imshow(x_[1300])

plt.axis('off')

plt.subplot(2, 2, 4)

plt.imshow(x_[2000])

plt.axis('off')

plt.show()
#reshape

x = x_.reshape(-1,64,64,1)

print("x shape: ", x.shape)



y = y_

print("y shape: ", y.shape)
#train test split process

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)



print("x_train shape: ",x_train.shape)

print("x_test shape: ",x_test.shape)

print("y_train shape: ",y_train.shape)

print("y_test shape: ",y_test.shape)
from sklearn.metrics import confusion_matrix

import itertools



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



model = Sequential()



model.add(Conv2D(filters = 64, kernel_size = (6,6),padding = 'Same', activation ='relu', input_shape = (64,64,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.25))



#**************************



model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(256, activation = "relu"))

model.add(Dense(512, activation = "relu"))

model.add(Dense(256, activation = "relu"))

model.add(Dense(10, activation='softmax'))



#**************************

optimizer = RMSprop(lr = 0.0001)

model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 30, batch_size = 16, verbose = 1 )
plt.plot(history.history["loss"], label = "Train loss")

plt.plot(history.history["val_loss"], label = "Test Loss")

plt.title("Loss graph")

plt.xlabel("Num of epochs")

plt.ylabel("Loss value")

plt.legend()

plt.show()

plt.plot(history.history["accuracy"], label = "Train accuracy")

plt.plot(history.history["val_accuracy"], label = "Test accuracy")

plt.title("Accuracy graph")

plt.xlabel("Num of epochs")

plt.ylabel("Acc value")

plt.legend()

plt.show()