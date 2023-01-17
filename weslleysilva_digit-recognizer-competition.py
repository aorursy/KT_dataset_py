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
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import CSVLogger

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

teste = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train.head(5)
y = train["label"]

x = train.drop(labels=["label"],axis=1)
x = x.values.reshape(-1,28,28,1)

teste = teste.values.reshape(-1, 28, 28, 1)

y = to_categorical(y, num_classes=10)
x_training, x_validation, y_training, y_validation = train_test_split(x,

                                                                      y,

                                                                      test_size=0.33,

                                                                      shuffle=True)
data_generator = ImageDataGenerator(rescale=1./255,

                                    rotation_range=10,

                                    zoom_range=0.15, 

                                    width_shift_range=0.1,

                                    height_shift_range=0.1)

data_generator.fit(x_training)
model = Sequential()



model.add(Conv2D(filters=32,

                kernel_size=(5,5),

                padding='Same',

                activation='relu',

                input_shape=(28,28,1)))

model.add(Conv2D(filters=32,

                kernel_size=(5,5),

                padding='Same',

                activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.5))



model.add(Conv2D(filters=64, kernel_size=(3,3),padding='Same', 

                 activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3),padding='Same', 

                 activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.5))





model.add(Flatten())

model.add(Dense(8129,activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(2048, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(10, activation="softmax"))



model.compile(optimizer=RMSprop(lr=0.0001,

                                rho=0.9,

                                epsilon=1e-08,

                                decay=0.00001),

              loss="categorical_crossentropy",

              metrics=["accuracy"])
predictions = model.predict_classes(teste, verbose=1)

pd.DataFrame({"ImageId":list(range(1,len(predictions)+1)),

              "Label":predictions}).to_csv("kaggle_submission.csv",

                                           index=False,

                                           header=True)