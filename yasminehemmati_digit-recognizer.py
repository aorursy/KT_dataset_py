# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import seaborn as sns

from keras.preprocessing.image import ImageDataGenerator

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
print(train.shape)

train.head()
Y_train = train["label"]

X_train = train.drop('label', axis=1)
X_train = X_train / 255.0

test = test / 255.0

X_train = X_train.values.reshape(-1,28,28, 1)

test = test.values.reshape(-1,28,28, 1)

print("x_train shape: ",X_train.shape)

print("test shape: ",test.shape)


def display(i):

    img = X_train[i]

    plt.imshow(img.reshape((28,28)), cmap=plt.cm.gray_r)
#plt.show(plt.imshow(X_train[1][1]))

display(3)
xtrain, xtest, ytrain, ytest = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 123)


ytrain = to_categorical(ytrain, num_classes = 10)

ytest = to_categorical(ytest, num_classes = 10)

#DATA AUGMENTATION

datagen = ImageDataGenerator(featurewise_center=False,

    featurewise_std_normalization=False,

    rotation_range=30,

    width_shift_range=0.2,

    height_shift_range=0.2,

    zoom_range = 0.1,           

    horizontal_flip=False,

    vertical_flip=False)





datagen.fit(xtrain)
model = Sequential()

#Convolutional Layer

#RELU layer

#Pooling Layer

#Fully connected layer

#Use dropout regularization in between layers

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))





model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))



model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(64,activation='relu'))

model.add(Dropout(0.20))

model.add(Dense(10, activation = "softmax"))

model.compile(loss= 'categorical_crossentropy', optimizer ='adam', metrics = ['accuracy'])

steps_per_epoch = len(xtrain)/ 32
from keras.callbacks import ReduceLROnPlateau

annealer = ReduceLROnPlateau(monitor='val_accuracy', patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.0001)
hist = model.fit_generator(datagen.flow(xtrain, ytrain, batch_size=32),

                           steps_per_epoch= steps_per_epoch,

                           epochs=10, 

                           verbose=2, 

                           validation_data=(xtest, ytest), 

                           callbacks=[annealer])
predicted = model.predict(test)

print(predicted)


predicted = np.argmax(predicted,axis = 1)

y_pred = pd.Series(predicted,name="Label")

print(y_pred)
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),y_pred],axis = 1)

submission.to_csv("submission.csv",index=False)