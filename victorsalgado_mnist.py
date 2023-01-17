import numpy as np 

import pandas as pd

import tensorflow as tf



from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



import os

print(os.listdir("../input"))
# training data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
X_train = train.drop(labels=["label"],axis=1)

y_train = train["label"]

y_train = to_categorical(y_train, num_classes = 10)

del train



# normalize data

X_train = X_train / 255.0

test = test / 255.0
# reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
# separate features and labels on training set

from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state = 2)
# CNN implementation



# Archicture: IN -> [[Conv2D relu] * 2 -> MaxPool2D -> Dropout] * 2 -> Flatten -> Dense -> Dropout -> Out



from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



model = Sequential()

#

model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

#

model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

# fully connected

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
from keras.optimizers import Adam



opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics=["accuracy"])
# parameters

epochs = 10

batch_size = 250
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.05, # Randomly zoom image 

        width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                    epochs = epochs, validation_data = (X_val,y_val), 

                    steps_per_epoch=X_train.shape[0])
predictions = model.predict(X_val)

loss, accu = model.evaluate(X_val, y_val)

predictions
accu
predictions_test_data = model.predict(test)



y, x = predictions_test_data.shape



predictions_test_data_normalized = np.ones(y, dtype=int)



for idx, item in enumerate(predictions_test_data):

    predictions_test_data_normalized[idx] = int(item.argmax())



predictions_test_data_normalized
data_to_submit = pd.DataFrame(columns=['Label'])

data_to_submit['Label'] = predictions_test_data_normalized

data_to_submit.insert(0, 'ImageID', range(1, 1 + len(data_to_submit)))

data_to_submit

data_to_submit.to_csv('submission.csv', index = False)