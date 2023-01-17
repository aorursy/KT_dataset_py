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
# Import Required Library

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import np_utils
def MNIST_Model():

    # Build model

    model = Sequential()

    #model.add(Convolution2D(num_pixels, 5, 5, input_shape = (28, 28, 1), activation="relu"))

    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu"))

    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128, activation="relu"))

    model.add(Dense(num_classes, activation="softmax"))

    

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer='adam', matrics=['accuracy'])

    return model
seed = 9

np.random.seed(seed)
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
images = train.iloc[:, 1:].values

test_images = test.iloc[:, 0:].values
images = images.reshape(images.shape[0],28, 28, 1).astype('float32')

test_images= test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
# scaling

scaled_value = np.max(images)

images = images / scaled_value

test_images = test_images / scaled_value



num_pixels = images.shape[1]
labels = train.iloc[:, 0].values.astype('int32')
encoded_labels = np_utils.to_categorical(labels)

num_classes = encoded_labels.shape[1]
model = MNIST_Model()
# model.fit(images, encoded_labels, epochs=10, batch_size=10, verbose=2)

model.fit(images, encoded_labels, epochs = 10, batch_size=10, validation_split=0.1,verbose=2)
scores = model.predict(test_images)

scores.argmax(1)
# Summarizing the model

model.summary()
preds = model.predict_classes(test_images, verbose = 0)



def write_preds(preds, fname):

    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)



write_preds(preds, "sample_submission.csv")