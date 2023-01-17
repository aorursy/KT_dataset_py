# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Activation

from keras.optimizers import SGD

from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.utils.np_utils import to_categorical
train = pd.read_csv("../input/train.csv")



test_images = (pd.read_csv("../input/test.csv").values).astype('float32')



train_images = (train.ix[:,1:].values).astype('float32')

train_labels = train.ix[:,0].values.astype('int32')



train_labels = to_categorical(train_labels, num_classes=10)



train_images = train_images.reshape([42000, 28, 28])

test_images = test_images.reshape([28000, 28, 28])



train_images = np.expand_dims(train_images, axis=3)

test_images = np.expand_dims(test_images, axis=3)
model=Sequential()

model.add(Convolution2D(16,(2, 2), input_shape=[28, 28, 1]))

model.add(Activation("relu"))

model.add(MaxPooling2D())

model.add(Convolution2D(64,(2, 2)))

model.add(Activation("relu"))

model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(64))

model.add(Activation("relu"))

model.add(Dropout(0.5))

model.add(Dense(10))

model.add(Activation("softmax"))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(sgd, "categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, nb_epoch=10)
predictions = model.predict(test_images)

prediction = np.argmax(predictions, axis=1)