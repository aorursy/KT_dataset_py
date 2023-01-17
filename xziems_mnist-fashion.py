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
import pandas as pd

import numpy as np

import keras.layers.core as core

import keras.layers.convolutional as conv

import keras.models as models

import keras.utils.np_utils as kutils



train = pd.read_csv("../input/fashion-mnist_train.csv").values

test  = pd.read_csv("../input/fashion-mnist_test.csv").values

print(train.shape)
n_epoch = 3



batch_size = 128

img_rows, img_cols = 28, 28



n_filters_1 = 128

n_filters_2 = 64

n_filters_3 = 64

n_conv = 3



trainX = train[:,1:].reshape(train.shape[0], img_rows, img_cols, 1)

trainX = trainX.astype(float)

trainX /= 255.0



trainY = kutils.to_categorical(train[:,0])

n_classes = trainY.shape[1]
cnn = models.Sequential()



#Convolutes over input pixels

cnn.add(conv.Conv2D(n_filters_1, n_conv, n_conv,

                           activation="relu",

                           input_shape=(img_rows, img_cols, 1),

                           border_mode='same'))

#Max Pools over an area of 2x2 pixels

cnn.add(conv.MaxPooling2D(strides=(2, 2)))



#Convolutes over previous convolution

cnn.add(conv.Conv2D(n_filters_1, n_conv, n_conv,

                           activation="relu",

                           border_mode='same'))

#Max Pools over an area of 2x2 pixels

cnn.add(conv.MaxPooling2D(strides=(2, 2)))



cnn.add(conv.Conv2D(n_filters_2, n_conv, n_conv,

                           activation="relu",

                           border_mode='same'))

#Max Pools over an area of 2x2 pixels

cnn.add(conv.MaxPooling2D(strides=(2, 2)))



cnn.add(conv.Conv2D(n_filters_2, n_conv, n_conv,

                           activation="relu",

                           border_mode='same'))



#Sixth Layer

cnn.add(conv.MaxPooling2D(strides=(2, 2)))



cnn.add(core.Flatten())

cnn.add(core.Dropout(0.18))

cnn.add(core.Dense(128, activation="relu"))

cnn.add(core.Dense(n_classes, activation="softmax"))



# Summarize the CNN thus far

cnn.summary()
#Compile the CNN

cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#train the CNN...

cnn.fit(trainX, trainY, batch_size=batch_size, epochs=n_epoch, verbose=1)
testX = test.reshape(test.shape[0], img_rows, img_cols, 1)

testX = testX.astype(float)

testX /= 255.0



yPred = cnn.predict_classes(testX)



np.savetxt('mnist-vggnet.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')