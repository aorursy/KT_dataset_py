# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras import models

from keras import layers

from keras.callbacks import ReduceLROnPlateau



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Load the data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



x_train = train.drop('label', axis=1).values

y_train = train['label'].values

x_train.shape = -1, 28, 28, 1

x_test = test.values

x_test.shape = -1, 28, 28, 1
print(x_train.shape, y_train.shape)

print(x_test.shape)
model = models.Sequential([

    layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),

    layers.MaxPooling2D(),  # 19 -> 9

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),

    layers.MaxPooling2D(),  # 9 -> 4

    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),

    layers.MaxPooling2D(),  # 4 -> 2

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.25),

    layers.Dense(64, activation='relu'),

    layers.Dense(10, activation='softmax'),

])

model.summary()

model.compile(optimizer='rmsprop',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

# 当标准评估停止提升时，降低学习速率

reduce_lr = ReduceLROnPlateau(verbose=1)

model.fit(x_train, y_train, epochs=32,

          validation_split=0.2,

          callbacks=[reduce_lr])
results = model.predict(x_test)

results = np.argmax(results, axis=1)



results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)

submission.to_csv("cnn_mnist_datagen.csv", index=False)