# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Flatten, Input, Dense, Activation, Dropout

from keras.callbacks import EarlyStopping

from keras.utils import to_categorical

from matplotlib import pyplot

train_dir = '../input/digit-recognizer/train.csv'



data = pd.read_csv(train_dir)

# extract x,y training data

y_train = data['label']

y_train = to_categorical(y_train.values, num_classes=10)

print(y_train.shape)

x_train = data.drop(labels = ['label'], axis = 1)

x_train = x_train.to_numpy()

# normalise values into [0,1] range to speed up training

x_train = x_train/255

# resize to represent 28x28 image

# x_train shape changes from (42000,784) to (42000, 28, 28, 1) 

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

print(x_train.shape)
for i in range(9):

    pyplot.subplot(330 + 1 + i)

    pyplot.imshow(x_train[i][:,:,0])
test_dir = '../input/digit-recognizer/test.csv'

test = pd.read_csv(test_dir)

test = test.to_numpy()

test = test/255

test = test.reshape(test.shape[0], 28, 28, 1)
simple_cnn = Sequential()

simple_cnn.add(Conv2D(filters=20, kernel_size=(2, 2), activation='relu', input_shape=(28,28,1)))

simple_cnn.add(MaxPool2D())

simple_cnn.add(Conv2D(filters=20, kernel_size=(2, 2), activation='relu'))

simple_cnn.add(MaxPool2D())

simple_cnn.add(Flatten())

simple_cnn.add(Dense(units = 120, activation = 'relu'))

simple_cnn.add(Dense(units = 10, activation = 'softmax'))



simple_cnn.compile('adam', 'categorical_crossentropy',metrics = ['acc'])

simple_cnn.summary()
# Early stopping dependent on validation accuracy

es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=10)



history = simple_cnn.fit(x_train, y_train, validation_split = 0.2, epochs = 500, callbacks = [es])
pyplot.plot(history.history['acc'], label='train')

pyplot.plot(history.history['val_acc'], label='val')

pyplot.legend()

pyplot.show()
preds = simple_cnn.predict(test)

preds = np.argmax(preds,axis = 1)

preds
results = pd.DataFrame()

results['ImageId'] = np.arange(len(preds)) + 1

results['Label'] = pd.Series(preds)

results

# index false so we don't write row names

results.to_csv('submission.csv', index = False)