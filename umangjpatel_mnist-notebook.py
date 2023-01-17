# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting library



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_dataset = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

y_train = train_dataset['label'].values

x_train = train_dataset.iloc[:, 1:].values / 255.0

print("Training => features : {}, labels : {}".format(x_train.shape, y_train.shape))

x_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv').values / 255.0

print("Testing => features : {}".format(x_test.shape))
cnn_x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

cnn_x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

print("CNN Training => features : {}".format(cnn_x_train.shape))

print("CNN Testing => features : {}".format(cnn_x_test.shape))
import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense



EPOCHS = 20



lenet_model = Sequential()

lenet_model.add(Conv2D(filters=6, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))

lenet_model.add(MaxPooling2D(pool_size = (2,2)))

lenet_model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))

lenet_model.add(MaxPooling2D(pool_size = (2,2)))

lenet_model.add(Flatten())

lenet_model.add(Dense(units = 128, activation = 'relu'))

lenet_model.add(Dense(units = 84, activation = 'relu'))

lenet_model.add(Dense(units = 10, activation = 'softmax'))



lenet_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

lenet_history = lenet_model.fit(cnn_x_train, y_train, epochs = EPOCHS)

lenet_loss, lenet_acc = lenet_history.history['loss'], lenet_history.history['acc']



plt.plot(range(EPOCHS), lenet_loss, color='red')

plt.title('LeNet-5 Loss Curve')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.show()



plt.plot(range(EPOCHS), lenet_acc, color='green')

plt.title('LeNet-5 Accuracy Curve')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.show()
lenet_preds = np.argmax(lenet_model.predict(cnn_x_test), axis=1)

print("LeNet-5 model predictions : {}".format(lenet_preds))
submission_dict = {'ImageId' : [idx + 1 for idx in range(x_test.shape[0])], 'Label' : lenet_preds}

submission_file = pd.DataFrame(submission_dict)

submission_file.to_csv('submission.csv')