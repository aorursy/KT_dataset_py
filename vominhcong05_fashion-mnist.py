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
training_data = pd.read_csv('../input/fashion-mnist_train.csv')
training_data.head(5)
%matplotlib inline

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten

from keras.utils.np_utils import to_categorical
X_train = training_data.iloc[:, 1::].values

Y_train = training_data.iloc[:, 0].values
sample = np.reshape(X_train[0, :], (28, 28))
plt.figure(figsize=(5, 5))

for i in range(0, 9):

    plt.subplot(3, 3, i+1)

    sample = np.reshape(X_train[i, :], (28, 28))

    plt.imshow(sample, 'gray')

    title = 'label: ' + str(Y_train[i])

    plt.title(title, fontsize=10)
X_train = X_train/255.0
# Reshape

X_train = X_train.reshape((-1, 28, 28, 1))
Y_train = to_categorical(Y_train)
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, random_state = 0, test_size=0.3)
print(X_train.shape)

print(X_val.shape)
print(Y_train.shape)

print(Y_val.shape)
model = Sequential()
model.add(Conv2D(filters=32, activation='relu', kernel_size=(5, 5), input_shape=(28, 28, 1), padding='same'))

model.add(Conv2D(filters=32, activation='relu', kernel_size=(5, 5), padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(filters=64, activation='relu', kernel_size=(3, 3), padding='same'))

model.add(Conv2D(filters=64, activation='relu', kernel_size=(3, 3), padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.2))



model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=100, epochs=12, validation_data=(X_val, Y_val))
data_test = pd.read_csv('../input/fashion-mnist_test.csv')
data_test.head(10)
X_test = data_test.iloc[:, 1:].values
X_test =  X_test/255.0
Y_test = data_test['label']

Y_test = to_categorical(Y_test)
X_test.shape
X_test = X_test.reshape((-1, 28, 28, 1))
prediction = model.predict(X_test)
label = np.argmax(prediction, axis=1)
test_id = np.reshape(range(1, len(prediction) + 1), label.shape)
my_submission = pd.DataFrame({'ImageId': test_id, 'Label': label})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)