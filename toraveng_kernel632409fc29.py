import numpy as np 

import pandas as pd

import keras

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.models import Sequential

from keras import optimizers
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')



valid_part = 10

test_id = test.id

test = test.drop('id', axis=1)

y_train = train.label

x_train = train.drop('label', axis=1)



train_size = int(x_train.shape[0] / valid_part * (valid_part - 1))

x_valid, y_valid = x_train[train_size:], y_train[train_size:]

x_train, y_train = x_train[:train_size], y_train[:train_size]
scaler = MinMaxScaler(feature_range=(-1, 1))



rows, cols = 28, 28

x_train = x_train.astype('float32')

test = test.astype('float32')



scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_valid = scaler.transform(x_valid)

test = scaler.transform(test)

print(x_train.min(), x_train.max())
x_train = x_train.reshape(x_train.shape[0], rows, cols)

for i in range(10):

    plt.subplot(1, 10, i+1)

    plt.title(y_train[i])

    plt.imshow(x_train[i], cmap='gray')

    plt.axis('Off')
x_train = x_train.reshape(x_train.shape[0], rows, cols, 1)

test = test.reshape(test.shape[0], rows, cols, 1)

x_valid = x_valid.reshape(x_valid.shape[0], rows, cols, 1)



y_train = keras.utils.to_categorical(y_train, 10)

y_valid = keras.utils.to_categorical(y_valid, 10)
epochs = 12

batch_size = 64



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(rows, cols, 1)))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer='sgd',

              metrics=['accuracy'])



history_callback = model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=0,

          validation_data=(x_valid, y_valid))

loss_history = history_callback.history["loss"]

np_loss_history = np.array(loss_history)

np.savetxt("loss_history.csv", np_loss_history, delimiter=",")

label = model.predict(test)

label = np.argmax(label, axis=1)

df = pd.DataFrame({'id': test_id, 'label': label})

df.to_csv('submission.csv', index=False)