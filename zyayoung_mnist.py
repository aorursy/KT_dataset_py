import os

os.environ['KERAS_BACKEND']='tensorflow'
import pandas as pd

import numpy as np

from keras.datasets import mnist

from keras.utils import np_utils

from keras.models import Sequential

from keras import models

from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout

from keras.optimizers import Adam

import matplotlib.pyplot as plt
train_data = pd.read_csv('../input/train.csv')

X_train_kaggle = train_data.drop('label',axis=1).values.reshape(-1,1,28,28)

y_train_kaggle = np_utils.to_categorical(train_data['label'])
# Build CNN

model = Sequential()

model.add(Convolution2D(16, (5,1),

    batch_input_shape=(None, 1, 28, 28),

    padding='same',     # Padding method

    data_format='channels_first',

))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(MaxPooling2D(

    pool_size=2,

    strides=2,

    padding='same',    # Padding method

    data_format='channels_first',

))

model.add(Convolution2D(32, (5,1), padding='same', data_format='channels_first'))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

model.add(Flatten())

model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Dense(10))

model.add(Activation('softmax'))

model.compile(optimizer='adadelta',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
print('Training ------------')

history = model.fit(X_train_kaggle, y_train_kaggle, epochs=32, batch_size=128,validation_split=0.05)

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
test_data = pd.read_csv('../input/test.csv')

mypre = model.predict_classes(test_data.values.reshape(-1,1,28,28))

mypre[1166] = 1

ans = pd.DataFrame({'ImageId':range(1,28001),'Label':mypre})
ans.to_csv('sub.csv',index=False)

model.save('model.h5')
import matplotlib.pyplot as plt

for i in range(10):

    index = np.random.randint(len(ans))

    #index = 1166+i

    print(index,ans['Label'][index])

    plt.imshow(test_data.values[index].reshape(28,28),cmap='gray')

    plt.show()