from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical

import pandas as pd
import numpy as np
#  DEFINE CONSTANTS

INPUT_SHAPE = 784
NUM_CATEGORIES = 10

LABEL_DICT = {
 0: "T-shirt/top",
 1: "Trouser",
 2: "Pullover",
 3: "Dress",
 4: "Coat",
 5: "Sandal",
 6: "Shirt",
 7: "Sneaker",
 8: "Bag",
 9: "Ankle boot"
}

# LOAD THE RAW DATA
train_raw = pd.read_csv('../input/fashion-mnist_train.csv').values
test_raw = pd.read_csv('../input/fashion-mnist_test.csv').values
# split into X and Y, after one-hot encoding
train_x, train_y = (train_raw[:,1:], to_categorical(train_raw[:,0], num_classes = NUM_CATEGORIES))
test_x, test_y = (test_raw[:,1:], to_categorical(test_raw[:,0], num_classes = NUM_CATEGORIES))

# normalize the x data
train_x = train_x / 255
test_x = test_x / 255
# BUILD THE MODEL
model = Sequential()

model.add(Dense(512, input_dim = INPUT_SHAPE))
model.add(Activation('relu'))

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(NUM_CATEGORIES))
model.add(Activation('softmax'))

# compile it - categorical crossentropy is for multiple choice classification
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# train the model!
model.fit(train_x,
          train_y,
          epochs = 8,
          batch_size = 32,
          validation_data = (test_x, test_y))
# how'd the model do?
model.evaluate(train_x, train_y)