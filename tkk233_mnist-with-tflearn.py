import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
train_file = "../input/train.csv"

img_rows = img_cols = 28
NUM_CLASSES = 10

BATCH_SIZE = 64
EPOCHS = 1
VALIDATION_SPLIT = 0.2
def data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label, NUM_CLASSES)
    num_images = raw.shape[0]
    x_as_arr = raw.values[:, 1:]
    x_shaped_arr = x_as_arr.reshape(num_images, img_rows, img_cols, 1)
    out_x = x_shaped_arr / 255
    return out_x, out_y

raw_data = pd.read_csv(train_file)
x, y = data_prep(raw_data)
model = Sequential()
model.add(Conv2D(20, kernel_size = (3, 3), activation = 'relu',
                 input_shape = (img_rows, img_cols, 1)))
model.add(Conv2D(20, kernel_size = (3, 3), activation = 'relu'))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(NUM_CLASSES, activation = 'softmax'))
model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = 'adam',
              metrics = ['accuracy'])

model.fit(x, y, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = VALIDATION_SPLIT)

preds = model.predict(x)
for i in range(10):
    high = pred = 0
    for n, tag in enumerate(preds[i]):
        if tag > high:
            high, pred = tag, n
    img = x[i][:,:,0]
    plt.imshow(img, cmap=plt.cm.gray)
    plt.pause(1)
    plt.close()
    print('prediction: ', pred)
    for i, val in enumerate(y[i]):
        if val:
            actual = i
    print('actual: ', actual)