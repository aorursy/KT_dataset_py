import numpy as np 

import pandas as pd

from matplotlib.pyplot import imshow

import matplotlib.pyplot as plt

from PIL import Image

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.constraints import maxnorm

from keras.optimizers import SGD

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils

from keras import backend as K

from random import randint

%matplotlib inline



np.random.seed(7)

K.set_image_dim_ordering('th')
X_FNAME = "../input/alphanum-hasy-data-X.npy"

Y_FNAME = "../input/alphanum-hasy-data-y.npy"

SYMBOL_FNAME = "../input/symbols.csv"
X = np.load(X_FNAME)

y = np.load(Y_FNAME)

SYMBOLS = pd.read_csv(SYMBOL_FNAME) 

SYMBOLS = SYMBOLS[["symbol_id", "latex"]]



print("X.shape", X.shape)

print("y.shape", y.shape)

print("SYMBOLS")

SYMBOLS.head(2)
def symbol_id_to_symbol(symbol_id = None):

    if symbol_id:

        symbol_data = SYMBOLS.loc[SYMBOLS['symbol_id'] == symbol_id]

        if not symbol_data.empty:

            return str(symbol_data["latex"].values[0])

        else:

            print("This should not have happend, wrong symbol_id = ", symbol_id)

            return None

    else: 

        print("This should not have happend, no symbol id passed")

        return None        



# test some values

print("21 = ", symbol_id_to_symbol(21))

print("32 = ", symbol_id_to_symbol(32))

print("90 = ", symbol_id_to_symbol(90))
f, ax = plt.subplots(2, 3, figsize=(12, 10))

ax_x = 0

ax_y = 0



for i in range(6):

    randKey = randint(0, X.shape[0])

    ax[ax_x, ax_y].imshow(X[randKey], cmap='gray')

    ax[ax_x, ax_y].title.set_text("Value : " + symbol_id_to_symbol(y[randKey]))



    # for proper subplots

    if ax_x == 1:

        ax_x = 0

        ax_y = ax_y + 1

    else:

        ax_x = ax_x + 1
# print labels vs frequency matrix

unique, counts = np.unique(y, return_counts=True)

y_info_dict = { "labels" : unique, "counts": counts }

y_info_frame = pd.DataFrame(y_info_dict)



y_info_frame["labels"] = y_info_frame["labels"].apply(lambda x: symbol_id_to_symbol(x))

y_info_frame.head()
f, ax = plt.subplots(figsize=(10, 20))

y_info_frame["counts"].plot(kind='barh', legend=False, color="rgbkymc", alpha=0.5)

wrap = ax.set_yticklabels(list(y_info_frame["labels"]))



rects = ax.patches

bar_labels_counts = list(y_info_frame["counts"])



for i in range(len(bar_labels_counts)):

    label_value = str(bar_labels_counts[i])

    ax.text(40, rects[i].get_y(), label_value, ha='center',

          va='bottom', size='medium', color="black", fontweight="bold")
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



print("Train dataset shape")

print(X_train.shape, y_train.shape)

print("Test dataset shape")

print(X_test.shape, y_test.shape)
# normalize inputs from 0-255 to 0-1

X_train = X_train / 255

X_test = X_test / 255



# one hot encode outputs

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)



num_classes = y_test.shape[1]

print("num_classes = ", num_classes)
num_pixels = X_train.shape[1] * X_train.shape[2]

X_train = X_train.reshape(X_train.shape[0], num_pixels)

X_test = X_test.reshape(X_test.shape[0], num_pixels)



def model_ANN():

    # create model

    model = Sequential()

    model.add(Dense(num_pixels, input_dim=num_pixels,

                    kernel_initializer='normal', activation='relu'))

    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

    # Compile model

    model.compile(loss='categorical_crossentropy',

                  optimizer='adam', metrics=['accuracy'])

    

    print(model.summary())

    return model
# build the model

model = model_ANN()

# Fit the model

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=100, verbose=2)
# summarize history for accuracy

f, ax = plt.subplots(figsize=(12, 4))

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# summarize history for accuracy

f, ax = plt.subplots(figsize=(12, 4))

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
GENERATED_MODEL = "alpha_hasy_model_v1.h5"

model.save(GENERATED_MODEL)
from keras.models import load_model



model = load_model(GENERATED_MODEL)
test_image_key = 100 # change to test on other images

test_image = X_test[test_image_key]

test_image = test_image.reshape(1, test_image.shape[0])

test_label = y_test[test_image_key]



classes = model.predict(test_image)

output = symbol_id_to_symbol(np.argmax(classes))
f, ax = plt.subplots(figsize=(4, 4))



test_image_2d = test_image.reshape(32, 32)

test_label_value = symbol_id_to_symbol(np.argmax(test_label))



ax.imshow(test_image_2d, cmap='gray')

f.suptitle("Real Output: " + test_label_value, fontsize=20)



ax.set_xlabel('Predicted Output: ' + output, fontsize=20)