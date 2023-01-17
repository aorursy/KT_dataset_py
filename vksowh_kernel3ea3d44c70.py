from __future__ import print_function

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.preprocessing.image import img_to_array

from PIL import Image

import numpy as np

import os



path_train = "../input/fingers/train/"

path_test = "../input/fingers/test/"
batch_size = 32

num_classes = 12

epochs = 2



def get_data(path, X_array, Y_array):

    for filename in os.listdir(path):

        if filename[-3 : ] == "png":

            imgname = filename.split('.')[0]

            # label = imgname[-2 : ]

            hand = imgname[-1 : ]

            num = int(imgname[-2 : -1])

            if hand == 'L':

                label = num

            else:

                label = num + 6



            img = Image.open(path + filename)

            arr = img_to_array(img) / 255



            X_array.append(arr)

            Y_array.append(label)

    return X_array, Y_array





def load_data():

    print("[INFO] loading images ...")

    X_train = []

    Y_train = []

    X_test = []

    Y_test = []



    X_train, Y_train = get_data(path_train, X_train, Y_train)

    X_test, Y_test = get_data(path_test, X_test, Y_test)



    X_train = np.array(X_train)

    X_test = np.array(X_test)



    Y_train = np.array(Y_train)

    Y_test = np.array(Y_test)



    return (X_train, Y_train), (X_test, Y_test)
# load train and test data from fingers directory

(x_train, y_train), (x_test, y_test) = load_data()



print("x_train shape: ", x_train.shape)

print(x_train.shape[0], "train samples")

print(x_test.shape[0], "test samples")





# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))



model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))



# initiate RMSprop optimizer

opt = keras.optimizers.RMSprop(learning_rate = 0.0001, decay=1e-6)



# train the model using RMSprop

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size,

        epochs=epochs,

        validation_data=(x_test, y_test),

        shuffle=True)
# score trained model.

scores = model.evaluate(x_test, y_test, verbose=1)

print("Test loos: ", scores[0])

print("Test accuracy: ", scores[1])