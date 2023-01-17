import pandas as pd

import numpy as np

import keras 

from matplotlib import pyplot as plt

from keras.utils import to_categorical

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout

from keras.layers import Activation, Flatten, Dense

from keras.preprocessing.image import ImageDataGenerator
data_train = pd.read_csv("../input/digit-recognizer/train.csv")

data_test = pd.read_csv("../input/digit-recognizer/test.csv")

print("Dataset ready")
#converts to numpy array

data_train=data_train.as_matrix()

data_test=data_test.as_matrix()
#x is lable y is data in this case, but convention other way around

y=data_train[:,0:1]

#our model expects one-hot-vector

y=to_categorical(y)

y.shape
x=data_train[:,1:]
x=x.reshape(42000,28,28,1)

x.shape



data_test=data_test.reshape(28000,28,28,1)

data_test.shape
model = keras.models.Sequential()



model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(64, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = 3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(128, kernel_size = 4, activation='relu'))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(.001),metrics=['accuracy'])

epochs = 3

batch_size = 256

datagen = ImageDataGenerator(

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    zoom_range = 0.10

    )

# compute quantities required for featurewise normalization

# (std, mean, and principal components if ZCA whitening is applied)

datagen.fit(x)



# fits the model on batches with real-time data augmentation:

model.fit_generator(datagen.flow(x, y, batch_size=batch_size),

                    steps_per_epoch=len(x) / batch_size, epochs=epochs)



# here's a more "manual" example

for e in range(epochs):

    print('Epoch', e)

    batches = 0

    for x_batch, y_batch in datagen.flow(x, y, batch_size= batch_size):

        model.fit(x_batch, y_batch, validation_split=0.2)

        batches += 1

        if batches >= len(x) / batch_size:

            # we need to break the loop by hand because

            # the generator loops indefinitely

            break
results=model.predict_classes(data_test)

submission = pd.DataFrame({"ImageId": list(range(1,len(results)+1)),"Label": results})



submission.to_csv("submission.csv", index=False)
