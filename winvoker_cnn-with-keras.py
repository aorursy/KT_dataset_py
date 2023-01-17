import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt # data visualization



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")

test  = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv")

train.head()
train.shape
y_test = test.label.values.reshape(-1,1) # 1 at the end is necessary because keras needs it.

x_test = np.array(test.drop(["label"],axis=1)).reshape(test.shape[0],28,28,1)
from keras.utils import to_categorical



y_train = train.label.values.reshape(-1,1) # 1 at the end is necessary because keras needs it.

x_train = np.array(train.drop(["label"],axis=1)).reshape(train.shape[0],28,28,1) # same as here

print(y_train.shape,x_train.shape)

y_train = to_categorical(y_train,num_classes=10)

y_test = to_categorical(y_test,num_classes=10)
x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train=x_train/255.0

x_test=x_test/255.0
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

        featurewise_center=False, 

        samplewise_center=False, 

        featurewise_std_normalization=False,

        samplewise_std_normalization=False,  

        zca_whitening=False,

        rotation_range=0.5, 

        zoom_range = 0.5, 

        width_shift_range=0.5,  # 

        height_shift_range=0.5,  # 

        horizontal_flip=False, 

        vertical_flip=False)



datagen.fit(x_train)
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.optimizers import Adam



batch = 5

epoch = 10



model = Sequential()

model.add(Conv2D(filters=64,kernel_size=(4,4), activation='relu',input_shape=(28,28,1)))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=64,kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=128,kernel_size=(3,3), activation='relu'))



model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))



optimizer = Adam()

model.compile(loss="categorical_crossentropy" ,optimizer = optimizer ,metrics=['accuracy'])

#history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch),epochs=epoch,validation_data=(x_test, y_test),steps_per_epoch=x_train.shape[0] // batch)

history = model.fit(x_train,y_train,batch_size=batch,verbose = 1, epochs = epoch,validation_data=(x_test,y_test))

score = model.evaluate(x_test, y_test)
plt.plot(history.history['val_loss'],label="validation loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.title("Test Loss")

plt.legend()

plt.show()
predictions = model.predict(x_test)

print("Test loss :",score[0])

print("Test Accuracy : ",score[1])