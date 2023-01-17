import os

root='/kaggle/input/Sign-language-digits-dataset/'

print(os.listdir(root))
import numpy as np

x=np.load(root+'X.npy')

y=np.load(root+'Y.npy')
print(x.shape)

print(y.shape)
y[0]
x[0]
import matplotlib.pyplot as plt

plt.imshow(x[0],cmap=plt.cm.gray)

plt.show()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y)

x_train, x_val, y_train, y_val=train_test_split(x_train,y_train)
from keras.optimizers import Adam ,RMSprop

from keras.models import  Sequential

from keras.layers.core import Dense, Flatten, Dropout

from keras.callbacks import EarlyStopping

from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D

from keras.preprocessing import image
model = Sequential([

        Convolution2D(32,(3,3), activation='relu',input_shape=(64,64,1)),

        BatchNormalization(),

        Convolution2D(32,(3,3), activation='relu'),

        MaxPooling2D(),

        BatchNormalization(),

        Convolution2D(64,(3,3), activation='relu'),

        BatchNormalization(),

        Convolution2D(64,(3,3), activation='relu'),

        MaxPooling2D(),

        Flatten(),

        BatchNormalization(),

        Dense(512, activation='relu'),

        BatchNormalization(),

        Dense(10, activation='softmax')])

model.compile(Adam(lr=0.001,decay=1e-6), loss='categorical_crossentropy', metrics=['accuracy'])

gen = image.ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1)

batches = gen.flow(x_train.reshape(1159, 64, 64,1), y_train, batch_size=64)

val_batches=gen.flow(x_val.reshape(387, 64, 64,1), y_val, batch_size=64)
history=model.fit_generator(generator=batches, steps_per_epoch=len(x_train), epochs=4, 

                    validation_data=val_batches, validation_steps=val_batches.n)
loss,acc=model.evaluate(x_test.reshape((516, 64, 64,1)),y_test)

print(loss,acc)