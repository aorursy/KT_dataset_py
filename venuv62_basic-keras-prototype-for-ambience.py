import numpy as np

np.random.seed(123)

from keras.models import Sequential

import tensorflow as tf

import cv2

import pandas as pd
# create a 'classy_ambience' map for images

labels_df = pd.read_csv('../input/train.csv')

labels_df['classy_ambience'] = labels_df.apply(lambda row: 1 if '7' in str(row['labels']) else 0, axis=1)

photo_biz_df = pd.read_csv('../input/train_photo_to_biz_ids.csv')

photos_df = labels_df.merge(photo_biz_df,on='business_id')

print(list(photos_df))

print(len(photos_df))

print(photos_df.head(2))



img_width = 28

img_height = 28
import glob

import os

from PIL import Image



X_train_init = []

Y_train_init = []

training_size = 2000

files = glob.glob ("../input/train_photos/*.jpg")

count = 0

for myFile in files:

    img = cv2.imread(myFile)

#    img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )

    img = cv2.resize(img,(img_width,img_height))

    X_train_init.append (img)

    fname = os.path.splitext(os.path.basename(myFile))[0]

    y_train_val = photos_df.loc[photos_df['photo_id'] == int(fname)]['classy_ambience']

    if count%500 == 0:

        print("y_train_val - ",y_train_val, "y_train_val type", type(y_train_val),type(Y_train_init))

    Y_train_init.append(y_train_val.iloc[0])

    

    count +=1



    if count%training_size == 0:

        break



split = int(training_size*0.8)

print("split = ",split)

X_train = X_train_init[:split]

X_test = X_train_init[split:]

Y_train = Y_train_init[:split]

Y_test = Y_train_init[split:]



print('X_train shape:', np.array(X_train).shape)

print(":: y_train shape",np.array(Y_train).shape)
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras import optimizers

from keras import applications

from keras.models import Model



model = Sequential()

#Y_train = np.zeros(10000)

print("X_train, Y-train type - ",type(Y_train),":",type(Y_train))

print("X shapes - ",np.array(X_train).shape)

print("Y shapes - ",np.array(Y_train).shape)

datagen = ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rescale=1./255,

    rotation_range=50,

    width_shift_range=0.4,

    height_shift_range=0.4,

    horizontal_flip=True)







model.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Convolution2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Convolution2D(64, (1, 1)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(1, 1), dim_ordering="tf"))



model.add(Convolution2D(64, (1, 1)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(1, 1), dim_ordering="tf"))



model.add(Convolution2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.4))

model.add(Dense(1))

model.add(Activation('sigmoid'))



datagen.fit(np.array(X_train))

bsize = 32

foo = datagen.flow(np.array(X_train), np.array(Y_train), batch_size=bsize)

#model.compile(loss='binary_crossentropy',

#              optimizer='rmsprop',

#              metrics=['accuracy'])

model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])





#history = model.fit_generator(foo,steps_per_epoch=len((X_train)) / bsize, validation_data=(np.array(X_test),np.array(Y_test)),verbose=1,epochs=40)

history = model.fit(np.array(X_train), np.array(Y_train),validation_split=0.2,batch_size=bsize,epochs=10)
model.evaluate(np.array(X_test),np.array(Y_test))



import matplotlib.pyplot as plt

# list all data in history

print(history.history.keys())



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