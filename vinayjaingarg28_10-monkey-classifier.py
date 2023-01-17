%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import cv2 
from keras import regularizers
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Flatten,Dropout,BatchNormalization
from keras.optimizers import Adam, SGD, Adadelta
train_dir = '../input/training/training/'
test_dir = '../input/validation/validation'
labels_file_dir = '../input/monkey_labels.txt'

img_size = 224
train_datagen = ImageDataGenerator(
        brightness_range=[0.8,1.0],
        channel_shift_range=40,
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(train_dir, 
                                                    target_size=(img_size, img_size),
                                                    batch_size=10,
                                                    shuffle=True,
                                                    class_mode='categorical')

# Test generator
test_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = test_datagen.flow_from_directory(test_dir, 
                                                  target_size=(img_size, img_size), 
                                                  batch_size=5,
                                                  shuffle=False,
                                                  class_mode='categorical')
model  = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape=(img_size, img_size,3)))
model.add(Conv2D(128, (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))         
    
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))

model.add(Flatten())
# model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
# model.add(Dropout(0.3))
model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.02)))
model.add(Dense(10, activation='softmax'))

model.compile(metrics=['accuracy'], optimizer=Adam(lr=0.0001), loss='categorical_crossentropy')
model.summary()
model.fit_generator(train_generator, steps_per_epoch=110, 
                    epochs=70, validation_data=validation_generator, validation_steps=45)
model.save_weights('monkey-classifier-weights.h5')
model.save('monkey-model.h5')

history = model.history
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
import random
from keras.preprocessing import image


testlist=np.empty([224,224,3])
label=[]
dim = (224,224)
dirlist = os.listdir(test_dir)
for dirs in dirlist:
    imagefile = test_dir +'/' + dirs + '/' + (random.choice(os.listdir(test_dir+'/'+dirs)))
    img = image.load_img(imagefile,target_size = dim)
    img_arr = image.img_to_array(img)
    img_arr/=255.0
    img_arr = np.reshape(img_arr , [1,224,224,3])
    result = model.predict_classes(img_arr)
    plt.show(img)
    print(f'Result={result}, actual={dirs}')
