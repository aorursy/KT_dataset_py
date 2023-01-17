# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
Height = 100

Width = 100

Channel = 3

Batch_size = 32
train_dir = "/kaggle/input/fruits/fruits-360_dataset/fruits-360/Training"

test_dir = "/kaggle/input/fruits/fruits-360_dataset/fruits-360/Test"



 

train_gen = ImageDataGenerator(rescale = 1./255,

                               validation_split = 0.2,

                               shear_range=0.2,

                               zoom_range=0.2,

                               horizontal_flip=True)



train_datagen = train_gen.flow_from_directory(train_dir,

                                              target_size = (Height,Width),

                                              batch_size = Batch_size,

                                              class_mode = 'categorical')

validation_generator = train_gen.flow_from_directory(train_dir, # same directory as training data

                                                         target_size=(Height,Width),

                                                         batch_size=Batch_size,

                                                         class_mode='categorical',

                                                         subset='validation') 



sample_image,_ = next(train_datagen)
def ImagePlot(arr):

    fig,axis = plt.subplots(1,5,figsize = (20,20))

    axis = axis.flatten()

    for img,ax in zip(arr,axis):

        ax.imshow(img)

        ax.axis('off')

    plt.tight_layout()

    plt.show()
ImagePlot(sample_image[:5])
#building the network

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout

#This is basically -1

AUTO = tf.data.experimental.AUTOTUNE

AUTO
model = Sequential()

model.add(Conv2D(filters = 16,kernel_size = (5,5),strides = (1,1),padding = 'same',activation = 'relu',input_shape = (Height,Width,Channel)))

model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2),padding = 'valid'))

model.add(Conv2D(filters = 32,kernel_size = (5,5),strides = (1,1),padding = 'same',activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2),padding = 'same'))

model.add(Conv2D(filters = 64,kernel_size = (2,2), strides = (1,1),padding = 'same',activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2),padding = 'same'))

model.add(Conv2D(filters = 128,kernel_size = (2,2), strides = (1,1),padding = 'same',activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2),strides = (2,2),padding = 'same'))

model.add(Flatten())

model.add(Dense(1024,activation = 'relu'))

model.add(Dense(256,activation = 'relu'))

model.add(Dense(120,activation = 'softmax'))
model.summary()
model.compile(optimizer = 'rmsprop',

             loss = 'categorical_crossentropy',

             metrics=['accuracy'])
history = model.fit_generator(train_datagen,

                  steps_per_epoch = train_datagen.samples // Batch_size,

                  epochs = 10,

                  validation_data = validation_generator,

                  validation_steps = validation_generator.samples // Batch_size,

                  )
model.save('fruits_360 classifier.h5')
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(30)



plt.figure(figsize = (8,8))

plt.subplot(1,2,1)

plt.plot(epochs_range,acc,label = 'Training Accuracy')

plt.plot(epochs_range,val_acc,label = 'Validation Accuracy')

plt.legend(loc = 'lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1,2,2)

plt.plot(epochs_range,loss,label = 'Training loss')

plt.plot(epochs_range,val_loss,label = 'Validation loss')

plt.legend(loc = 'upper right')

plt.title('Training and Validation Accuracy')

plt.show()