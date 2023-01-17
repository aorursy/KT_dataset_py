# code to get the train,test files 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# let's get the path of the train and test files ( since our train and test folders contain several files we
# need to figure out the different classes available in our data)

import pathlib
PATH = '../input/intel-image-classification/seg_train/seg_train'
data_dir = pathlib.Path(PATH)
classes=np.array([item.name for item in data_dir.glob('*') if item.name!='LICENSE.txt'])
print(classes)

test_path = '../input/intel-image-classification/seg_test/seg_test'
test_data = pathlib.Path(test_path)

# now we have the path for our train images and test images
# Since our dataset is small we need to create some more tranining data 
# We will perform Data Augmentation for this ( basically we will take our train images and make more images from them
# by zooming in-out,  scaling them , moving horizontally and vertically.)
# this is done using the ImageDataGenerator

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IDG=ImageDataGenerator(rescale=1./255,width_shift_range=.1,height_shift_range=.1,zoom_range=.1,validation_split=.2,rotation_range=10)

train=IDG.flow_from_directory(PATH,subset='training',target_size=(150,150),classes=list(classes),batch_size=64)

# here we have created training data in batches of 64 images
# we followed the same procedure with our validation data

validation=IDG.flow_from_directory(PATH,subset='validation',batch_size=64,target_size=(150,150),classes=list(classes))
# we did not perform image Augmentation on our test data because we want it keep it intact
from tensorflow.keras.preprocessing.image import ImageDataGenerator
IDG_test=ImageDataGenerator(rescale=1./255)
test=IDG_test.flow_from_directory(test_path)
# let's take a look at some of the images in our dataset
from matplotlib import pyplot as plt
image_batch,batch_lable=next(train)
def show_images(image_batch,batch_lable):
    plt.figure(figsize=(20,20))
    for i in range(35):
        plt.subplot(7,5,i+1)
        plt.imshow(image_batch[i])
        plt.title(classes[batch_lable[i].argmax()])
        plt.axis('off')
show_images(image_batch,batch_lable)        

# we have shown 35 images from our dataset along with there labels
# let's download all the required libraries and functions

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
# let's define our model now .

model=Sequential([])
model.add(Conv2D(32,(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dropout(.5))
model.add(Dense(100,activation='relu'))
model.add(Dropout(.5))
model.add(Dense(6,activation='softmax'))

model.summary()
import tensorflow as tf
lo=tf.keras.losses.categorical_crossentropy
model.compile(optimizer='adam',loss=lo,metrics=['accuracy'])

history=model.fit_generator(train,epochs=30,steps_per_epoch=train.samples//64,validation_data=validation,validation_steps=validation.samples//64)