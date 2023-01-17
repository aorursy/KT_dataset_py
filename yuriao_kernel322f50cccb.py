## course work: https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/supplement/4ijuU/the-cats-vs-dogs-dataset



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import random

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



model = tf.keras.models.Sequential([

# Note the input shape is the desired size of the image 150x150 with 3 bytes color

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2), 

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 

    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(), 

    # 512 neuron hidden layer

    tf.keras.layers.Dense(512, activation='relu'), 

    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')

    tf.keras.layers.Dense(1, activation='sigmoid')  

])



model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])



TRAINING_DIR = '../input/train' 

train_datagen = ImageDataGenerator(rescale=1.0/225.,rotation_range=40, width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2, zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,batch_size=20,class_mode='binary',target_size=(150, 150)) 

VALIDATION_DIR = '../input/test1' 

validation_datagen = ImageDataGenerator( rescale = 1.0/255.,rotation_range=40, width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2, zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,batch_size=20,class_mode  = 'binary',target_size = (150, 150))



history = model.fit_generator(train_generator,epochs=5,verbose=1,validation_data=validation_generator)



count=0;

classes=np.array(['id','label'])

for i in os.listdir('../input/test1/test1/'):

    img = image.load_img('../input/test1/test1/'+i, target_size=(150,150))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    images=np.vstack([x])

    resultt = model.predict(images)

    if resultt>0.5:

        print('1')

        classes = np.vstack((classes, [count+1,1]))

    else:

        print('0')

        classes = np.vstack((classes, [count+1,0]))

    

    count=count+1;



    output=pd.DataFrame(data=classes[1:,1:],index=classes[1:,0],columns=classes[0,1:])  # 1st row as the column names

    

    output.to_csv('../sampleSubmission.csv')

# Any results you write to the current directory are saved as output.
output