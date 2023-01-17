## based on: https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/supplement/4ijuU/the-cats-vs-dogs-dataset

## alsobased on: kernel "Getting started with CNN and VGG16"



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



# define network

model = tf.keras.models.Sequential([

# Note the input shape is the desired size of the image 150x150 with 3 bytes color

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2), 

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(256, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    #Flatten layer: 

    tf.keras.layers.Flatten(),

    # 512 neuron fully connected layer

    tf.keras.layers.Dense(512, activation='relu'), 

    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('non catcus') and 1 for the other ('catcus')

    tf.keras.layers.Dense(1, activation='sigmoid')  

])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
# define training, validation, test materials

TRAINING_DIR='../input/train/train/' 

TRAINING_LABEL_DIR='../input/train.csv'

TESTING_DIR = '../input/test/test/' 
# read training labels as dataframe

# second line: shuffle data

all_label=pd.read_csv(TRAINING_LABEL_DIR)

msk = np.random.rand(len(all_label)) < 0.8

training_label=all_label.loc[msk]

validation_label=all_label.loc[~msk]
len(training_label)
#train_datagen = ImageDataGenerator(rescale=1.0/225.,rotation_range=90, width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2, zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

#validation_datagen = ImageDataGenerator( rescale = 1.0/255.,rotation_range=90, width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2, zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

train_datagen = ImageDataGenerator(rescale=1.0/225.)

validation_datagen = ImageDataGenerator( rescale = 1.0/255.)
train_generator = train_datagen.flow_from_dataframe (dataframe=training_label,directory=TRAINING_DIR,batch_size=20,x_col='id',y_col='has_cactus',class_mode='other',target_size=(150, 150))

validation_generator = validation_datagen.flow_from_dataframe (dataframe=validation_label,directory=TRAINING_DIR,batch_size=20,x_col='id',y_col='has_cactus',class_mode='other',target_size=(150, 150)) 
history = model.fit_generator(train_generator,epochs=10,verbose=1,validation_data=validation_generator)
count=0;

classes=np.array(['idx','id','has_cactus'])

for i in os.listdir(TESTING_DIR):

    img = image.load_img(TESTING_DIR+i, target_size=(150,150))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    images=np.vstack([x])

    resultt = model.predict(images)

    if resultt>0.5:

        print('1')

        classes = np.vstack((classes, [i,i,1]))

    else:

        print('0')

        classes = np.vstack((classes, [i,i,0]))

    

    count=count+1;



    output=pd.DataFrame(data=classes[1:,1:],index=classes[1:,0],columns=classes[0,1:])  # 1st row as the column names

    

    output.to_csv('sampleSubmission.csv',index=False)



print('fin')

# Any results you write to the current directory are saved as output.
output
