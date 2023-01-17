from keras import models

from keras import layers

from keras import optimizers

 

# Create the model

model = models.Sequential()

model.add(layers.Conv2D(64, (11,11), activation='relu', input_shape=(128, 128, 3))) 

model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))

          

model.add(layers.Conv2D(128, (11,11), activation='relu')) 

model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))

          

# Add new layers

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(6, activation='softmax'))



# Show a summary of the model. Check the number of trainable parameters

model.summary()
import os

base_dir = '/kaggle/input/duth-cv-2019-2020-hw-4/vehicles'

train_dir = os.path.join(base_dir, 'train')

validation_dir = os.path.join(base_dir, 'val')
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1./255)

val_datagen  = ImageDataGenerator(rescale=1./255)



# --------------------

# Flow training images in batches of 20 using train_datagen generator

# --------------------

train_generator = train_datagen.flow_from_directory(train_dir,

                                                    batch_size=20,

                                                    class_mode='categorical',

                                                    # color_mode='grayscale',

                                                    target_size=(128,128),

                                                    shuffle=True)     

# --------------------

# Flow validation images in batches of 20 using test_datagen generator

# --------------------

validation_generator =  val_datagen.flow_from_directory(validation_dir,

                                                        batch_size=20,

                                                        class_mode='categorical',

                                                        #  color_mode='grayscale',

                                                         target_size=(128,128)) 
# Compile the model

model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])

# Train the model

history = model.fit_generator(

      train_generator,

      steps_per_epoch=train_generator.samples/train_generator.batch_size ,

      epochs=1,

      validation_data=validation_generator,

      validation_steps=validation_generator.samples/validation_generator.batch_size,

      verbose=1)

 

# Save the model

model.save('small_last4.h5')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import tensorflow as tf

from keras.preprocessing import image

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import csv



model = tf.keras.models.load_model('/kaggle/working/small_last4.h5')

rowlist = [['Id', 'Category']]



for dirname, _, filenames in os.walk('/kaggle/input/duth-cv-2019-2020-hw-4/vehicles/test'):

    for filename in filenames:

        path = os.path.join(dirname, filename)

        img = image.load_img(path, target_size=(128, 128), grayscale=False, interpolation='bilinear')

        

        x = image.img_to_array(img)

        x = np.expand_dims(x, axis=0)

        

        classes_pred = model.predict(x)

        cls_pred = np.argmax(classes_pred)

        rowlist.append([filename, cls_pred])

        #print(filename, cls_pred)

        with open('output.csv', 'w', newline='') as file:

            writer = csv.writer(file)

            writer.writerows(rowlist)

        