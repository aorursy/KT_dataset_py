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
print(len(os.listdir("/kaggle/input/dogs-cats-images/dog vs cat/dataset/test_set/cats")))

print(len(os.listdir("/kaggle/input/dogs-cats-images/dog vs cat/dataset/test_set/dogs")))

print(len(os.listdir("/kaggle/input/dogs-cats-images/dog vs cat/dataset/training_set/cats")))

print(len(os.listdir("/kaggle/input/dogs-cats-images/dog vs cat/dataset/training_set/dogs")))
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagenerator=ImageDataGenerator(rescale=1.0/255)
training_data=datagenerator.flow_from_directory(directory='/kaggle/input/dogs-cats-images/dog vs cat/dataset/training_set',

                                 batch_size=32,

                                target_size=(100,100),

                                 class_mode='binary'

                                )

testing_data=datagenerator.flow_from_directory(directory='/kaggle/input/dogs-cats-images/dog vs cat/dataset/test_set',

                                 batch_size=32,

                                target_size=(100,100),

                                 class_mode='binary'

                                )
model=tf.keras.Sequential([tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=training_data.image_shape),

                          tf.keras.layers.MaxPooling2D(2,2),

                          tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

                          tf.keras.layers.MaxPooling2D(2,2), 

                          tf.keras.layers.Conv2D(128,(3,3),activation='relu'),

                          tf.keras.layers.MaxPooling2D(2,2), 

                           tf.keras.layers.Flatten(),

                           tf.keras.layers.Dense(100,activation='relu'),

                           tf.keras.layers.Dense(1,activation='sigmoid')

                          ])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
fitted_model = model.fit_generator(training_data,

                        epochs = 15,

                        validation_data = testing_data

                        )
from PIL import Image

 

im = Image.open("/kaggle/input/dogs-cats-images/dog vs cat/dataset/test_set/cats/cat.4744.jpg")

im.show()