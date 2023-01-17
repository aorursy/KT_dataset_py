# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path='../input/gtzan-dataset-music-genre-classification/Data/images_original'
data_gen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
image_datagen=data_gen.flow_from_directory(path,target_size=(300,300),batch_size=32,class_mode='categorical')
from tensorflow.keras.applications import VGG16
model=VGG16(include_top=False,input_shape=(300,300,3))
for layer in model.layers:

    layer.trainable=False
output=model.layers[-1].output

model_final=tf.keras.layers.Flatten()(output)

model_final=tf.keras.layers.Dense(512,activation='relu')(model_final)

model_final=tf.keras.layers.Dense(64,activation='relu')(model_final)

model_final=tf.keras.layers.Dense(10,activation='softmax')(model_final)
model=tf.keras.models.Model(model.input,model_final)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit_generator(image_datagen,epochs=10)