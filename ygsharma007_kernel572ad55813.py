# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, Activation, BatchNormalization
from keras import backend as K

import matplotlib.pyplot as plt
print(os.listdir("../input"))

# dimension of our images
img_width,img_height = 150,150
filenames = os.listdir("/kaggle/input/dogs-vs-cats/train/train/")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)
        
df = pd.DataFrame({
    'filename' : filenames,
    'category' : categories
})
    
df.head()
df.tail()
df['category'].value_counts().plot.bar()
train_data_path = '/kaggle/input/dogs-vs-cats/train/train/'
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation = 'softmax'))
 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
df['category'] = df['category'].replace({0:'cat', 1: 'dog'})
from sklearn.model_selection import train_test_split

# Splitting the data into train and validation dataset
train_df, validate_df = train_test_split(df, test_size=0.20, random_state = 42)
train_df.head()
# Removing the old indexes
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
train_df.head()
nb_train_samples = train_df.shape[0]
nb_validation_samples = validate_df.shape[0]
epochs = 20
batch_size = 15
# augmentation configuration for training
train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True) 

# augmentation configuration for testing
test_datagen = ImageDataGenerator(rescale = 1. / 255)
# Training Data Generator
train_generator = train_datagen.flow_from_dataframe(
            train_df,
            train_data_path,
            x_col = 'filename',
            y_col = 'category',
            target_size = (img_width, img_height),
            batch_size = batch_size,
            class_mode = 'categorical')

# Validation Data Generator
validation_generator = test_datagen.flow_from_dataframe(
            validate_df,
            train_data_path,
            x_col = 'filename',
            y_col = 'category',
            target_size = (img_width, img_height),
            batch_size = batch_size,
            class_mode = 'categorical')
model.fit_generator(
        train_generator,
        steps_per_epoch = nb_train_samples // batch_size,
        epochs = epochs,
        validation_data = validation_generator,
        validation_steps = nb_validation_samples // batch_size)
