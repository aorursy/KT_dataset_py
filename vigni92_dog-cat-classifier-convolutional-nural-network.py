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
import keras

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense



my_cnn = Sequential()



my_cnn.add(Conv2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'))



my_cnn.add(MaxPooling2D(pool_size = (2,2)))



my_cnn.add(Conv2D(32,(3,3), activation = 'relu'))



my_cnn.add(MaxPooling2D(pool_size=(2,2)))



my_cnn.add(Flatten())



my_cnn.add(Dense(units=128, kernel_initializer = 'uniform', activation = 'relu'))



my_cnn.add(Dense(units = 96, kernel_initializer = 'uniform', activation = 'relu'))



my_cnn.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



my_cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



from keras.preprocessing.image import ImageDataGenerator



train_model = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)



test_model = ImageDataGenerator(rescale=1./255)



train_set = train_model.flow_from_directory('../input/dataset/dataset/training_set', target_size=(64,64), batch_size=32, class_mode='binary')



test_set = test_model.flow_from_directory('../input/dataset/dataset/test_set', target_size=(64,64), batch_size=32, class_mode='binary')



my_cnn.fit_generator(train_set, steps_per_epoch=8000, epochs=2, validation_data=test_set, validation_steps=2000)