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
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
# Visualize the Image
img = plt.imread('/kaggle/input/intel-image-classification/seg_test/seg_test/mountain/22780.jpg')
plt.imshow(img)
datagen = ImageDataGenerator()
train_data = datagen.flow_from_directory(target_size = (150,150),directory = '/kaggle/input/intel-image-classification/seg_train/seg_train')
valid_data = datagen.flow_from_directory(target_size = (150,150),directory = '/kaggle/input/intel-image-classification/seg_test/seg_test')
model_layers = VGG16(include_top = False , input_shape = (150,150,3))

model_layers.trainable = False
for layers in model_layers.layers:
    layers.trainable = True
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense
model = Sequential()
model.add(model_layers)
model.add(GlobalAveragePooling2D())
model.add(Dense(6,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(train_data,epochs = 10,validation_data = valid_data)

model.save('model.h5')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.plot(history.history['loss'],label='Training Loss')
plt.legend()
plt.show()

plt.plot(history.history['val_accuracy'],label='Validation accuracy')
plt.plot(history.history['accuracy'],label='Training accuracy')
plt.legend()
plt.show()
import pickle 
with open('model.pickle','wb') as f:
    pickle.dump(model,f)
history.history.keys()
