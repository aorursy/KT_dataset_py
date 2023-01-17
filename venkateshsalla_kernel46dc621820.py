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
from keras.layers import Input, Lambda,Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
# Resize all images to this
IMAGE_SIZE=[224,224]
train_path='../input/horses-or-humans-dataset/null'
valid_path='../input/horses-or-humans-dataset/null'
#Add preprocessing layer to the front of VGG
vgg=VGG16(input_shape=IMAGE_SIZE+[3],weights='imagenet',include_top=False)
for layer in vgg.layers:
    layer.trainable=False
#This is useful for getting number of classes
folders=glob('../input/horses-or-humans-dataset/horse-or-human/train/*')
#Layers
x=Flatten()(vgg.output)
prediction=Dense(len(folders),activation='softmax')(x)
#create a model object
model=Model(inputs=vgg.input,outputs=prediction)
#View the structure of model
model.summary()
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow_from_directory('../input/horses-or-humans-dataset/horse-or-human/train',
                                              target_size=(224,224),
                                              batch_size=32,
                                              class_mode='categorical')
test_set=test_datagen.flow_from_directory('../input/horses-or-humans-dataset/horse-or-human/validation',
                                              target_size=(224,224),
                                              batch_size=32,
                                              class_mode='categorical')
#Fit the model
r=model.fit_generator(training_set,validation_data=test_set,epochs=5,steps_per_epoch=len(training_set),
                     validation_steps=len(test_set))
# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')
# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
#Saving the model
import tensorflow as tf
from keras.models import load_model
model.save('facefeatures_new_model.h5')



