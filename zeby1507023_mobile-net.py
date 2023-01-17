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
from glob import glob

import os

import pandas as pd

from skimage.io import imread

from scipy.ndimage import zoom

import matplotlib.pyplot as plt

    

from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical

import numpy as np
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential

from keras.models import Model

from keras.layers import Conv2D, Dense, BatchNormalization, Flatten

from keras.optimizers import Adam

from keras.applications.mobilenet_v2 import MobileNetV2
import os

print(os.listdir('../input/chest-xray-pneumonia/chest_xray/chest_xray'))
IMAGE_SIZE =[224,224]

Train ="../input/chest-xray-pneumonia/chest_xray/chest_xray/train"

Test ="../input/chest-xray-pneumonia/chest_xray/chest_xray/test"

Val = "../input/chest-xray-pneumonia/chest_xray/chest_xray/val"
model = MobileNetV2(input_shape=(224,224,3),include_top=False, weights='imagenet')
model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
model.summary()
x = Flatten()(model.output)

predict=Dense(2,activation='softmax')(x)

model_1=Model(inputs=model.input,outputs=predict)
model_1.summary()
for layer in model.layers:

    layer.trainable = False
from keras.utils import plot_model

plot_model(model_1, to_file='model.png',show_layer_names =True,show_shapes = True)
from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.mobilenet import preprocess_input
img1 = "../input/chest-xray-pneumonia/chest_xray/test/NORMAL/IM-0011-0001-0001.jpeg"

img1 = image.load_img(img1, target_size=(224, 224))

x1 = image.img_to_array(img1)

x1 = np.expand_dims(x1, axis=0)

x1 = preprocess_input(x1)



features1 = model_1.predict(x1)
print(features1)
from keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator( 

    rescale=1. / 255, 

    shear_range=0.2, 

    zoom_range=0.2, 

    horizontal_flip=True) 

  

test_datagen = ImageDataGenerator(rescale=1. / 255) 
train_generator = train_datagen.flow_from_directory( 

    Train, 

    target_size=(224, 224), 

    batch_size=64, 

    class_mode='categorical') 

  

validation_generator = test_datagen.flow_from_directory( 

    Test, 

    target_size=(224, 224), 

    batch_size=32, 

    class_mode='categorical') 
model_1.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
model_1.save_weights('mobile.h5')
r = model_1.fit_generator(train_generator,

                       validation_data = validation_generator,

                       epochs =10,

                       steps_per_epoch = 100,

                       validation_steps = 25)
img1 = "../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/person80_bacteria_389.jpeg"

img1 = image.load_img(img1, target_size=(224, 224))

x1 = image.img_to_array(img1)

x1 = np.expand_dims(x1, axis=0)

x1 = preprocess_input(x1)



features1 = model_1.predict(x1)
print(features1)