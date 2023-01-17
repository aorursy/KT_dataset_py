# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.resnet50 import ResNet50

from keras.models import Sequential, Model

from keras.layers import Input, Flatten, Dense

from keras import optimizers
train_data_dir = '../input/1056lab-archit-heritage-elem-recognit/train'

generator = ImageDataGenerator(width_shift_range = 0.3,

                                     height_shift_range = 0.3,

                                     rotation_range = 30, 

                                     horizontal_flip=True,

                                     vertical_flip=True,

                                     validation_split = 0.2)
train_generator = generator.flow_from_directory(train_data_dir,

                                               target_size=(128,128),

                                               color_mode='rgb',

                                               batch_size=16,

                                               class_mode='categorical',

                                               shuffle=True)
input_tensor = Input(shape=(128,128,3))

ResNet50 = ResNet50(include_top = False,weights='imagenet',input_tensor=input_tensor)
top_model = Sequential()

top_model.add(Flatten(input_shape=ResNet50.output_shape[1:]))

top_model.add(Dense(128,activation='relu'))

top_model.add(Dense(64,activation='relu'))

top_model.add(Dense(10,activation='sigmoid'))
model = Model(ResNet50.input, top_model(ResNet50.output))
model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(),metrics = ['accuracy'])
history = model.fit_generator(

    train_generator,

    steps_per_epoch=447,

    epochs=40)
import glob

from keras.preprocessing.image import load_img, img_to_array
img_array_list = []

test_dir = '../input/1056lab-archit-heritage-elem-recognit/test'

img_list = glob.glob(test_dir + '/*.jpg')

img_list.sort()

for i in img_list:

    img = load_img(i, color_mode='rgb', target_size=(128,128,3))

    img_array_list.append(img_to_array(img))



X_test = np.array(img_array_list)
p = model.predict(X_test)
sample = pd.read_csv('../input/1056lab-archit-heritage-elem-recognit/sampleSubmission.csv')

sample['class'] = np.argmax(p,axis=1)

sample.to_csv('submission.csv',index = False)