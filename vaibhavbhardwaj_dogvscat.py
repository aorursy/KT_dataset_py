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
%reload_ext autoreload
%autoreload 2
%matplotlib inline
PATH = "/kaggle/input"
sz=224
batch_size=64
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense
from keras.applications import ResNet50
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.resnet50 import preprocess_input
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
train_datagen = ImageDataGenerator(
    rotation_range=30,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
test_datagen=ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory("/kaggle/input/data/train/",
    target_size=(sz, sz),
    batch_size=batch_size, class_mode='binary')
test_set = test_datagen.flow_from_directory("/kaggle/input/data/validation/",
                                            target_size=(sz, sz),
    batch_size=batch_size, class_mode='binary')
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(sz,sz,3), activation = 'relu' ))

#pooling
model.add(MaxPooling2D(pool_size=(2,2)))
#adding 2nd layer
model.add(Conv2D(32, (3, 3), input_shape=(sz,sz,3), activation = 'relu' ))
model.add(MaxPooling2D(pool_size=(2,2)))
#flatenning 
model.add(Flatten())
#adding connection
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

#compiling CNN
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
plot_model(model, to_file='/tmp/model_plot.png', show_shapes=True, show_layer_names=True)

#Image('/tmp/model_plot.png')
#I = imread ();
img = plt.imread('/tmp/model_plot.png')
plt.imshow(img)
plt.show()
model.summary()


#fitting model to data
model.fit_generator(train_generator, train_generator.n // batch_size, epochs=10, workers=4,
                   validation_data=test_set, validation_steps=test_set.n // batch_size)


model.save_weights('/tmp/cat_dog_1.h5')
img = image.load_img('/kaggle/input/data/evaluation/dog/105.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
print (classes)
eval_set = test_datagen.flow_from_directory("/kaggle/input/data/evaluation/",
                                            target_size=(sz, sz),
    batch_size=batch_size, class_mode='binary')


model.predict_classes(eval_set)
model.load_weights('/tmp/cat_dog_1.h5')
img = image.load_img('/kaggle/input/data/evaluation/cat/114.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
print (classes)
