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
from keras.layers import Input,Lambda,Dense,Flatten

from keras.models import Model

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from glob import glob

import matplotlib.pyplot as plt
IMAGE_SIZE =[224,224]

Train ="../input/chest-xray-pneumonia/chest_xray/chest_xray/train"

Test ="../input/chest-xray-pneumonia/chest_xray/chest_xray/test"
vgg = VGG16(input_shape =IMAGE_SIZE+[3],weights='imagenet',include_top = False )
for layer in vgg.layers:

    layer.trainable = False
folder = glob("../input/chest-xray-pneumonia/chest_xray/train")

print(folder)
x = Flatten()(vgg.output)
predition = Dense(2,activation = 'softmax')(x)

model =Model(inputs =vgg.input,outputs =predition)
model.summary()
model.compile(loss='categorical_crossentropy', 

                  optimizer='adam', 

                  metrics=['accuracy'])
from keras.utils import plot_model

plot_model(model, to_file='model.png',show_layer_names =True,show_shapes = True)
img1 = "../input/chest-xray-pneumonia/chest_xray/chest_xray/test/NORMAL/IM-0011-0001-0001.jpeg"

img1 = image.load_img(img1, target_size=(224, 224))

x1 = image.img_to_array(img1)

x1 = np.expand_dims(x1, axis=0)

x1 = preprocess_input(x1)



features1 = model.predict(x1)
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

    batch_size=500, 

    class_mode='categorical') 

  

validation_generator = test_datagen.flow_from_directory( 

    Test, 

    target_size=(224, 224), 

    batch_size=500, 

    class_mode='categorical') 
r = model.fit_generator(train_generator,

                       validation_data = validation_generator,

                       epochs =3,

                       steps_per_epoch = 40,

                       validation_steps = 10)
plt.plot(r.history['accuracy'])

plt.plot(r.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(r.history['loss'])

plt.plot(r.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model.save_weights('vgg16.h5')


import matplotlib.pyplot as plt

img1 = "../input/chest-xray-pneumonia/chest_xray/chest_xray/train/PNEUMONIA/person1162_virus_1949.jpeg"

img1 = image.load_img(img1, target_size=(224, 224))

x1 = image.img_to_array(img1)

x1 = np.expand_dims(x1, axis=0)

x1 = preprocess_input(x1)



features1 = model.predict(x1)
print(features1)
img = "../input/chest-xray-pneumonia/chest_xray/chest_xray/train/NORMAL/NORMAL2-IM-1277-0001-0002.jpeg"

img = image.load_img(img, target_size=(224, 224))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = preprocess_input(x)



features = model.predict(x)
print(features)
from keras.preprocessing.image import ImageDataGenerator

from keras.applications import VGG16

from keras.layers.core import Dropout

from keras.layers.core import Flatten

from keras.layers.core import Dense

from keras.layers import Input

from keras.models import Model

from keras.optimizers import SGD

from sklearn.metrics import classification_report





import matplotlib.pyplot as plt

import numpy as np

import pickle

import os
import numpy as np

import matplotlib.pyplot as plt

import sklearn

from sklearn.metrics import confusion_matrix

#from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

import numpy as np

import sklearn

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from mlxtend.plotting import plot_confusion_matrix
text_img,text_label = next(validation_generator)

text_label = text_label[:,0]

text_label
prediction = model.predict_generator(validation_generator, steps=1, verbose=0)

cm = confusion_matrix(text_label,np.round(prediction[:,0]))

cm_plot_label = ['Normal','Pneumonia']

plot_confusion_matrix(cm,cm_plot_label)
score  = model.evaluate(validation_generator)

print(score[1])
from sklearn.metrics import accuracy_score

accuracy_score(text_label,np.round(prediction[:,0]))