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
from keras.applications.vgg16 import VGG16

from keras.layers import Input

# load model

model = VGG16(include_top=False)

model.summary()
new_input = Input(shape=(640, 480, 3))

model = VGG16(include_top=False, input_tensor=new_input, pooling='avg')# summarize the model

model.summary()
from keras.applications.resnet50 import ResNet50
new_input = Input(shape=(240, 240, 3))

model1 = ResNet50(weights=None, input_tensor=new_input, classes=10)

model1.summary()
from keras.applications.inception_v3 import InceptionV3

new_input = Input(shape=(224, 224, 3))

modelv3 = InceptionV3(weights=None, input_tensor=new_input, classes=5)

modelv3.summary()
from keras.applications.inception_v3 import InceptionV3

# load model

model2 = InceptionV3()

# summarize the model

model2.summary()
# example of loading the resnet50 model

from keras.applications.resnet50 import ResNet50

# load model

model3 = ResNet50()

# summarize the model

model3.summary()
modelv = VGG16()

modelv.summary()
from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.applications.vgg16 import preprocess_input

from keras.applications.vgg16 import decode_predictions

# load an image from file

image = load_img('/kaggle/input/goldfish/gold-fish.jpg', target_size=(300, 300))

# convert the image pixels to a numpy array

image = img_to_array(image)

# reshape data for the model

image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# prepare the image for the VGG model

image = preprocess_input(image)
# predict the probability across all output classes

yhat = model.predict(image)

# convert the probabilities to class labels

label = decode_predictions(yhat)

# retrieve the most likely result, e.g. highest probability

label
from keras.applications.vgg16 import VGG16

from keras.models import Model

from keras.layers import Dense

from keras.layers import Flatten

# load model without classifier layers

model = VGG16(include_top=False, input_shape=(300, 300, 3))

# add new classifier layers

flat1 = Flatten()(model.outputs)

class1 = Dense(1024, activation='relu')(flat1)

output = Dense(10, activation='softmax')(class1)

# define new model

model = Model(inputs=model.inputs, outputs=output)

# summarize

model.summary()
model = VGG16(include_top=False, input_shape=(300, 300, 3))

# mark loaded layers as not trainable

for layer in model.layers:

    print(layer)

    layer.trainable = False
model.summary()
# load model without classifier layers

model = VGG16(include_top=False, input_shape=(300, 300, 3))

# mark some layers as not trainable

model.get_layer('block4_conv1').trainable = False

model.get_layer('block4_conv2').trainable = False

model.get_layer('block4_conv3').trainable = False

model.get_layer('block5_conv1').trainable = False
model.summary()
from keras.applications.resnet import ResNet50,ResNet101, ResNet152, preprocess_input

model2 = ResNet50()

model2.summary()
model2.layers.pop()

model2.layers.pop()

model2.summary()
from keras.layers import Dense,Flatten,Conv2D

from keras.models import Model
new_layer = Dense(20, activation='softmax', name='my_dense')



inp = model2.input

out = new_layer(model2.layers[-1].output)



model4 = Model(inp, out)

model4.summary()
model2.summary()
for layer in model2.layers:

    layer.trainable = False

flat1 = model2.input

conv = Conv2D(filters=1024, kernel_size=(4,4), strides=(1,1),input_shape=(7,7,2048), activation = "relu")

new_layer = Dense(20, activation='softmax', name='my_dense')



out = conv(model2.layers[-1].output)

inp = model2.input



model4 = Model(inp, out)

out = new_layer(model4.layers[-1].output)

model4 = Model(inp, out)

model4.summary()

new_layer = Dense(20, activation='softmax', name='my_dense')



inp = model2.input

out = new_layer(model2.layers[-1].output)



model4 = Model(inp, out)

model4.summary()