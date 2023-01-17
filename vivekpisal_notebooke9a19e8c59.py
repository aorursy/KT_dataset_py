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
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Lambda,Flatten,Dense,Input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
IMG_SIZE=[224,224]

train_path="/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train"
test_path="/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/test"
vgg16=VGG16(input_shape=IMG_SIZE +[3] ,weights='imagenet',include_top=False)
for layer in vgg16.layers:
    layer.trainable=False
vgg16.summary()
len(os.listdir(train_path))
x=Flatten()(resnet.output)
prediction=Dense(len(os.listdir(train_path)),activation='softmax')(x)

model=Model(inputs=vgg16.input,outputs=prediction)
model.summary()
model.compile(
loss='categorical_crossentropy',
optimizer='adam',
metrics=['accuracy'])
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.20,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow_from_directory(train_path,target_size=(224,224),batch_size=32,class_mode='categorical')
test_set=test_datagen.flow_from_directory(test_path,target_size=(224,224),batch_size=32,class_mode='categorical')
r=model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=20,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)
y_pred=model.predict(test_set)
y_pred=np.argmax(y_pred,axis=1)
y_pred
from tensorflow.keras.models import load_model

model.save('vgg16.h5')
MODEL_PATH ='vgg16.h5'

# Load your trained model
model2=load_model(MODEL_PATH)
model2.summary()
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(224,224))
    # convert to array
    img = img_to_array(img)
    img=img/255
    x=np.expand_dims(img,axis=0)
    img_data=preprocess_input(x)
    return img_data

# load an image and predict the class
	# load the image
img = load_image('/kaggle/input/covid19-xray-dataset-train-test-sets/xray_dataset_covid19/train/NORMAL/IM-0049-0001.jpeg')
# predict the class
a=np.argmax(model.predict(img), axis=1)
print(a,model.predict(img))
