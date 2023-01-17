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
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
## Importing the libraries
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
# resizing all images

IMAGE_SIZE = [224,224]

train_path = '../input/10-monkey-species/training/training'
valid_path = '../input/10-monkey-species/validation/validation'
train_path
valid_path
## Using imagenet pretrained weights
inception = InceptionV3(input_shape= IMAGE_SIZE + [3], weights='imagenet', include_top= False)
#not training the existing weights

for layer in inception.layers:
    layer.trainable = False
# getting number of folders which are output classes
folders = glob('../input/10-monkey-species/training/training/*')
folders
len(folders)
x = Flatten()(inception.output)
prediction = Dense(len(folders),activation='softmax')(x)

#creating a model object

model = Model(inputs = inception.input, outputs= prediction)
model.summary()
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
## Using image data generator to import the images from the dataset

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('../input/10-monkey-species/training/training',
                                                 target_size=(224,224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('../input/10-monkey-species/validation/validation',
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode='categorical')
## Fitting the model

r = model.fit_generator(
    training_set,
    validation_data = test_set,
    epochs=25,
    steps_per_epoch = len(training_set),
    validation_steps = len(test_set)
)

# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
y_pred = model.predict(test_set)
y_pred
y_pred = np.argmax(y_pred, axis=1)
y_pred.shape
y_pred = pd.Series(y_pred)
y_pred
common_name_dict = {0:'mantled_howler',1:'patas_monkey',2:'bald_uakari',3:'japanese_macaque',
       4:'pygmy_marmoset',5:'white_headed_capuchin',6:'silvery_marmoset',7:'common_squirrel_monkey',
       8:'black_headed_night_monkey',9:'nilgiri_langur'}
y_pred = y_pred.map(common_name_dict)
y_pred