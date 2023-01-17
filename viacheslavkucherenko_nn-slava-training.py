!pip install ../input/kerasapplications/keras-team-keras-applications-3b180cb -f ./ --no-index
import json
import time

import keras
import tensorflow
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import (
    Dense,Conv3D, Dropout, Activation, Flatten, Input, BatchNormalization, GlobalAveragePooling2D, Add, Conv2D, AveragePooling2D, 
    LeakyReLU, Concatenate 
)

from keras import regularizers
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from keras.utils import to_categorical
from keras.optimizers import SGD

from keras.utils.vis_utils import plot_model
from keras_preprocessing.image import ImageDataGenerator

from keras.models import Model
from keras import backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M
from keras.applications.inception_v3 import InceptionV3

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

train_root='../input/indoor-scenes-cvpr-2019/indoorCVPR_09/Images/'

IM_WIDTH=299
IM_HEIGHT=299
EPOCH=100
batch_size=128
NB_CLASS=67

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    rescale=1./255)

train_generator = datagen.flow_from_directory(
  train_root,
  target_size=(IM_WIDTH, IM_HEIGHT),
  batch_size=batch_size,
  class_mode='categorical', subset='training')

vaild_generator = datagen.flow_from_directory(
  train_root,
  target_size=(IM_WIDTH, IM_HEIGHT),
  batch_size=batch_size,
  class_mode='categorical', subset='validation')
base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.01)(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(NB_CLASS, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
print(model.summary())
#compiling
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

# training
model.fit_generator(train_generator,
                    epochs=EPOCH,
                    validation_data=vaild_generator,
                    steps_per_epoch=train_generator.n/batch_size,
                    shuffle=True,
                    validation_steps=vaild_generator.n/batch_size,
                    verbose=1)
model.save('slava_model.h5')
model.save_weights("slava_weigths.h5")
history=model.history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("slava_model_result.png")
plt.show()