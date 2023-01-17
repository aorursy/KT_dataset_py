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
train_Data = pd.read_csv('../input/Kannada-MNIST/train.csv')
dig_mnist  =  pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')
test_data  = pd.read_csv('../input/Kannada-MNIST/test.csv')
train_Data.head() #validation set :)

import seaborn as sns

g = sns.countplot(train_Data['label'])
g = sns.countplot(dig_mnist['label'])
train = train_Data[:]
val   = dig_mnist[:]
train_label = np.float32(train.label)
val_label = np.float32(val.label)
train_image = np.float32(train[train.columns[1:]])
val_image = np.float32(val[val.columns[1:]])
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range = 10,
    horizontal_flip = False,
    zoom_range = 0.15)
print(train_label.shape)
print(val_label.shape)
train_label = to_categorical(train_label) 
val_label = to_categorical(val_label)
print(train_label.shape)
print(val_label.shape)
train_image = train_image/255.0 
val_image = val_image/255.0
train_image = train_image.reshape(train_image.shape[0],28,28,1)
val_image = val_image.reshape(val_image.shape[0],28,28,1)
print(train_image.shape)
print(val_image.shape)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,Activation
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint,History,EarlyStopping,LearningRateScheduler
from keras.layers import LeakyReLU
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm

model=Sequential()
model.add(Conv2D(28,(5,5),padding='same',input_shape=(28,28,1)))
model.add(Conv2D(28,(5,5)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(28,(5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32,(5,5),padding='same',input_shape=(28,28,1)))
model.add(Conv2D(32,(5,5)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.45))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()
from keras.optimizers import Adam, Adadelta, RMSprop, SGD
from keras import optimizers
model.compile(loss = 'categorical_crossentropy' , optimizer= optimizers.SGD(lr=0.001 , momentum=0.9) , metrics=['accuracy'])
# training
checkpoint = ModelCheckpoint("custom.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
history = model.fit_generator(datagen.flow(train_image,train_label, batch_size=256),
                              epochs = 20,
                              shuffle=True,
                              validation_data = (val_image,val_label),
                              verbose = 1,
                              steps_per_epoch=117,callbacks= [checkpoint ,early])


test = test_data[:]
test_id = (test.id)
test_image = np.float32(test[test.columns[1:]])
test_image = test_image/255.0 
test_image = test_image.reshape(test_image.shape[0],28,28,1)
test_image.shape
prediction = model.predict_classes(test_image)
submission = pd.DataFrame({'id':test_id,'label':prediction})
print(submission.head())
filename = 'submission.csv'
submission.to_csv(filename,header=True,index=False)
output = pd.read_csv('submission.csv')
output.head()

!cp /kaggle/working/submission.csv /Kannada-MNIST/