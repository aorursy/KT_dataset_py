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
import os
import random
import matplotlib.pyplot as plt
china_data_dir = '../input/pulmonary-chest-xray-abnormalities/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png/'
china_data_normal = [china_data_dir+'{}'.format(i) for i in os.listdir(china_data_dir) if '_0.png' in i]
china_data_tb = [china_data_dir+'{}'.format(i) for i in os.listdir(china_data_dir) if '_1.png' in i]
monto_dir = '../input/pulmonary-chest-xray-abnormalities/Montgomery/MontgomerySet/CXR_png/'
monto_data_normal = [monto_dir+'{}'.format(i) for i in os.listdir(monto_dir) if '_0.png' in i]
monto_data_tb = [monto_dir+'{}'.format(i) for i in os.listdir(monto_dir) if '_1.png' in i]
len(china_data_normal), len(china_data_tb), len(monto_data_normal),len(monto_data_tb)
train = china_data_normal[:261] + china_data_tb[:271] #+ monto_data_normal[:65] + monto_data_tb[:43]
val = china_data_normal[261 : 296] + china_data_tb[271:306] #+ monto_data_normal[65:70] + monto_data_tb[43:48]
test = china_data_normal[296:] + china_data_tb[306:] #+ monto_data_normal[70:] + monto_data_tb[48:]
random.shuffle(train)
random.shuffle(val)
random.shuffle(test)
import matplotlib.image as mpimg
for im in train[0:6]:
    img = mpimg.imread(im)
    imgplot = plt.imshow(img)
    plt.show()
nrows = 512
ncols = 512
channels = 3
import cv2
def read_process_image(list_images):
    X=[]
    y=[]
    
    for img in list_images:
        X.append(cv2.resize(cv2.imread(img, cv2.IMREAD_COLOR), (nrows, ncols), interpolation = cv2.INTER_CUBIC))
        if '_0.png' in img:
            y.append(0)
        else:
            y.append(1)
            
    return X, y
X_train, y_train = read_process_image(train)
X_val, y_val = read_process_image(val)
X_test, y_test = read_process_image(test)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
X_test = np.array(X_test)
y_test = np.array(y_test)
import seaborn as sns
sns.countplot(y_train)
sns.countplot(y_val)
sns.countplot(y_test)
X_train.shape, y_train.shape

y_val.shape
n_train = len(X_train)
n_val = len(X_val)
batch_size = 4
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
from keras import regularizers
import keras
model = models.Sequential()
inputs = layers.Input(shape=(512,512,3))
conv_1_a = layers.Conv2D(filters=16, kernel_size = (3,3), strides = (2,2),padding='same', name = 'conv_1_a', kernel_initializer=keras.initializers.he_normal())(inputs)
conv_1_a = layers.normalization.BatchNormalization()(conv_1_a) 
conv_1_a = keras.layers.Activation('relu')(conv_1_a)
conv_1_b = layers.Conv2D(filters=16, kernel_size = (3,3), strides = (2,2),padding='same',name = 'conv_1_b', kernel_initializer=keras.initializers.he_normal())(conv_1_a)
conv_1_b = layers.normalization.BatchNormalization()(conv_1_b)
conv_1_b = keras.layers.Activation('relu')(conv_1_b)
conv_1_c = layers.Conv2D(filters=16, kernel_size = (1,1), strides = (4,4),padding='same',name = 'conv_1_c', kernel_initializer=keras.initializers.he_normal())(inputs)
conv_1_c = layers.normalization.BatchNormalization()(conv_1_c)
conv_1_c = keras.layers.Activation('relu')(conv_1_c)
conv_1_d = layers.Add()([conv_1_b,conv_1_c])
maxpool_1 = layers.MaxPool2D(pool_size = (3,3), strides = (2,2),  name = 'maxpool_1')(conv_1_d)

conv_2_a = layers.Conv2D(filters=32, kernel_size = (3,3), strides = (1,1),padding='same',name = 'conv_2_a', kernel_initializer=keras.initializers.he_normal())(maxpool_1)
conv_2_a = layers.normalization.BatchNormalization()(conv_2_a)
conv_2_a = keras.layers.Activation('relu')(conv_2_a)
conv_2_b = layers.Conv2D(filters=32, kernel_size = (3,3), strides = (1,1),padding='same',name = 'conv_2_b', kernel_initializer=keras.initializers.he_normal())(conv_2_a)
conv_2_b = layers.normalization.BatchNormalization()(conv_2_b)
conv_2_b = keras.layers.Activation('relu')(conv_2_b)
conv_2_c = layers.Conv2D(filters=32, kernel_size = (1,1), strides = (1,1),padding='same',name = 'conv_2_c', kernel_initializer=keras.initializers.he_normal())(maxpool_1)
conv_2_c = layers.normalization.BatchNormalization()(conv_2_c)
conv_2_c = keras.layers.Activation('relu')(conv_2_c)
conv_2_d = layers.Add()([conv_2_b,conv_2_c])
maxpool_2 = layers.MaxPool2D(pool_size = (3,3), strides = (2,2),name = 'maxpool_2')(conv_2_d)

conv_3_a = layers.Conv2D(filters=48, kernel_size = (3,3), strides = (1,1),padding='same',name = 'conv_3_a', kernel_initializer=keras.initializers.he_normal())(maxpool_2)
conv_3_a = layers.normalization.BatchNormalization()(conv_3_a)
conv_3_a = keras.layers.Activation('relu')(conv_3_a)
conv_3_b = layers.Conv2D(filters=48, kernel_size = (3,3), strides = (1,1),padding='same',name = 'conv_3_b', kernel_initializer=keras.initializers.he_normal())(conv_3_a)
conv_3_b = layers.normalization.BatchNormalization()(conv_3_b)
conv_3_b = keras.layers.Activation('relu')(conv_3_b)
conv_3_c = layers.Conv2D(filters=48, kernel_size = (1,1), strides = (1,1), padding='same',name = 'conv_3_c', kernel_initializer=keras.initializers.he_normal())(maxpool_2)
conv_3_c = layers.normalization.BatchNormalization()(conv_3_c)
conv_3_c = keras.layers.Activation('relu')(conv_3_c)
conv_3_d = layers.Add()([conv_3_b,conv_3_c])
maxpool_3 = layers.MaxPool2D(pool_size = (3,3), strides = (2,2),  name = 'maxpool_3')(conv_3_d)

conv_4_a = layers.Conv2D(filters=64, kernel_size = (3,3), strides = (1,1), padding='same',name = 'conv_4_a', kernel_initializer=keras.initializers.he_normal())(maxpool_3)
conv_4_a = layers.normalization.BatchNormalization()(conv_4_a)
conv_4_a = keras.layers.Activation('relu')(conv_4_a)
conv_4_b = layers.Conv2D(filters=64, kernel_size = (3,3), strides = (1,1),padding='same',name = 'conv_4_b', kernel_initializer=keras.initializers.he_normal())(conv_4_a)
conv_4_b = layers.normalization.BatchNormalization()(conv_4_b)
conv_4_b = keras.layers.Activation('relu')(conv_4_b)
conv_4_c = layers.Conv2D(filters=64, kernel_size = (1,1), strides = (1,1),padding='same',name = 'conv_4_c', kernel_initializer=keras.initializers.he_normal())(maxpool_3)
conv_4_c = layers.normalization.BatchNormalization()(conv_4_c)
conv_4_c = keras.layers.Activation('relu')(conv_4_c)
conv_4_d = layers.Add()([conv_4_b,conv_4_c])
maxpool_4 = layers.MaxPool2D(pool_size = (3,3), strides = (2,2), name = 'maxpool_4')(conv_4_d)

conv_5_a = layers.Conv2D(filters=80, kernel_size = (3,3), strides = (1,1), padding='same',name = 'conv_5_a', kernel_initializer=keras.initializers.he_normal())(maxpool_4)
conv_5_a = layers.normalization.BatchNormalization()(conv_5_a)
conv_5_a = keras.layers.Activation('relu')(conv_5_a)
conv_5_b = layers.Conv2D(filters=80, kernel_size = (3,3), strides = (1,1), padding='same',name = 'conv_5_b', kernel_initializer=keras.initializers.he_normal())(conv_5_a)
conv_5_b = layers.normalization.BatchNormalization()(conv_5_b)
conv_5_b = keras.layers.Activation('relu')(conv_5_b)
conv_5_c = layers.Conv2D(filters=80, kernel_size = (1,1), strides = (1,1),padding='same',name = 'conv_5_c', kernel_initializer=keras.initializers.he_normal())(maxpool_4)
conv_5_c = layers.normalization.BatchNormalization()(conv_5_c)
conv_5_c = keras.layers.Activation('relu')(conv_5_c)
conv_5_d = layers.Add()([conv_5_b,conv_5_c])
maxpool_5 = layers.MaxPool2D(pool_size = (3,3), strides = (2,2), name = 'maxpool_5')(conv_5_d)

globalpool = layers.GlobalAveragePooling2D()(maxpool_5)

dense_1 = layers.Dense(64, activation = 'relu', name = 'dense_1', kernel_initializer=keras.initializers.he_normal())(globalpool)
outputs = layers.Dense(1, activation='sigmoid')(dense_1)
model = keras.Model(inputs,outputs)
model.compile(optimizer=keras.optimizers.Adam(lr=1e-5, beta_1=0.9,beta_2=0.999,epsilon=1e-07,), loss='binary_crossentropy', metrics=['acc'])
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255,)
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
history = model.fit(train_generator,
                  steps_per_epoch=n_train//batch_size,
                  epochs = 32,
                  validation_data = val_generator,
                  validation_steps = n_val//batch_size)
model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')
test_datagen = ImageDataGenerator(rescale=1./255)
pred = model.predict(X_test)
count = 0
correct = 0
for i in range(len(y_test)):
    count += 1
    if pred[i][0]>=0.5:
        if y_test[i] == 1:
            correct += 1
    else:
        if y_test[i] == 0:
            correct += 1
    print(pred[i],y_test[i])
acc = correct/count
print(count)
print('acc:', acc)
pred = model.predict(X_test)
len(pred)
pred
