# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy.random import shuffle

from keras.applications.resnet50 import ResNet50, preprocess_input

from keras.models import Sequential

import keras

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint,EarlyStopping

from keras.utils import to_categorical

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import gc

import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
TRAINDF_PATH = '/kaggle/input/bengaliai-cv19'

IMG_PATH = '/kaggle/input/bengaliai/256_train/256'

df_train = pd.read_csv(TRAINDF_PATH + '/train.csv')
df_train.head()
BATCH_SIZE = 64

DIM = (256,256)
import cv2



class BengaliGenerator(keras.utils.Sequence):

    def __init__(self,data,batch_size,dim, shuffle):

        self.data = data

        self.labels1 = pd.get_dummies(data['grapheme_root'], columns = ['grapheme_root'])

        self.labels2 = pd.get_dummies(data['vowel_diacritic'], columns = ['vowel_diacritic'])

        self.labels3 = pd.get_dummies(data['consonant_diacritic'], columns = ['consonant_diacritic'])

        self.list_ids = data.index.values

        self.batch_size = batch_size

        self.dim = dim

        self.shuffle = shuffle

        self.on_epoch_end()

        

    def __len__(self):

        return int(np.floor(len(self.data)/self.batch_size))

    

    def __getitem__(self,index):

        batch_ids = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        

        valid_ids = [self.list_ids[k] for k in batch_ids]

        X = np.empty((self.batch_size, *self.dim, 3))

        Y1 = np.empty((self.batch_size, 168), dtype = int)

        Y2 = np.empty((self.batch_size, 11), dtype = int)

        Y3 = np.empty((self.batch_size, 7), dtype = int)

        

        for i, k in enumerate(valid_ids):

            X[i,:, :, :] = cv2.imread(IMG_PATH + self.data['image_id'][k] + '.png') 

            Y1[i,:] = self.labels1.loc[k, :].values

            Y2[i,:] = self.labels2.loc[k, :].values

            Y3[i,:] = self.labels3.loc[k, :].values

            

        return X, [Y1, Y2, Y3]

    

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.list_ids))

        if self.shuffle:

            shuffle(self.indexes)
from sklearn.model_selection import train_test_split



train_X, val_X = train_test_split(df_train, test_size = 0.2, random_state = 2019)
train_generator = BengaliGenerator(train_X, BATCH_SIZE, DIM, True)

val_generator = BengaliGenerator(val_X, BATCH_SIZE, DIM, True)
# x,y = train_generator.next()
# x,y = next(train_generator)
# resnet=ResNet50(weights='../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',

#                include_top=False,input_shape=(*DIM,3))
resnet=ResNet50(weights=None,

               include_top=False,input_shape=(*DIM,3))
resnet.summary()
from keras.layers import Conv2D



resnet.layers[2] = Conv2D(64, (2, 2),

                      strides=(2, 2),

                      padding='valid',

                      kernel_initializer='he_normal',

                      name='conv1')



resnet.layers[2].build((None,262,262,3))
resnet.summary()
from keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense

from keras.models import Model



def build_model(nclasses1, nclasses2, nclasses3):

    

    res_output = resnet.output

    x = GlobalAveragePooling2D()(res_output)

    x = Dropout(0.5)(x)

    out1 = Dense(nclasses1, activation = 'softmax')(x)

    out2 = Dense(nclasses2, activation = 'softmax')(x)

    out3 = Dense(nclasses3, activation = 'softmax')(x)

    

    model = Model(inputs = resnet.input, outputs = [out1,out2,out3])

    

#     for layer in model.layers:

#         layer.trainable=True

    

    model.compile(

        loss='categorical_crossentropy',

        optimizer=Adam(lr=4e-4),

        metrics=['accuracy']

    )

    

    return model
model = build_model(168, 11, 7)
est=EarlyStopping(monitor='val_loss',patience=5, min_delta=0.005)

check_point = ModelCheckpoint('resnet50_1.pth', monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')

call_backs=[est, check_point]



history = model.fit_generator(

    train_generator,

    steps_per_epoch=int(len(train_X)/BATCH_SIZE),

    validation_data=val_generator,

    validation_steps = int(len(val_X))/BATCH_SIZE,

     epochs=5,

   callbacks=call_backs)