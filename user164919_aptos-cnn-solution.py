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
import os, sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import skimage.io

from skimage.transform import resize

from imgaug import augmenters as iaa

from imgaug import parameters as iap

from tqdm import tqdm

import PIL

from PIL import Image, ImageOps

import cv2

from sklearn.utils import class_weight, shuffle

from keras.losses import binary_crossentropy

from keras.applications.resnet50 import preprocess_input

import keras.backend as K

import tensorflow as tf

from sklearn.metrics import f1_score, fbeta_score

from keras.utils import Sequence

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split



WORKERS = 2

CHANNEL = 3



import warnings

warnings.filterwarnings("ignore")

SIZE = 512

NUM_CLASSES = 5
df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')



x = df_train['id_code']

y = df_train['diagnosis']



x, y = shuffle(x, y, random_state=8)
y = to_categorical(y, num_classes=NUM_CLASSES)

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15,

                                                      stratify=y, random_state=8)

print(train_x.shape)

print(train_y.shape)

print(valid_x.shape)

print(valid_y.shape)



class_weight = (np.max(np.sum(y,0))) / np.sum(y,0)

print(np.sum(y,0))

print(class_weight)
# https://github.com/aleju/imgaug

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential([

    sometimes(

        iaa.OneOf([

            iaa.Add((-10, 10), per_channel=0.5),

            iaa.Multiply((0.9, 1.1), per_channel=0.5),

            iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5),

            iaa.Affine(rotate=iap.DiscreteUniform(-180, 180))

        ])

    ),

    iaa.Fliplr(0.5),

    iaa.Crop(percent=(0, 0.1))

],random_order=True)

class My_Generator(Sequence):



    def __init__(self, image_filenames, labels, batch_size, is_train=True, mix=False):

        self.image_filenames, self.labels = image_filenames, labels

        self.batch_size = batch_size

        self.is_train = is_train

        self.on_epoch_end()

        self.is_mix = mix



    def __len__(self):

        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))



    def __getitem__(self, idx):

        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]



        if(self.is_train):

            return self.train_generate(batch_x, batch_y)

        return self.valid_generate(batch_x, batch_y)



    def on_epoch_end(self):

        self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)

    

    def mix_up(self, x, y):

        lam = np.random.beta(0.2, 0.4)

        ori_index = np.arange(int(len(x)))

        index_array = np.arange(int(len(x)))

        np.random.shuffle(index_array)        

        

        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]

        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]

        

        return mixed_x, mixed_y



    def train_generate(self, batch_x, batch_y):

        batch_images = []



        for (sample, label) in zip(batch_x, batch_y):

            img = cv2.imread('../input/aptos2019-blindness-detection/train_images/'+sample+'.png')

            img = cv2.resize(img, (SIZE, SIZE))

            img = seq.augment_image(img)

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        batch_y = np.array(batch_y, np.float32)

        if(self.is_mix):

            batch_images, batch_y = self.mix_up(batch_images, batch_y)

        return batch_images, batch_y



    def valid_generate(self, batch_x, batch_y):

        batch_images = []



        for (sample, label) in zip(batch_x, batch_y):

            img = cv2.imread('../input/aptos2019-blindness-detection/train_images/'+sample+'.png')

            img = cv2.resize(img, (SIZE, SIZE))

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        batch_y = np.array(batch_y, np.float32)

        return batch_images, batch_y
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, load_model

from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,

                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D)

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.callbacks import ModelCheckpoint

from keras import metrics

from keras.optimizers import Adam 

from keras import backend as K

import keras

from keras.models import Model
def create_model(input_shape, n_out):

    input_tensor = Input(shape=input_shape)

    base_model_2 = InceptionResNetV2(include_top=False,

                   weights=None,

                   input_tensor=input_tensor)    

    base_model_2.load_weights('../input/inceptionresnetv2/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5')

    x = GlobalAveragePooling2D()(base_model_2.output)

    #x = Dropout(0.5)(x)

    #x = Dense(1024, activation='relu')(x)

    #x = Dropout(0.5)(x)

    final_output = Dense(n_out, activation='softmax', name='final_output')(x)

    model = Model(input_tensor, final_output)

    

    return model
# create callbacks list

from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,

                             EarlyStopping, ReduceLROnPlateau,CSVLogger)



epochs = 15; batch_size = 16

checkpoint = ModelCheckpoint('../working/combo.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, 

                                   verbose=1, mode='auto', epsilon=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=7)



csv_logger = CSVLogger(filename='../working/training_log.csv',

                       separator=',',

                       append=True)

callbacks_list = [checkpoint, csv_logger, reduceLROnPlat, early]



train_generator = My_Generator(train_x, train_y, batch_size, is_train=True)

train_mixup = My_Generator(train_x, train_y, batch_size, is_train=True, mix=True)

valid_generator = My_Generator(valid_x, valid_y, batch_size, is_train=False)



model = create_model(

    input_shape=(SIZE,SIZE,3), 

    n_out=NUM_CLASSES)
# warm up model

for layer in model.layers:

    layer.trainable = False



for i in range(-1,0):

    model.layers[i].trainable = True



model.compile(

    loss='categorical_crossentropy',

    optimizer='sgd')



model.fit_generator(

    train_mixup,

    steps_per_epoch=np.ceil(float(len(train_y)) / float(128)),

    epochs=7,

    class_weight=class_weight,

    max_queue_size=16, workers=WORKERS, use_multiprocessing=True,

    verbose=1)
# train all layers

for layer in model.layers:

    layer.trainable = True



callbacks_list = [checkpoint, csv_logger, reduceLROnPlat]

model.compile(loss='categorical_crossentropy',

            optimizer=Adam(lr=1e-4),

            metrics=['accuracy'])



model.fit_generator(

    train_mixup,

    steps_per_epoch=np.ceil(float(len(train_x)) / float(batch_size)),

    validation_data=valid_generator,

    validation_steps=np.ceil(float(len(valid_x)) / float(batch_size)),

    class_weight=class_weight,

    epochs=epochs,

    verbose=1,

    max_queue_size=16, workers=2, use_multiprocessing=True,

    callbacks=callbacks_list)
submit = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

model.load_weights('../working/combo.h5')

predicted = []
for i, name in tqdm(enumerate(submit['id_code'])):

    path = os.path.join('../input/aptos2019-blindness-detection/test_images/', name+'.png')

    image = cv2.imread(path)

    image = cv2.resize(image, (SIZE, SIZE))

    score_predict = model.predict((image[np.newaxis])/255)

    label_predict = np.argmax(score_predict)

    predicted.append(str(label_predict))
submit['diagnosis'] = predicted

submit.to_csv('submission.csv', index=False)

submit.head()