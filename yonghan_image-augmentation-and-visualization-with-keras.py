import os

import sys



import numpy as np

import pandas as pd

import cv2



from PIL import Image

from matplotlib import pyplot as plt



from keras.applications.inception_v3 import preprocess_input

from keras.preprocessing.image import ImageDataGenerator, load_img
DATA_PATH = '../input/aptos2019-blindness-detection'

TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train_images')

TRAIN_LABEL_PATH = os.path.join(DATA_PATH, 'train.csv')



df_train = pd.read_csv(TRAIN_LABEL_PATH)

df_train['diagnosis'] = df_train['diagnosis'].astype('str')

df_train = df_train[['id_code', 'diagnosis']]

if df_train['id_code'][0].split('.')[-1] != 'png':

    for index in range(len(df_train['id_code'])):

        df_train['id_code'][index] = df_train['id_code'][index] + '.png'

X_train = df_train
def generator(datagen):

    return datagen.flow_from_dataframe(

        dataframe=X_train, 

        directory=TRAIN_IMG_PATH,

        x_col='id_code',

        y_col='diagnosis',

        target_size=(299, 299),

        color_mode='rgb',

        class_mode='categorical',

        batch_size=32,

        shuffle=False

    )



def visualization(generator):

    fig, ax = plt.subplots(1, 5, figsize=(30,50))

    count = 0

    for X_batch, y_batch in generator:

        while count < 5:

            ax[count].imshow(X_batch[count])

            count += 1

        break



        plt.show()
datagen1 = ImageDataGenerator(

    rescale=1./255

)

visualization(generator(datagen1))
datagen2 = ImageDataGenerator(

    rescale=1./255,

    horizontal_flip=True,

    vertical_flip=True

)

visualization(generator(datagen2))
datagen3 = ImageDataGenerator(

    rescale=1./255,

    brightness_range=[0.5, 1.5]

)

visualization(generator(datagen3))
datagen4 = ImageDataGenerator(

    rescale=1./255,

    rotation_range=30,

    brightness_range=[0.5, 1.5],

    zoom_range=[0.8, 1.2],

    horizontal_flip=True,

    vertical_flip=False

)

visualization(generator(datagen4))
datagen5 = ImageDataGenerator(

    preprocessing_function=preprocess_input

)

visualization(generator(datagen5))