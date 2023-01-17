import numpy as np

import pandas as pd

from matplotlib import pyplot as plt 

from skimage import io, transform

import cv2

import tensorflow as tf
import os

print(os.listdir("../input"))
FAST_RUN=False

FAST_PREDICT=True

IMAGE_WIDTH=128

IMAGE_HEIGHT=128

CHANNEL=3

input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNEL)

mask_shape=(IMAGE_WIDTH, IMAGE_HEIGHT)

image_size=(IMAGE_WIDTH, IMAGE_HEIGHT)

batch_size=10

epochs=3
metadata = pd.read_csv("../input/metadata.csv")

train_masks = pd.read_csv("../input/train_masks.csv")

testdata = pd.read_csv("../input/sample_submission.csv")
print(train_masks.shape)

print(testdata.shape)
if FAST_RUN:

    epochs=1

    train_masks = train_masks.sample(1000).reset_index()



if FAST_PREDICT: 

    testdata = testdata.sample(batch_size).reset_index()
testdata.head()
metadata.head()
train_masks.head()
filenames = train_masks.img.str.split(".")

maskfilenames = filenames.str[0] + "_mask.gif"

train_masks['img_mask'] = maskfilenames

train_masks['angle'] = filenames.str[0].str.split("_").str[1].astype(int)

train_masks.head()
sample = train_masks.sample()

fig=plt.figure(figsize=(16, 8))

for index, s in sample.iterrows():

    original_image = io.imread('../input/train/'+s.img)

    masked_image = io.imread('../input/train_masks/'+s.img_mask)

    plt.subplot(2, 2, 1)

    plt.imshow(original_image)

    plt.subplot(2, 2, 2)

    plt.imshow(masked_image)
from skimage.transform import AffineTransform, warp

def shift(image, translation_matrix):

    transformer = AffineTransform(translation=translation_matrix)

    return warp(image, transformer, mode='wrap', preserve_range=True)
def tranform_image(original_image, mask_image):

    image = original_image

    mask = mask_image

    

    isHorizontalFlip = np.random.random() < 0.5

    isShift = np.random.random() < 0.5



    if isShift:

        translation_matrix = np.random.random_integers(-10, 10), np.random.random_integers(-10, 10)

        image = shift(image, translation_matrix)

        mask = shift(mask, translation_matrix)



    if isHorizontalFlip:

        image = image[:, ::-1]

        mask = mask[:, ::-1]

        

    image = image / 255.0

    mask = mask / 255.0

    

    return image, mask
def data_gen_small(data_dir, mask_dir, df_data, precess_batch_size, original_image_shape, mask_image_shape):

    while True:

        for k, ix in df_data.groupby(np.arange(len(df_data))//precess_batch_size):

            imgs = []

            labels = []

            for index, row in ix.iterrows():

                # images

                original_img = io.imread(data_dir + row.img)

                resized_img = transform.resize(original_img, image_size, mode='constant')

                # masks

                original_mask = io.imread(mask_dir + row.img_mask, as_gray=True)

                resized_mask = transform.resize(original_mask, image_size, mode='constant')

                

                image, mask = tranform_image(resized_img, resized_mask)

                

                imgs.append(image)

                labels.append(np.expand_dims(mask, axis=2))

                

            imgs = np.array(imgs)

            labels = np.array(labels)

            yield imgs, labels
train_gen = data_gen_small("../input/train/", "../input/train_masks/", train_masks, batch_size, input_shape, mask_shape)
fig=plt.figure(figsize=(16, 16))

for i in [1, 2, 3, 4]:

    img, msk = next(train_gen)

    plt.subplot(4, 2, i*2-1)

    plt.imshow(img[0]*255.0)

    plt.subplot(4, 2, i*2)

    plt.imshow(msk[0].reshape(128, 128))

    
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization



inputs = Input(shape=input_shape)

# 128



down1 = Conv2D(64, (3, 3), padding='same')(inputs)

down1 = BatchNormalization()(down1)

down1 = Activation('relu')(down1)

down1 = Conv2D(64, (3, 3), padding='same')(down1)

down1 = BatchNormalization()(down1)

down1 = Activation('relu')(down1)

down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

# 64



down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)

down2 = BatchNormalization()(down2)

down2 = Activation('relu')(down2)

down2 = Conv2D(128, (3, 3), padding='same')(down2)

down2 = BatchNormalization()(down2)

down2 = Activation('relu')(down2)

down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

# 32



down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)

down3 = BatchNormalization()(down3)

down3 = Activation('relu')(down3)

down3 = Conv2D(256, (3, 3), padding='same')(down3)

down3 = BatchNormalization()(down3)

down3 = Activation('relu')(down3)

down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)

# 16



down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)

down4 = BatchNormalization()(down4)

down4 = Activation('relu')(down4)

down4 = Conv2D(512, (3, 3), padding='same')(down4)

down4 = BatchNormalization()(down4)

down4 = Activation('relu')(down4)

down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)

# 8



center = Conv2D(1024, (3, 3), padding='same')(down4_pool)

center = BatchNormalization()(center)

center = Activation('relu')(center)

center = Conv2D(1024, (3, 3), padding='same')(center)

center = BatchNormalization()(center)

center = Activation('relu')(center)

# center



up4 = UpSampling2D((2, 2))(center)

up4 = concatenate([down4, up4], axis=3)

up4 = Conv2D(512, (3, 3), padding='same')(up4)

up4 = BatchNormalization()(up4)

up4 = Activation('relu')(up4)

up4 = Conv2D(512, (3, 3), padding='same')(up4)

up4 = BatchNormalization()(up4)

up4 = Activation('relu')(up4)

up4 = Conv2D(512, (3, 3), padding='same')(up4)

up4 = BatchNormalization()(up4)

up4 = Activation('relu')(up4)

# 16



up3 = UpSampling2D((2, 2))(up4)

up3 = concatenate([down3, up3], axis=3)

up3 = Conv2D(256, (3, 3), padding='same')(up3)

up3 = BatchNormalization()(up3)

up3 = Activation('relu')(up3)

up3 = Conv2D(256, (3, 3), padding='same')(up3)

up3 = BatchNormalization()(up3)

up3 = Activation('relu')(up3)

up3 = Conv2D(256, (3, 3), padding='same')(up3)

up3 = BatchNormalization()(up3)

up3 = Activation('relu')(up3)

# 32



up2 = UpSampling2D((2, 2))(up3)

up2 = concatenate([down2, up2], axis=3)

up2 = Conv2D(128, (3, 3), padding='same')(up2)

up2 = BatchNormalization()(up2)

up2 = Activation('relu')(up2)

up2 = Conv2D(128, (3, 3), padding='same')(up2)

up2 = BatchNormalization()(up2)

up2 = Activation('relu')(up2)

up2 = Conv2D(128, (3, 3), padding='same')(up2)

up2 = BatchNormalization()(up2)

up2 = Activation('relu')(up2)

# 64



up1 = UpSampling2D((2, 2))(up2)

up1 = concatenate([down1, up1], axis=3)

up1 = Conv2D(64, (3, 3), padding='same')(up1)

up1 = BatchNormalization()(up1)

up1 = Activation('relu')(up1)

up1 = Conv2D(64, (3, 3), padding='same')(up1)

up1 = BatchNormalization()(up1)

up1 = Activation('relu')(up1)

up1 = Conv2D(64, (3, 3), padding='same')(up1)

up1 = BatchNormalization()(up1)

up1 = Activation('relu')(up1)

# 128



outputs = Conv2D(1, (1, 1), activation='sigmoid')(up1)



model = Model(inputs=inputs, outputs=outputs)

optimizer = tf.train.RMSPropOptimizer(0.0001)
model.compile(

    optimizer=optimizer, 

    loss="binary_crossentropy", 

    metrics=["accuracy"]

)
steps_per_epoch=np.ceil(float(len(train_masks)) / float(batch_size)).astype(int)

history = model.fit_generator(

    train_gen, 

    steps_per_epoch=steps_per_epoch,

    epochs=epochs

)
model.save("model.h5")
def test_gen_small(data_dir, df_data, precess_batch_size, original_image_shape):

    while True:

        for k, ix in df_data.groupby(np.arange(len(df_data))//precess_batch_size):

            imgs = []

            labels = []

            for index, row in ix.iterrows():

                # images

                original_img = io.imread(data_dir + row.img)

                resized_img = transform.resize(original_img, original_image_shape) / 255.0

                imgs.append(resized_img)



            imgs = np.array(imgs)

            yield imgs

test_gen = test_gen_small("../input/test/", testdata, batch_size, input_shape)
img = next(test_gen)

fig=plt.figure(figsize=(16, 8))

for i in [1, 2, 3, 4]:

    plt.subplot(1, 4, i)

    plt.imshow(img[i-1]*255.0)
steps = np.ceil(float(len(testdata)) / float(batch_size)).astype(int)

y_predicted = model.predict_generator(

    test_gen, 

    steps=steps

)
fig=plt.figure(figsize=(16, 8))

for i in [1, 2, 3, 4]:

    y_predict = y_predicted[i-1]

    plt.subplot(1, 4, i)

    plt.imshow(y_predict.reshape(128, 128))