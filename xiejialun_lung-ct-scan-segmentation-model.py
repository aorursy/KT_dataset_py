!conda install -c conda-forge gdcm -y
import os

import cv2

import random

import pydicom

import warnings

import numpy as np

import pandas as pd

from glob import glob

from tqdm import tqdm

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import get_custom_objects



warnings.filterwarnings('ignore')

print('Tensorflow version : {}'.format(tf.__version__))
def get_segmentation_model():

    

    class FixedDropout(tf.keras.layers.Dropout):

        def _get_noise_shape(self, inputs):

            if self.noise_shape is None:

                return self.noise_shape



            symbolic_shape = tf.keras.backend.shape(inputs)

            noise_shape = [symbolic_shape[axis] if shape is None else shape

                           for axis, shape in enumerate(self.noise_shape)]

            return tuple(noise_shape)



    def DiceCoef(y_trues, y_preds, smooth=1e-5, axis=None):

        intersection = tf.reduce_sum(y_trues * y_preds, axis=axis)

        union = tf.reduce_sum(y_trues, axis=axis) + tf.reduce_sum(y_preds, axis=axis)

        return tf.reduce_mean((2*intersection+smooth) / (union + smooth))



    def DiceLoss(y_trues, y_preds):

        return 1.0 - DiceCoef(y_trues, y_preds)



    get_custom_objects().update({'swish': tf.keras.layers.Activation(tf.nn.swish)})

    get_custom_objects().update({'FixedDropout':FixedDropout})

    get_custom_objects().update({'DiceCoef' : DiceCoef})

    get_custom_objects().update({'DiceLoss' : DiceLoss})

    

    print('Load segmentation model...')

    model = tf.keras.models.load_model('../input/lung-ct-segmentation-pretrain/osic_segmentation_model.h5')

    return model
model = get_segmentation_model()
def Load_random_ct_scan(BS=8):

    

    DIM = 256

    CT_PATH = '../input/osic-pulmonary-fibrosis-progression/train/'

    patients = os.listdir(CT_PATH)

    images = np.zeros((BS, DIM, DIM, 3), dtype=np.uint8)

    

    for idx in range(BS):

        random_patient = np.random.choice(patients, 1)[0]

        fns = os.listdir(os.path.join(CT_PATH, random_patient))

        random_fn = np.random.choice(fns, 1)[0]

        

        dcm = pydicom.dcmread(os.path.join(os.path.join(CT_PATH, random_patient), random_fn))

        image = dcm.pixel_array

        image = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)



        if image.shape[0] != 512 or image.shape[1] != 512:

            old_x, old_y = image.shape[0], image.shape[1]

            x = (image.shape[0] - 512) // 2

            y = (image.shape[1] - 512) // 2

            image = image[x : old_x-x, y : old_y-y]

            image = image[:512, :512]



        image = cv2.resize(image, (DIM,DIM), cv2.INTER_AREA)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        images[idx] = image

    

    return images / 255.0
DEMO_BATCH = 8

for idx in range(DEMO_BATCH):

    images = Load_random_ct_scan()

    pred_masks = model.predict(images, verbose=0)

    pred_masks = (pred_masks>0.5).astype(np.float32)

    

    plt.figure(figsize=(24,12))

    for idx, (image, mask) in enumerate(zip(images, pred_masks)):

        plt.subplot(1,8,idx+1)

        plt.imshow(image)

        plt.imshow(mask[:,:,0], alpha=0.35)

        plt.xticks([])

        plt.yticks([])

    plt.show()