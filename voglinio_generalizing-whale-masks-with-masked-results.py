import numpy as np

import pandas as pd



seed = 66

np.random.seed(seed)

import cv2

import json

import glob

import os

from tqdm import *

from shutil  import copyfile, rmtree 





from matplotlib import pyplot as plt



from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, Model

from keras.layers import Dropout, Flatten, Dense, SpatialDropout2D, Input

from keras import applications

from keras.preprocessing import image

from keras.preprocessing.image import load_img



from keras import backend as K

print (K.image_dim_ordering())

print (K.image_data_format())

from tqdm import tqdm_notebook, tqdm
TRAIN_PATH = '../input/humpback-whale-identification/train/'

TEST_PATH = '../input/humpback-whale-identification/test/'

MASK_PATH = '../input/whales-masks-500/masks/'
mask_files = os.listdir(MASK_PATH)

mask_files = [m for m in mask_files if 'mask' in m]

X = []

M = []

for mask_file in mask_files:

    img_file = mask_file.split('_')[0] + '.jpg'

    img = cv2.imread(TRAIN_PATH + img_file)

    mask = cv2.imread(MASK_PATH + mask_file, 0)

    mask[mask>108]= 255

    mask[mask<=108]= 0

    X.append(img)

    M.append(mask)

  

X = np.array(X)

M = np.array(M)

!pip install segmentation_models
!pip install albumentations
from albumentations import (

    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,

    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,

    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,

    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomGamma, Rotate,IAAAffine

)
aug_null = Compose([])

aug = Compose([ 

    Blur(p=0.5, blur_limit=2),

    IAAAffine(p=0.5, shear=5),

    HorizontalFlip(p=0.5),              

    #VerticalFlip(p=0.5),              

    Rotate(limit=5, p=0.3),

    #CLAHE(p=0.3),

    RandomContrast(p=0.2, limit=0.1),

    RandomBrightness(p=0.2, limit=0.1),

    #RandomGamma(p=0.2, gamma_limit=(90, 110))

])

import keras

from segmentation_models.backbones.classification_models.classification_models.resnet import preprocess_input



class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, X, M, batch_size=32,

                 dim=(299,299),  shuffle=True, 

                 preprocess_input=preprocess_input, 

                 aug=aug_null, min_mask=2 ):

        'Initialization'

        self.X = X

        self.M = M

        self.batch_size = batch_size

        self.n_classes = 1

        self.shuffle = shuffle

        self.preprocess_input = preprocess_input

        self.aug = aug

        self.on_epoch_end()

        self.dim = dim



    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor((len(self.X) / self.batch_size) / 1) )



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        

        end_index = min((index+1)*self.batch_size, len(self.indexes))

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]





        # Generate data

        X, Y = self.__data_generation(indexes)



        return X, Y



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.X))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __data_generation(self, indexes):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        

        batch_size = len(indexes)

        

        # Initialization

        XX = np.empty((batch_size, self.dim[0], self.dim[1], 3), dtype='float32')

        YY = np.empty((batch_size, self.dim[0], self.dim[1], 1), dtype='float32')



        # Generate data

        for i, ID in enumerate(indexes):

            # Store sample

            img = self.X[ID]

            if img.shape[0] != self.dim[0]:

                img = cv2.resize(img, self.dim, cv2.INTER_CUBIC)

            mask = self.M[ID]

            if mask.shape[0] != self.dim[0]:

                mask = cv2.resize(mask, self.dim, cv2.INTER_AREA)

            

            # Store class

            augmented = self.aug(image=img, mask=mask)

            aug_img = augmented['image']

            aug_mask = augmented['mask']

            aug_mask = np.expand_dims(aug_mask, axis=-1)

            aug_mask = aug_mask/255

            

            assert (np.max(aug_mask)<= 1.0 and  np.min(aug_mask) >= 0)

            aug_mask[aug_mask>0.5] = 1

            aug_mask[aug_mask<0.5] = 0

            

            YY[i,] = aug_mask.astype('float32')

            XX[i,] = aug_img.astype('float32')

    

       

        XX = self.preprocess_input(XX)



        return XX, YY
from segmentation_models.backbones import get_preprocessing

preprocess_input = get_preprocessing('resnet34')



training_generator = DataGenerator(X[:400], M[:400], batch_size=16,  dim=(384, 384), aug=aug, 

                                   preprocess_input=preprocess_input)

valid_genarator = DataGenerator(X[400:], M[400:], batch_size=16, dim=(384, 384), aug=aug_null, 

                                preprocess_input=preprocess_input, shuffle=False)
x, y= training_generator[7]

np.max(x), x.shape, y.shape, np.max(y), np.unique(y)
plt.imshow(y[9, ..., 0])

plt.show()
import keras.backend as K

from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

from keras.optimizers import Adam

from segmentation_models import Unet, FPN



model = FPN(backbone_name='resnet34', encoder_weights='imagenet', activation='sigmoid', classes=1, dropout=0.5)

from keras.losses import binary_crossentropy

import keras.backend as K

import tensorflow as tf





def dice_coeff_L(y_true, y_pred):

    smooth = 1.

    y_pred_sig = tf.nn.sigmoid(y_pred)



    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred_sig)

    intersection = K.sum(y_true_f * y_pred_f)

    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return score



def dice_coeff(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return score





def dice_loss(y_true, y_pred):

    loss = 1 - dice_coeff(y_true, y_pred)

    return loss





def bce_dice_loss(y_true, y_pred):

    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    return loss
from keras.callbacks import ReduceLROnPlateau

from keras.optimizers import SGD



training_generator = DataGenerator(X[:400], M[:400], batch_size=16,  dim=(384, 384), aug=aug, 

                                   preprocess_input=preprocess_input)

valid_genarator = DataGenerator(X[400:], M[400:], batch_size=16, dim=(384, 384), aug=aug_null, 

                                preprocess_input=preprocess_input, shuffle=False)



model.compile(optimizer=Adam(lr=0.001),

          loss=bce_dice_loss,

          metrics=[dice_coeff])



epochs = 40



early_stopping = EarlyStopping(patience=10, verbose=1, monitor='val_dice_coeff', mode='max')

model_checkpoint = ModelCheckpoint("fpnresnet34_384_{epoch:02d}-{val_loss:.3f}-{val_dice_coeff:.3f}.hdf5", 

                                   save_best_only=True, 

                                   save_weights_only=True, 

                                   monitor='val_dice_coeff', verbose=1, mode='max', period=2)

reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.000001, verbose=1, monitor='val_dice_coeff', mode='max')



 

history = model.fit_generator( training_generator,

                                     validation_data=valid_genarator,

                                     epochs=epochs,

                                     callbacks=[ reduce_lr, early_stopping, model_checkpoint], 

                                     verbose=1)
aug_null = Compose([])





class TestDataGenerator(keras.utils.Sequence):

    'Generates data for Keras'

    def __init__(self, X, batch_size=32,

                 dim=(299,299),  shuffle=True, 

                 preprocess_input=preprocess_input, 

                 aug=aug_null, min_mask=2 ):

        'Initialization'

        self.X = X

        self.batch_size = batch_size

        self.n_classes = 1

        self.shuffle = shuffle

        self.preprocess_input = preprocess_input

        self.aug = aug

        self.dim = dim

        self.on_epoch_end()



    def set_aug(self, aug):

        self.aug = aug

        self.on_epoch_end()

      

    def __len__(self):

        'Denotes the number of batches per epoch'

        return int(np.floor((len(self.X) / self.batch_size) / 1) )



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        

        end_index = min((index+1)*self.batch_size, len(self.indexes))

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        # Find list of IDs



        # Generate data

        xx = self.__data_generation(indexes)



        return xx



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.X))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __data_generation(self, indexes):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        

        batch_size = len(indexes)

        

        # Initialization

        XX = np.empty((batch_size, self.dim[0], self.dim[1], 3), dtype='float32')



        # Generate data

        for i, ID in enumerate(indexes):

            # Store sample

            img = self.X[ID]

            if img.shape[0] != self.dim[0]:

                img = cv2.resize(img, self.dim, cv2.INTER_CUBIC)

            

            

            # Store class

            augmented = self.aug(image=img)

            aug_img = augmented['image']

           

            XX[i,] = aug_img.astype('float32')

    

       

        XX = self.preprocess_input(XX)



        return XX
null_aug = Compose([])

valid_genarator = DataGenerator(X[400:], M[400:], batch_size=16, aug=null_aug, preprocess_input=preprocess_input, dim=(384, 384), 

                               shuffle=False)

preds = model.predict_generator(valid_genarator, verbose=1)



flip_aug = Compose([HorizontalFlip(p=1.0) ])

valid_genarator = DataGenerator(X[400:], M[400:], batch_size=16, aug=flip_aug, preprocess_input=preprocess_input,  dim=(384, 384), shuffle=False)

preds_hflip = model.predict_generator(valid_genarator, verbose=1)



blur_aug = Compose([Blur(p=1.0)])

valid_genarator = DataGenerator(X[400:], M[400:], batch_size=16, aug=blur_aug, preprocess_input=preprocess_input,  dim=(384, 384), shuffle=False)

preds_blur = model.predict_generator(valid_genarator, verbose=1)



TARGET_VAL = []

for i in range(len(preds)):

  pp = (preds[i] + np.fliplr(preds_hflip[i]) + preds_blur[i])/3

  TARGET_VAL.append(pp)



TARGET_VAL = np.array(TARGET_VAL)  
SIZE = 384

f, axarr = plt.subplots(6, 6)

f.set_figwidth(20)

f.set_figheight(15)

kernel = np.ones((3,3),np.uint8)



for i in range(0, 36):

    img =  cv2.resize(X[400+i], (384, 384))

    mask = ((TARGET_VAL[i, ..., 0]) > 0.25).astype('uint8')

    back = ((TARGET_VAL[i, ..., 0]) <= 0.25).astype('uint8')



    img = np.stack([img[..., j] * mask + back*255 for j in range(3)], axis=-1)



    contours,hierarchy = cv2.findContours(mask, 1, 2)

  # Cycle through contours and add area to array

    areas = []

    for c in contours:

        areas.append(cv2.contourArea(c))



    # Sort array of areas by size

    sorted_areas = sorted(zip(areas, contours), key=lambda x: x[0], reverse=True)

    title = str(len(sorted_areas)) 

    

    cnt = sorted_areas[0][1]

    x1,y1,w,h = cv2.boundingRect(cnt)

    x2 = x1 + w

    y2 = y1 + h

    

    for j in range(1, len(sorted_areas)):

        cnt = sorted_areas[j][1]

        tx1,ty1,tw,th = cv2.boundingRect(cnt)

        tx2 = tx1 + tw

        ty2 = ty1 + th

        x1 = min(x1, tx1)

        y1 = min(y1, ty1)

        x2 = max(x2, tx2)

        y2 = max(y2, ty2)

    

    x = x1

    y = y1

    w = x2-x1

    h = y2-y1





    img_cropped = img[y:y+h, x:x+w]

    axarr[int(i/6), i%6].imshow(img_cropped, cmap='gray')

    axarr[int(i/6), i%6].axis('off')

plt.show()
SIZE = 384

f, axarr = plt.subplots(6, 6)

f.set_figwidth(20)

f.set_figheight(15)

kernel = np.ones((3,3),np.uint8)



for i in range(0, 36):

    img =  cv2.resize(X[400+i], (384, 384))

    mask = ((TARGET_VAL[i, ..., 0]) > 0.25).astype('uint8')



    img = np.stack([img[..., j] * mask  for j in range(3)], axis=-1)



    contours,hierarchy = cv2.findContours(mask, 1, 2)

  # Cycle through contours and add area to array

    areas = []

    for c in contours:

        areas.append(cv2.contourArea(c))



    # Sort array of areas by size

    sorted_areas = sorted(zip(areas, contours), key=lambda x: x[0], reverse=True)

    title = str(len(sorted_areas)) 

    

    cnt = sorted_areas[0][1]

    x1,y1,w,h = cv2.boundingRect(cnt)

    x2 = x1 + w

    y2 = y1 + h

    

    for j in range(1, len(sorted_areas)):

        cnt = sorted_areas[j][1]

        tx1,ty1,tw,th = cv2.boundingRect(cnt)

        tx2 = tx1 + tw

        ty2 = ty1 + th

        x1 = min(x1, tx1)

        y1 = min(y1, ty1)

        x2 = max(x2, tx2)

        y2 = max(y2, ty2)

    

    x = x1

    y = y1

    w = x2-x1

    h = y2-y1





    img_cropped = img[y:y+h, x:x+w]

    axarr[int(i/6), i%6].imshow(img_cropped, cmap='gray')

    axarr[int(i/6), i%6].axis('off')

plt.show()
SIZE = 384

f, axarr = plt.subplots(6, 6)

f.set_figwidth(20)

f.set_figheight(15)

kernel = np.ones((3,3),np.uint8)



for i in range(0, 36):

    img =  cv2.resize(X[400+i], (384, 384))

    mask = ((TARGET_VAL[i, ..., 0]) > 0.25).astype('uint8')



    img = np.stack([img[..., j]  for j in range(3)], axis=-1)



    contours,hierarchy = cv2.findContours(mask, 1, 2)

  # Cycle through contours and add area to array

    areas = []

    for c in contours:

        areas.append(cv2.contourArea(c))



    # Sort array of areas by size

    sorted_areas = sorted(zip(areas, contours), key=lambda x: x[0], reverse=True)

    title = str(len(sorted_areas)) 

    

    cnt = sorted_areas[0][1]

    x1,y1,w,h = cv2.boundingRect(cnt)

    x2 = x1 + w

    y2 = y1 + h

    

    for j in range(1, len(sorted_areas)):

        cnt = sorted_areas[j][1]

        tx1,ty1,tw,th = cv2.boundingRect(cnt)

        tx2 = tx1 + tw

        ty2 = ty1 + th

        x1 = min(x1, tx1)

        y1 = min(y1, ty1)

        x2 = max(x2, tx2)

        y2 = max(y2, ty2)

    

    x = x1

    y = y1

    w = x2-x1

    h = y2-y1





    img_cropped = img[y:y+h, x:x+w]

    axarr[int(i/6), i%6].imshow(img_cropped, cmap='gray')

    axarr[int(i/6), i%6].axis('off')

plt.show()
SIZE = 384

f, axarr = plt.subplots(6, 6)

f.set_figwidth(20)

f.set_figheight(15)

kernel = np.ones((3,3),np.uint8)



for i in range(0, 36):

    img =  cv2.resize(X[400+i], (384, 384))

    img =  cv2.cvtColor(img, cv2.cv2.COLOR_BGR2GRAY)

    mask = ((TARGET_VAL[i, ..., 0]) > 0.25).astype('uint8')

    back = ((TARGET_VAL[i, ..., 0]) <= 0.25).astype('uint8')

    img = img * mask + back*255



    contours,hierarchy = cv2.findContours(mask, 1, 2)

  # Cycle through contours and add area to array

    areas = []

    for c in contours:

        areas.append(cv2.contourArea(c))



    # Sort array of areas by size

    sorted_areas = sorted(zip(areas, contours), key=lambda x: x[0], reverse=True)

    title = str(len(sorted_areas)) 

    

    cnt = sorted_areas[0][1]

    x1,y1,w,h = cv2.boundingRect(cnt)

    x2 = x1 + w

    y2 = y1 + h

    

    for j in range(1, len(sorted_areas)):

        cnt = sorted_areas[j][1]

        tx1,ty1,tw,th = cv2.boundingRect(cnt)

        tx2 = tx1 + tw

        ty2 = ty1 + th

        x1 = min(x1, tx1)

        y1 = min(y1, ty1)

        x2 = max(x2, tx2)

        y2 = max(y2, ty2)

    

    x = x1

    y = y1

    w = x2-x1

    h = y2-y1





    img_cropped = img[y:y+h, x:x+w]

    axarr[int(i/6), i%6].imshow(img_cropped, cmap='gray')

    axarr[int(i/6), i%6].axis('off')

plt.show()