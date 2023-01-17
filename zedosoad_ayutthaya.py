# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

from spectral import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.





# My imports

import tensorflow as tf



import keras

import cv2



import matplotlib.pyplot as plt

import math

from math import sqrt, pi, exp, floor, ceil

from skimage.io import imsave, imread

from keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import StandardScaler



from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Lambda

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.layers import *

from keras.layers import multiply, Concatenate, Activation

from keras import optimizers

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

from keras.layers import BatchNormalization

from keras.utils import np_utils

import keras.backend as K

from keras.callbacks import Callback

from keras.losses import binary_crossentropy, categorical_crossentropy



from sklearn.metrics import jaccard_similarity_score



import cv2



from sklearn.metrics import classification_report, confusion_matrix



from matplotlib import pyplot as PLT





from keras.callbacks import History 

history = History()





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





from keras.layers import Conv2D

from keras.layers import BatchNormalization

from keras.layers import Activation

from keras.layers import Add

from keras.layers import ZeroPadding2D



import keras.backend as K

from keras.layers import Input

from keras.layers import MaxPooling2D, SpatialDropout2D

from keras.layers import GlobalAveragePooling2D

from keras.layers import ZeroPadding2D

from keras.layers import Dense

from keras.models import Model

from keras.engine import get_source_inputs

from keras.layers import UpSampling2D

from keras.layers import Concatenate

crop_size = 112

n_channels = 5

n_classes = 2

crop_rate = 1

batch_size = 1

weights = np.array([2.5, 1, 1])
# Pre-processing of crops

def create_crops(img):

    image_rows = img.shape[0] 

    image_cols = img.shape[1]

    n_imgs = 0

    

    correct = (1-crop_rate)*crop_size

    n_imgs_row = ceil((image_rows-correct)/(crop_size*crop_rate))

    n_imgs_col = ceil((image_cols-correct)/(crop_size*crop_rate))

    imgs_cropped = np.zeros((n_imgs_row*n_imgs_col, crop_size, crop_size, img.shape[2]))



    print('-'*30)

    print('Creating image crops...')

    print('-'*30)

    

    cnt = 0;

    for i in range(0,n_imgs_row):

        if i < n_imgs_row - 1:

            y = round(i*crop_size*crop_rate)

        else:

            y = image_rows - crop_size

        

        for j in range(0,n_imgs_col):

            if j < n_imgs_col - 1:

                x = round(j*crop_size*crop_rate)

            else:

                x = image_cols - crop_size

            imgs_cropped[cnt] = img[y:(y+crop_size),x:(x+crop_size)]

            cnt = cnt + 1



    return(imgs_cropped[0:cnt,:,:,:])





# joins mask and data augmentation

def mask_trans(batches):

    while True:

        b = next(batches)

        bb = b[0]

        yield(bb[:,:,:,0:n_channels],bb[:,:,:,n_channels::])

        

        

        

smooth = 1





# Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"

# -> the score is computed for each class separately and then summed

# alpha=beta=0.5 : dice coefficient

# alpha=beta=1   : tanimoto coefficient (also known as jaccard)

# alpha+beta=1   : produces set of F*-scores

# implemented by E. Moebel, 06/04/18

def tversky_loss(y_true, y_pred):

    alpha = 0.5

    beta  = 0.5

    

    ones = K.ones(K.shape(y_true))

    p0 = y_pred      # proba that voxels are class i

    p1 = ones-y_pred # proba that voxels are not class i

    g0 = y_true

    g1 = ones-y_true

    

    num = K.sum(p0*g0, (0,1,2))

    den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))

    

    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]

    

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')

    return Ncl-T







def weighted_categorical_crossentropy2(weights):

    """

    A weighted version of keras.objectives.categorical_crossentropy

    

    Variables:

        weights: numpy array of shape (C,) where C is the number of classes

    

    Usage:

        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.

        loss = weighted_categorical_crossentropy(weights)

        model.compile(loss=loss,optimizer='adam')

    """

    

    weights = K.variable(weights)

        

    def loss(y_true, y_pred):

        smooth = 1.

        y_true_f = K.flatten(y_true)

        y_pred_f = K.flatten(y_pred)

        intersection = K.flatten(y_true * y_pred)

        loss = (2. * K.sum(intersection) + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

    

    return loss



def weighted_categorical_crossentropy(weights):

    """

    A weighted version of keras.objectives.categorical_crossentropy

    

    Variables:

        weights: numpy array of shape (C,) where C is the number of classes

    

    Usage:

        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.

        loss = weighted_categorical_crossentropy(weights)

        model.compile(loss=loss,optimizer='adam')

    """

    

    weights = K.variable(weights)

        

    def loss(y_true, y_pred):

        # scale predictions so that the class probas of each sample sum to 1

        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # clip to prevent NaN's and Inf's

        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # calc

        loss = y_true * K.log(y_pred) * weights

        #loss = -K.sum(loss, -1) + categorical_crossentropy(y_true, y_pred)

        loss = -K.sum(loss, -1)

        return loss

    

    return loss

def bce_dice_loss(y_true, y_pred):

    return categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    

def bce_dice_loss(y_true, y_pred):

    return categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)



def w_dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = y_true * y_pred

    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)



def dice_coef(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_coef_multilabel(y_true, y_pred):

    dice=0

    for index in range(3):

        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])

    return 1 - dice



def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = y_true * y_pred

    score = 1 - (2. * K.sum(intersection) + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

    

    

    return 1. - score



def jaccard_coef_int_nobg(y_true, y_pred):

    # __author__ = Vladimir Iglovikov

    y_pred_pos = K.round(K.clip(y_pred[:,:,:,0:2], 0, 1))



    intersection = K.sum(y_true[:,:,:,0:2] * y_pred_pos, axis=[0, -1, -2])

    sum_ = K.sum(y_true[:,:,:,0:2] + y_pred[:,:,:,0:2], axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)





def jaccard_coef_int_source(y_true, y_pred):

    # __author__ = Vladimir Iglovikov

    y_pred_all = K.argmax(y_pred, axis=-1)

    y_true_all = K.argmax(y_true, axis=-1)

    

    ones = K.ones_like(y_pred_all)*0

    twos = K.ones_like(y_pred_all)*1

    

    pred1 = K.cast(K.equal(y_pred_all, ones), 'float32')

    pred2 = K.cast(K.equal(y_pred_all, twos), 'float32')

    true1 = K.cast(K.equal(y_true_all, ones), 'float32')

    true2 = K.cast(K.equal(y_true_all, twos), 'float32')

    

    intersection1 = K.sum(pred1*true1)

    intersection2 = K.sum(pred2*true2)

    

    pred1sum = K.sum(pred1)

    pred2sum = K.sum(pred2)

    true1sum = K.sum(true1)

    true2sum = K.sum(true2)

    

    union1 = pred1sum + true1sum - intersection1

    union2 = pred2sum + true2sum - intersection2



    jac = (intersection1 + smooth) / (union1 + smooth)

    return jac



def jaccard_coef_int_transport(y_true, y_pred):

    # __author__ = Vladimir Iglovikov

    y_pred_all = K.argmax(y_pred, axis=-1)

    y_true_all = K.argmax(y_true, axis=-1)

    

    ones = K.ones_like(y_pred_all)*0

    twos = K.ones_like(y_pred_all)*1

    

    pred1 = K.cast(K.equal(y_pred_all, ones), 'float32')

    pred2 = K.cast(K.equal(y_pred_all, twos), 'float32')

    true1 = K.cast(K.equal(y_true_all, ones), 'float32')

    true2 = K.cast(K.equal(y_true_all, twos), 'float32')

    

    intersection1 = K.sum(pred1*true1)

    intersection2 = K.sum(pred2*true2)

    

    pred1sum = K.sum(pred1)

    pred2sum = K.sum(pred2)

    true1sum = K.sum(true1)

    true2sum = K.sum(true2)

    

    union1 = pred1sum + true1sum - intersection1

    union2 = pred2sum + true2sum - intersection2



    jac = (intersection2 + smooth) / (union2 + smooth)

    return jac



def jaccard_coef_int_both(y_true, y_pred):

    # __author__ = Vladimir Iglovikov

    y_pred_all = K.argmax(y_pred, axis=-1)

    y_true_all = K.argmax(y_true, axis=-1)

    

    ones = K.ones_like(y_pred_all)*0

    twos = K.ones_like(y_pred_all)*1

    

    pred1 = K.cast(K.equal(y_pred_all, ones), 'float32')

    pred2 = K.cast(K.equal(y_pred_all, twos), 'float32')

    true1 = K.cast(K.equal(y_true_all, ones), 'float32')

    true2 = K.cast(K.equal(y_true_all, twos), 'float32')

    

    intersection1 = K.sum(pred1*true1)

    intersection2 = K.sum(pred2*true2)

    

    pred1sum = K.sum(pred1)

    pred2sum = K.sum(pred2)

    true1sum = K.sum(true1)

    true2sum = K.sum(true2)

    

    union1 = pred1sum + true1sum - intersection1

    union2 = pred2sum + true2sum - intersection2



    jac = ((intersection2 + smooth) / (union2 + smooth) + (intersection1 + smooth) / (union1 + smooth))/2

    return jac













def true_positive_rate(y_true, y_pred):

    return (K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred))) + 0.0001)/(K.sum(y_true) + 0.0001)



def weighted_dice_coef(y_true, y_pred):

    mean = 0.21649066

    #w_1 = 1/mean**2

    #w_0 = 1/(1-mean)**2

    w_1 = 1

    w_0 = 1

    y_true_f_1 = K.flatten(y_true)

    y_pred_f_1 = K.flatten(y_pred)

    y_true_f_0 = K.flatten(1-y_true)

    y_pred_f_0 = K.flatten(1-y_pred)



    intersection_0 = K.sum(y_true_f_0 * y_pred_f_0)

    intersection_1 = K.sum(y_true_f_1 * y_pred_f_1)



    return intersection_1 / (K.sum(y_true_f_1) + K.sum(y_pred_f_1) - intersection_1)





def get_iou(A, B):

    batch_size = A.shape[0]

    metric = []

    for batch in range(batch_size):

        t, p = A[batch]>0, B[batch]>0

        

        intersection = np.logical_and(t, p)

        union = np.logical_or(t, p)

        

        iou = (np.sum(intersection) + 1e-10 )/ (np.sum(union) + 1e-10)

        metric.append(iou)

        

    return np.mean(metric)





def my_iou_metric(label, pred):

    return tf.py_func(get_iou, [label, pred>0.5], tf.float64)





def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = y_true * y_pred

    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)

    

    

    return 1. - score





def bce_dice_loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
### layer 0: source; layer 1: transport



# abrir imagens

imgs = []

for i in range(3):

    img = envi.open('../input/stack_prim_source_region{}.hdr'.format(i+1))

    imgs.append(np.array(img.open_memmap(writeable = True)))



    

# criar layer transport; por source e transport no fim 

for im in imgs:

    im[:,:,0] /= 2

    im[:,:,1] = np.clip(im[:,:,1] - im[:,:,0], 0, 1)

    tmp = im[:,:,0:2].copy()

    im[:,:,0:5] = im[:,:,2::]

    im[:,:,5::] = tmp.copy()



for i in range(3):

    imgs[i] = np.concatenate((imgs[i], 1 - np.clip(imgs[i][:,:,-2] + imgs[i][:,:,-1], 0, 1)[:,:,None]), axis = 2)
# normalização das imagens

mean = 0

var = 0



scaler = StandardScaler()

for i in range(len(imgs)):

    scaler.fit(imgs[i].reshape(-1, imgs[0].shape[2]))

    

    mean = mean + scaler.mean_

    var = var + scaler.var_

mean /= len(imgs)

var /= len(imgs)



for i in range(len(imgs)):

    for band in range(n_channels):

        imgs[i][:,:,band] -= mean[band]

        imgs[i][:,:,band] /= (sqrt(var[band]))
def split_train_val(x, idx):

    nums = [0, 1, 2, 3, 4]

    del nums[idx]

    

    train = np.concatenate(list( x[i] for i in nums ), axis = 0)

    val = x[idx]

    return(train, val)
def upsample(filters, kernel_size, strides, padding):

    return UpSampling2D(strides)



K.clear_session()

def build_generator():



    imgs = Input(shape=(crop_size,crop_size,n_channels))



    c1 = Conv2D(5, (3, 3), activation='relu', padding='same') (imgs)

    c1 = BatchNormalization()(c1)

    p1 = MaxPooling2D((2, 2)) (c1)



    c2 = Conv2D(10, (3, 3), activation='relu', padding='same') (p1)

    c2 = BatchNormalization()(c2)

    p2 = MaxPooling2D((2, 2)) (c2)



    c3 = Conv2D(20, (3, 3), activation='relu', padding='same') (p2)

    c3 = BatchNormalization()(c3)

    p3 = MaxPooling2D((2, 2)) (c3)



    c4 = Conv2D(40, (3, 3), activation='relu', padding='same') (p3)

    c4 = BatchNormalization()(c4)

    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)





    c5 = Conv2D(80, (3, 3), activation='relu', padding='same') (p4)

    c5 = BatchNormalization()(c5)



    u6 = upsample(40, (2, 2), strides=(2, 2), padding='same') (c5)

    u6 = BatchNormalization()(u6)

    u6 = concatenate([u6, c4])

    c6 = Conv2D(40, (3, 3), activation='relu', padding='same') (u6)

    c6 = BatchNormalization()(c6)

    

    u7 = upsample(20, (2, 2), strides=(2, 2), padding='same') (c6)

    u7 = BatchNormalization()(u7)

    u7 = concatenate([u7, c3])

    c7 = Conv2D(20, (3, 3), activation='relu', padding='same') (u7)

    c7 = BatchNormalization()(c7)

    

    u8 = upsample(10, (2, 2), strides=(2, 2), padding='same') (c7)

    u8 = BatchNormalization()(u8)

    u8 = concatenate([u8, c2])

    c8 = Conv2D(10, (3, 3), activation='relu', padding='same') (u8)

    c8 = BatchNormalization()(c8)

    

    u9 = upsample(5, (2, 2), strides=(2, 2), padding='same') (c8)

    u9 = BatchNormalization()(u9)

    u9 = concatenate([u9, c1], axis=3)

    c9 = Conv2D(5, (3, 3), activation='relu', padding='same') (u9)

    c9 = BatchNormalization()(c9)

    c9 = SpatialDropout2D(0.1)(c9)

    

    mask = Conv2D(3, (1, 1), activation='softmax', name='final') (c9)

        

    return Model(imgs, mask)

def lets_train(data_train, data_val, data_test, prediction_scores, idx):

    class My_Reduce_Lr(Callback):

    

        def __init__(self, patience, factor, min_lr):

            super(Callback, self).__init__()



            self.patience = patience

            self.factor = factor

            self.min_lr = min_lr

            self.losses = []

            self.wait_stabilization = patience



        def on_epoch_end(self, epoch, logs={}):

            val = logs['val_jaccard_coef_int_both']

            self.losses.append(val)



            if self.wait_stabilization <= 0:

                if val < self.losses[epoch - self.patience]:

                    new_lr = K.eval(self.model.optimizer.lr)*self.factor

                    if new_lr >= self.min_lr: 

                        K.set_value(self.model.optimizer.lr, new_lr)

                    self.wait_stabilization = self.patience + 1

                    print("New learning rate of {}".format(K.eval(self.model.optimizer.lr)))



            self.wait_stabilization = self.wait_stabilization - 1 





            return

    

    

    reduce_lr = My_Reduce_Lr(patience = 5, factor = 0.2, min_lr=0.00001)

    early_stopping = EarlyStopping(patience=30, verbose=1)



    filepath="weights_best_dice.hdf5"

    checkpointdice = ModelCheckpoint(filepath, monitor='val_weighted_dice_coef', verbose=1, save_best_only=True, mode='max')



    filepath="pos.hdf5"

    checkpoint2 = ModelCheckpoint(filepath, monitor='val_true_positive_rate', verbose=1, save_best_only=True, mode='max')



    filepath="weights_best_int_nobg.hdf5"

    checkpoint3 = ModelCheckpoint(filepath, monitor='val_jaccard_coef_int_nobg', verbose=1, save_best_only=True, mode='max')



    filepath="weights_best_int_s.hdf5"

    checkpoint4 = ModelCheckpoint(filepath, monitor='val_jaccard_coef_int_source', verbose=1, save_best_only=True, mode='max')



    filepath="weights_best_int_t.hdf5"

    checkpoint5 = ModelCheckpoint(filepath, monitor='val_jaccard_coef_int_transport', verbose=1, save_best_only=True, mode='max')



    filepath="weights_best_int_b.hdf5"

    checkpoint6 = ModelCheckpoint(filepath, monitor='val_jaccard_coef_int_both', verbose=1, save_best_only=True, mode='max')





    callbacks_list = [checkpoint3, checkpoint4, checkpoint5, checkpoint6, reduce_lr, early_stopping, history]



    

    

    datagen = ImageDataGenerator(

        rotation_range=0.30,

        zoom_range=0.1,

        shear_range=0.1,

        horizontal_flip=True,

        vertical_flip=True,

        fill_mode="reflect")



    train_batch = datagen.flow(data_train, np.zeros(data_train.shape[0]), batch_size=batch_size)

    train_batch = mask_trans(train_batch)

    

    model = build_generator()



    met = [jaccard_coef_int_nobg, jaccard_coef_int_source, jaccard_coef_int_transport, jaccard_coef_int_both, 'acc']



    Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss=weighted_categorical_crossentropy(weights), optimizer=Adam, metrics=met)



    my_history = model.fit_generator(train_batch,

                  steps_per_epoch=floor(data_train.shape[0]/batch_size),

                  epochs=100,

                  verbose=1,

                  callbacks=callbacks_list,

                  validation_data=(data_val[:,:,:,0:n_channels], data_val[:,:,:,n_channels::]))

    

    

    

    prediction_scores[idx] = model.predict(data_test[:,:,:,0:5], verbose = 1, batch_size = 32)
# divisão em train, test, val

prediction_scores = dict() #dictionery to store the predicted scores of individual models on the test dataset

for i in range(3):

    

    nums = [0, 1, 2]

    del nums[i] 

     

    crop_rate = 1

    data_train_val = np.concatenate(list( create_crops(imgs[ii]) for ii in nums ), axis = 0)

    crop_rate = 0.75

    data_test = create_crops(imgs[i])



    data_train_val_split = np.array_split(data_train_val, 5)





    for j in range(5):

        data_train, data_val = split_train_val(data_train_val_split, j)

        lets_train(data_train, data_val, data_test, prediction_scores, i*5 + j)

        '''

        aux = np.argmax(prediction_scores[i*5 + j][:,:,:,[2,0,1]], axis = 3)

        aux2 = np.argmax(data_test[:,:,:,[7,5,6]], axis = 3)

        

        for jj in range(aux.shape[0]):

        

            f, axarr = plt.subplots(1,2)

            axarr[0].imshow(aux[jj], cmap='gray')

            axarr[1].imshow(aux2[jj], cmap='gray')

            plt.show()

        '''
np.save('predictions.npy', prediction_scores)