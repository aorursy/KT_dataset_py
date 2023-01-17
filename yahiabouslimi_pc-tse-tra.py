# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os
import numpy as np

import sys 
import random
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt



import cv2

from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model
from keras import backend as K 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# seed to get the same result every time
seed = 69
np.random.seed = seed
# initializing image size and width and channels
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
# initializing training & testing path
TRAIN_PATH = '../input/t2-tsatra-prostate-cancer-mri/data/train/img/'
Y_TRAIN_PATH = '../input/t2-tsatra-prostate-cancer-mri/data/train/mask/'
TEST_PATH = '../input/t2-tsatra-prostate-cancer-mri/data/test/img/'
Y_TEST_PATH = '../input/t2-tsatra-prostate-cancer-mri/data/test/mask/'

# get a list of all training folder names
train_ids = next(os.walk(TRAIN_PATH))[1]
print (len(train_ids))
y_train_ids = next(os.walk(Y_TRAIN_PATH))[1]
print (len(y_train_ids))
# get a list of all testing folder names
test_ids = next(os.walk(TEST_PATH))[1]
print (len(test_ids))
y_test_ids = next(os.walk(Y_TEST_PATH))[1]
print (len(y_test_ids))
# sort the list of all training & testing folder names
train_ids = sorted(train_ids)
y_train_ids = sorted(y_train_ids)
test_ids = sorted(test_ids)
y_test_ids = sorted(y_test_ids)
# creating the X_train array:

lstFilesPNG = []  # create an empty list, the raw image data files is stored here
for dirName, subdirList, fileList in os.walk(TRAIN_PATH):
    for filename in fileList:
        if ".png" in filename.lower():  # check whether the file's DICOM
            lstFilesPNG.append(os.path.join(dirName,filename))
# len(lstFilesPNG)

lstPNG =  lstFilesPNG
lstPNG = sorted(lstPNG)
len(lstPNG)


import re
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


lstPNG.sort(key=natural_keys)
lstPNG[:18]
# put TRAIN images in a numpy array
X_train = np.zeros((len(lstPNG), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
# 1 mean BOOL (y/n)
# Y_train = np.zeros((len(lstPNG), IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool)
idx=0
for filenamePNG in lstPNG:
    # read the file
    img = imread(filenamePNG)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH, 1), mode = 'constant', preserve_range = True)
    X_train[idx] = img #fill empty X_Train with value from img
    idx=idx+1 
len(lstPNG) 
print (X_train.shape)
# creating the X_test array:

lsttestFilesPNG = []  # create an empty list, the raw image data files is stored here
for dirName, subdirList, fileList in os.walk(TEST_PATH):
    for filename in fileList:
        if ".png" in filename.lower():  # check whether the file's DICOM
            lsttestFilesPNG.append(os.path.join(dirName,filename))
# len(lstFilesPNG)

lsttestPNG =  lsttestFilesPNG
lsttestPNG = sorted(lsttestPNG)

lsttestPNG.sort(key=natural_keys)
lsttestPNG[:18]
# put TRAIN images in a numpy array
X_test = np.zeros((len(lsttestPNG), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
# 1 mean BOOL (y/n)
# Y_train = np.zeros((len(lstPNG), IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool)
idx=0
for filenamePNG in lsttestPNG:
    # read the file
    img = imread(filenamePNG)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH, 1), mode = 'constant', preserve_range = True)
    X_test[idx] = img #fill empty X_Train with value from img
    idx=idx+1  
# Y_train = np.zeros((len(lstPNG), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
Y_train = np.zeros((len(lstPNG), IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool)
lstMaskPNG = []  # create an empty list, the raw image data files is stored here
for dirName, subdirList, fileList in os.walk(Y_TRAIN_PATH):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's DICOM
            lstMaskPNG.append(os.path.join(dirName,filename))
lstMaskPNG.remove('../input/t2-tsatra-prostate-cancer-mri/data/train/mask/.ipynb_checkpoints/ProstateX-0000-10-checkpoint.jpg')

# len(lstFilesPNG)

lst_m_PNG =  lstMaskPNG
lst_m_PNG = sorted(lst_m_PNG)
lst_m_PNG.sort(key=natural_keys)
lst_m_PNG[:18]


idx=0
mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool)
for masknamePNG in lst_m_PNG:
    mask_ = imread(masknamePNG)
    mask = resize(mask_, (IMG_HEIGHT, IMG_WIDTH, 1), mode = 'constant', preserve_range = True)
    Y_train[idx] = mask #fill empty X_Train with value from img
    idx=idx+1 
print(Y_train.shape)
print(X_train.shape)

image_x = random.randint(0, 122)
print(image_x)
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()
imshow(lst_m_PNG[image_x])
plt.show()
def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x


def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
    '''
    2D Transposed Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(2, 2)})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    
    return x


def MultiResBlock(U, inp, alpha = 1.67):
    '''
    MultiRes Block
    
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''

    W = alpha * U

    shortcut = inp

    shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                         int(W*0.5), 1, 1, activation=None, padding='same')

    conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                        activation='relu', padding='same')

    conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                        activation='relu', padding='same')

    conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                        activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
    out = BatchNormalization(axis=3)(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    return out


def ResPath(filters, length, inp):
    '''
    ResPath
    
    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''


    shortcut = inp
    shortcut = conv2d_bn(shortcut, filters, 1, 1,
                         activation=None, padding='same')

    out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization(axis=3)(out)

    for i in range(length-1):

        shortcut = out
        shortcut = conv2d_bn(shortcut, filters, 1, 1,
                             activation=None, padding='same')

        out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization(axis=3)(out)

    return out


def MultiResUnet(height, width, n_channels):
    '''
    MultiResUNet
    
    Arguments:
        height {int} -- height of image 
        width {int} -- width of image 
        n_channels {int} -- number of channels in image
    
    Returns:
        [keras model] -- MultiResUNet model
    '''


    inputs = Input((height, width, n_channels))

    mresblock1 = MultiResBlock(32, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = ResPath(32, 4, mresblock1)

    mresblock2 = MultiResBlock(32*2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = ResPath(32*2, 3, mresblock2)

    mresblock3 = MultiResBlock(32*4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = ResPath(32*4, 2, mresblock3)

    mresblock4 = MultiResBlock(32*8, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = ResPath(32*8, 1, mresblock4)

    mresblock5 = MultiResBlock(32*16, pool4)

    up6 = concatenate([Conv2DTranspose(
        32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
    mresblock6 = MultiResBlock(32*8, up6)

    up7 = concatenate([Conv2DTranspose(
        32*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
    mresblock7 = MultiResBlock(32*4, up7)

    up8 = concatenate([Conv2DTranspose(
        32*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
    mresblock8 = MultiResBlock(32*2, up8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(mresblock8), mresblock1], axis=3)
    mresblock9 = MultiResBlock(32, up9)

    conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid')
    
    model = Model(inputs=[inputs], outputs=[conv10])

    return model

def dice_coef(y_true, y_pred):
    smooth = 0.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jacard(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum ( y_true_f * y_pred_f)
    union = K.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection/union
def saveModel(model):

    model_json = model.to_json()

    try:
        os.makedirs('models')
    except:
        pass
    
    fp = open('models/modelP.json','w')
    fp.write(model_json)
    model.save_weights('models/modelW.h5')
def evaluateModel(model, X_test, Y_test, batchSize):
    
    try:
        os.makedirs('results')
    except:
        pass 
    

    yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)

    yp = np.round(yp,0)

    for i in range(10):

        plt.figure(figsize=(20,10))
        plt.subplot(1,3,1)
        plt.imshow(X_test[i])
        plt.title('Input')
        plt.subplot(1,3,2)
        plt.imshow(Y_test[i].reshape(Y_test[i].shape[0],Y_test[i].shape[1]))
        plt.title('Ground Truth')
        plt.subplot(1,3,3)
        plt.imshow(yp[i].reshape(yp[i].shape[0],yp[i].shape[1]))
        plt.title('Prediction')

        intersection = yp[i].ravel() * Y_test[i].ravel()
        union = yp[i].ravel() + Y_test[i].ravel() - intersection

        jacard = (np.sum(intersection)/np.sum(union))  
        plt.suptitle('Jacard Index'+ str(np.sum(intersection)) +'/'+ str(np.sum(union)) +'='+str(jacard))

        plt.savefig('results/'+str(i)+'.png',format='png')
        plt.close()


    jacard = 0
    dice = 0
    
    
    for i in range(len(Y_test)):
        yp_2 = yp[i].ravel()
        y2 = Y_test[i].ravel()
        
        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection

        jacard += (np.sum(intersection)/np.sum(union))  

        dice += (2. * np.sum(intersection) ) / (np.sum(yp_2) + np.sum(y2))

    
    jacard /= len(Y_test)
    dice /= len(Y_test)
    


    print('Jacard Index : '+str(jacard))
    print('Dice Coefficient : '+str(dice))
    

    fp = open('models/log.txt','a')
    fp.write(str(jacard)+'\n')
    fp.close()

    fp = open('models/best.txt','r')
    best = fp.read()
    fp.close()

    if(jacard>float(best)):
        print('***********************************************')
        print('Jacard Index improved from '+str(best)+' to '+str(jacard))
        print('***********************************************')
        fp = open('models/best.txt','w')
        fp.write(str(jacard))
        fp.close()

        saveModel(model)


lst_Y_testFilesPNG = []  # create an empty list, the raw image data files is stored here
for dirName, subdirList, fileList in os.walk(TEST_PATH):
    for filename in fileList:
        if ".png" in filename.lower():  # check whether the file's DICOM
            lst_Y_testFilesPNG.append(os.path.join(dirName,filename))
# len(lstFilesPNG)

lst_Y_testPNG =  lst_Y_testFilesPNG
lst_Y_testPNG = sorted(lst_Y_testPNG)
lst_Y_testPNG = sorted(lst_Y_testPNG)
lst_Y_testPNG.sort(key=natural_keys)
len(lst_Y_testPNG)


# Y_train = np.zeros((len(lstPNG), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
Y_test = np.zeros((len(lst_Y_testPNG), IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool)
lst_Y_MaskPNG = []  # create an empty list, the raw image data files is stored here
for dirName, subdirList, fileList in os.walk(Y_TEST_PATH):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's DICOM
            lst_Y_MaskPNG.append(os.path.join(dirName,filename))
# len(lstFilesPNG)

lst_Y_m_PNG =  lst_Y_MaskPNG
lst_Y_m_PNG = sorted(lst_Y_m_PNG)
lst_Y_m_PNG = sorted(lst_Y_m_PNG)
lst_Y_m_PNG.sort(key=natural_keys)
len(lst_Y_m_PNG)

idx=0
mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool)
for masknamePNG in lst_Y_m_PNG:
    # read the file
    mask_ = imread(masknamePNG)
#     print (mask_.shape)
    mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH, 1), mode = 'constant', preserve_range = True)
#     print (mask_.shape)
    mask = np.maximum(mask, mask_)
#     print (mask.shape)
#     print (Y_train[idx].shape)
    Y_test[idx] = mask #fill empty X_Train with value from img
    idx=idx+1 
def trainStep(model, X_train, Y_train, X_test, Y_test, epochs, batchSize):

    history = [] #Creating a empty list for holding the loss later
    accuracy = []
    dice_coef = []
    jackard = []
    loss = []
    for epoch in range(epochs):
        print('Epoch : {}'.format(epoch+1))
        result = model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=1, verbose=1, callbacks=[history])     

        evaluateModel(model,X_test, Y_test,batchSize)
        history.append(result.history()) #Now append the loss after the training to the list.
        accuracy.append(result.history['accuracy'])
        loss.append(result.history['loss'])
        dice_coef.append(result.history['dice_coef'])
        jacard.append(result.history['jacard'])

    return model
from keras.callbacks import History 
import keras
from matplotlib import pyplot as plt
from keras.callbacks import History 
class SensitivitySpecificityCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        i = 0
        if epoch == i:
            evaluateModel(model,X_test, Y_test,10)
        i = i+1



epc = 100
model = MultiResUnet(height=128, width=128, n_channels=3)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef, jacard, 'accuracy'])

saveModel(model)

fp = open('models/log.txt','w')
fp.close()
fp = open('models/best.txt','w')
fp.write('-1.0')
fp.close()

history = model.fit(X_train, Y_train, 
                    validation_data = (X_test, Y_test), 
                    epochs=epc, 
                    batch_size=10, 
                    shuffle= True,
                    verbose = 1,
                    callbacks=[SensitivitySpecificityCallback()])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val'], loc='upper left')
plt.show()
plt.plot(history.history['jacard'])
plt.plot(history.history['val_jacard'])
plt.title('model jacard')
plt.ylabel('jacard')
plt.xlabel('epoch')
plt.legend(['jac', 'val'], loc='upper left')
plt.show()
plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('model dice_coef')
plt.ylabel('dice_coef')
plt.xlabel('epoch')
plt.legend(['dc', 'vdc'], loc='upper left')
plt.show()
idx = random.randint(0, len(X_train))

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose = 1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose = 1)
preds_test = model.predict(X_test, verbose = 1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test = (preds_test > 0.5).astype(np.uint8)
#perform some sanity test for some random training sample
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()
#perform some sanity test for some random training sample
ix = random.randint(0, 18)
imshow(X_train[int(X_test.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_test[int(Y_test.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()
#perform some sanity test for some random training sample
ix = random.randint(0, 25)
imshow(X_test[int(X_test.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_test[int(Y_test.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()
# imshow(preds_val_t[ix])
# plt.show()
# x = 
evaluateModel(model,X_test, Y_test,10)
#perform some sanity test for some random training sample
ix = random.randint(0, 66)
imshow(X_test[int(X_test.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_test[ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()
# imshow(preds_val_t[ix])
# plt.show()
# x = 
for i in range(18):
    ix = i
    imshow(X_test[int(X_test.shape[0]*0.9):][ix])
    plt.show()
    imshow(np.squeeze(Y_test[ix]))
    plt.show()
    imshow(np.squeeze(preds_val_t[ix]))
    plt.show()

