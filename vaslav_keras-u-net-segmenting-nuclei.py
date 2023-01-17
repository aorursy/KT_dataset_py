import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from IPython.display import clear_output

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf
# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 242
random.seed = seed
np.random.seed = seed
# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
"train = " +  str(len(train_ids)) +" | test = " +  str(len(test_ids))
X_train = None
Y_train = None
# Get and resize train images and masks
def loadTrainImagesAndMasks():
    global X_train
    global Y_train
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    minMask=99999
    maxMask=0
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        i = 0
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            i += 1
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        #print( str(n) +':' +  str(i)  +':')
        if i < minMask: minMask = i
        if i > maxMask: maxMask = i    
        Y_train[n] = mask

    print( 'minMask = ' + str(minMask) +'| maxMask = ' +  str(maxMask)  +':')

# Get and resize test images
def getTestData():
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        #img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        img = imread(path + '/images/' + id_ + '.png')
        if len(img.shape) == 2: # gray
            max_ = np.max(img)
            print(n,img.shape,'max:', max_,id_) # ,np.min(img),np.max(img)
            if max_ > 256:                # not "uint8"
                img = img/max_            # normalize 
                img = (img * 255).round().astype(np.uint8)
                img = np.stack([img,img,img], axis=2).astype("uint8") 
        elif img.shape[2] == 4:        # remove alpha channel
             img = img[:,:,:IMG_CHANNELS]        
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img
    print('Loaded test images', len(X_test))
    return X_test,sizes_test        
# Check if training data looks all right
def CheckTrainData():
    ix = random.randint(0, len(train_ids))
    imshow(X_train[ix])
    plt.show()
    imshow(np.squeeze(Y_train[ix]))
    plt.show()
# CheckTrainData()
# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2, y_true)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
# Dice coefficient
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Build U-Net model
def BuildModel(bg2 = 4,_Height=256, _Width=256, _Channels=3, outCannels = 1):
    sAct = 'elu'          # activation ['relu','elu'][1]
    sK_init = 'he_normal' # kernel_initializer
    sPad = 'same'         # padding
    input_name = 'input_h_w'
   
    inputs = Input((_Height, _Width, _Channels),name=input_name)
    s = Lambda(lambda x: x / 255) (inputs)
   
    c1 = Conv2D(2**(bg2+0), (3, 3), activation=sAct, kernel_initializer=sK_init, 
                padding=sPad, name='cv1_1') (s)
    c1 = Dropout(0.1, name='do1_2') (c1)
    c1 = Conv2D(2**(bg2+0), (3, 3), activation=sAct, kernel_initializer=sK_init, 
                padding=sPad, name='cv1_3') (c1)
    p1 = MaxPooling2D((2, 2), name='mp1_4') (c1)
    #p1 = BatchNormalization()(p1)
    
    c2 = Conv2D(2**(bg2+1), (3, 3), activation=sAct, kernel_initializer=sK_init,
                padding=sPad, name='cv2_1') (p1)
    c2 = Dropout(0.1, name='do2_2') (c2)
    c2 = Conv2D(2**(bg2+1), (3, 3), activation=sAct, kernel_initializer=sK_init,
                padding=sPad, name='cv2_3') (c2)
    p2 = MaxPooling2D((2, 2), name='mp2_4') (c2)
    #p2 = BatchNormalization()(p2)
    
    c3 = Conv2D(2**(bg2+2), (3, 3), activation=sAct, kernel_initializer=sK_init,
                padding=sPad, name='cv3_1') (p2)
    c3 = Dropout(0.2, name='do3_2') (c3)
    c3 = Conv2D(2**(bg2+2), (3, 3), activation=sAct, kernel_initializer=sK_init,
                padding=sPad, name='cv3_3') (c3)
    p3 = MaxPooling2D((2, 2), name='mp3_4') (c3)
    #p3 = BatchNormalization()(p3)
    
    c4 = Conv2D(2**(bg2+3), (3, 3), activation=sAct, kernel_initializer=sK_init,
                padding=sPad, name='cv4_1') (p3)
    c4 = Dropout(0.2, name='do4_2') (c4)
    c4 = Conv2D(2**(bg2+3), (3, 3), activation=sAct, kernel_initializer=sK_init,
                padding=sPad, name='cv4_3') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2), name='mp4_4') (c4)
    #p4 = BatchNormalization()(p4)
    ####=======================================================================
    c5 = Conv2D(2**(bg2+4), (3, 3), activation=sAct, kernel_initializer=sK_init,
                padding=sPad, name='cv5_1') (p4)
    c5 = Dropout(0.3, name='do5_2') (c5)
    c5 = Conv2D(2**(bg2+4), (3, 3), activation=sAct, kernel_initializer=sK_init,
                padding=sPad, name='cv5_3') (c5)
    ####=======================================================================
    u6 = Conv2DTranspose(2**(bg2+3), (2, 2), strides=(2, 2),
                         padding=sPad, name='up6_1') (c5)
    u6 = concatenate([u6, c4], name='up6_2')
    c6 = Conv2D(2**(bg2+3), (3, 3), activation=sAct, kernel_initializer=sK_init,
                padding=sPad, name='cv6_3') (u6)
    c6 = Dropout(0.2, name='do6_4') (c6)
    c6 = Conv2D(2**(bg2+3), (3, 3), activation=sAct, kernel_initializer=sK_init,
                padding=sPad, name='cv6_5') (c6)
    
    u7 = Conv2DTranspose(2**(bg2+2), (2, 2), strides=(2, 2),
                         padding=sPad, name='up7_1') (c6)
    u7 = concatenate([u7, c3], name='up7_2')
    c7 = Conv2D(2**(bg2+2), (3, 3), activation=sAct, kernel_initializer=sK_init,
                padding=sPad, name='cv7_3') (u7)
    c7 = Dropout(0.2, name='do7_4') (c7)
    c7 = Conv2D(2**(bg2+2), (3, 3), activation=sAct, kernel_initializer=sK_init,
                padding=sPad, name='cv7_5') (c7)
    #c7 = BatchNormalization()(c7)
    
    u8 = Conv2DTranspose(2**(bg2+1), (2, 2), strides=(2, 2),
                         padding=sPad, name='up8_1') (c7)
    u8 = concatenate([u8, c2], name='up8_2')
    c8 = Conv2D(2**(bg2+1), (3, 3), activation=sAct, kernel_initializer=sK_init,
                padding=sPad, name='cv8_3') (u8)
    c8 = Dropout(0.1, name='do8_4') (c8)
    c8 = Conv2D(2**(bg2+1), (3, 3), activation=sAct, kernel_initializer=sK_init,
                padding=sPad, name='cv8_5') (c8)
    #c8 = BatchNormalization()(c8)
    
    u9 = Conv2DTranspose(2**(bg2+0), (2, 2), strides=(2, 2),
                         padding=sPad, name='up9_1') (c8)
    u9 = concatenate([u9, c1], axis=3, name='up9_2')
    c9 = Conv2D(2**(bg2+0), (3, 3), activation=sAct, kernel_initializer=sK_init,
                padding=sPad, name='cv9_3') (u9)
    c9 = Dropout(0.1, name='do9_4') (c9)
    c9 = Conv2D(2**(bg2+0), (3, 3), activation=sAct, kernel_initializer=sK_init,
                padding=sPad, name='cv9_5') (c9)
    #c9 = BatchNormalization()(c9)
    
    outputs = Conv2D(outCannels, (1, 1), activation='sigmoid',name='outputs') (c9)
    return Model(inputs=[inputs], outputs=[outputs], name='U-Net' + str(2**(bg2)))
def polt_train(results,lbl_trn):
    lbl_val = 'val_'+lbl_trn
    plt.plot(results.epoch, results.history[lbl_trn], label="trn_" + lbl_trn + \
            " {0:.4f}".format(results.history[lbl_trn][-1]))
    plt.plot(results.epoch, results.history[lbl_val], label=lbl_val + \
            " {0:.4f}".format(results.history[lbl_val][-1]))
    plt.xlabel('Epochs')
    plt.ylabel(lbl_trn)
    plt.legend()
    plt.show()
def polt_train2(results,lbl_trn0,lbl_trn1):
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))

    lbl_val0 = 'val_'+lbl_trn0
    lbl_val1 = 'val_'+lbl_trn1
    axes[0].plot(results.epoch, results.history[lbl_trn0], label="trn_" + lbl_trn0 + \
                " {0:.4f}".format(results.history[lbl_trn0][-1]))
    axes[0].plot(results.epoch, results.history[lbl_val0], label=lbl_val0 + \
                " {0:.4f}".format(results.history[lbl_val0][-1]))
    #axes[0].set_title("<{}>".format(lbl_val0))
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel(lbl_trn0)
    axes[1].plot(results.epoch, results.history[lbl_trn1], label="trn_" + lbl_trn1 + \
                 " {0:.4f}".format(results.history[lbl_trn1][-1]))
    axes[1].plot(results.epoch, results.history[lbl_val1], label=lbl_val1 + \
                " {0:.4f}".format(results.history[lbl_val1][-1]))
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel(lbl_trn1)
    plt.legend()
    plt.show()    
# Fit model
def processFit(model, sModelCheckpoint='model-dsbowl2018-0.h5'):
    earlystopper = EarlyStopping(patience=10, verbose=1)
    checkpointer = ModelCheckpoint(sModelCheckpoint, verbose=1, save_best_only=True)
    for batch_size in (64,32,16):
        # load last model
        if os.path.isfile(sModelCheckpoint)==True:
            print(" LOAD model: ",sModelCheckpoint,'Batch Size:',batch_size)
            model =  load_model(sModelCheckpoint, custom_objects={'mean_iou': mean_iou,
                                                                  'dice_coef': dice_coef,})
        print('Start. Fit Model => Epochs:',iEpochs, 'batch_size:', batch_size)
        # ==========
        results = model.fit(X_train, Y_train, validation_split=validationSplit, 
                             batch_size=batch_size, epochs=iEpochs,
                             callbacks=[earlystopper, checkpointer],
                             verbose=2,
                           )
        # ==========
        sModelMidle = sModelCheckpoint+'-'+str(iEpochs) \
                                         + 'vl=' + str(results.history['val_loss'][-1]) \
                                         + '-minVL=' + str(min(results.history['val_loss'])) \
                                         + '.h5'
        model.save(sModelMidle)
        clear_output()
        polt_train(results,'mean_iou')
        polt_train2(results,'loss','dice_coef') 
# Threshold predictions
threshold = 0.62
from skimage import morphology
def imagePostProcessing(img,_threshold = threshold):
    img = (img > _threshold).astype(np.uint8)
    eroded = morphology.erosion(img, morphology.square(3))
    dilated = morphology.dilation(eroded, morphology.square(3))
    return dilated
# Threshold predictions
threshold = 0.60

# Create list of upsampled test masks (threshold)
def makeUpsampledPostProcessingResizeMasks(thhd = threshold):    # -> list[numpy.ndarray]
    '''Create list of upsampled test masks'''
    # global preds_test
    # global sizes_test
    preds_test_upsampled = []
    for i in range(len(preds_test)):
        preds_test_upsampled.append(resize(
                                    imagePostProcessing(np.squeeze(preds_test[i]),thhd), 
                                           (sizes_test[i][0], sizes_test[i][1]), 
                                           mode='constant', preserve_range=True)
                                   )
    return preds_test_upsampled
# Create list of upsampled test masks
def makeUpsampledResizeMasks(preds_test):    # -> list[numpy.ndarray]
    preds_test_upsampled = []
    # global preds_test
    # global sizes_test
    for i in range(len(preds_test)):
        preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                        mode='constant', preserve_range=True))
    return preds_test_upsampled   
# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
# Create submission DataFrame
def submission(fileCSV='sub-dsbowl2018-1.csv'):
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(fileCSV, index=False)
%%time
loadTrainImagesAndMasks()
X_test,sizes_test = getTestData()

model = BuildModel()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou,dice_coef])
print(model.summary)
%%time
# Fit model
validationSplit = 0.1 # validation split
iEpochs = 12
sModelCheckpoint ='model-dsbowl2018.h5'# 'model-dsbowl2018-351.h5'

processFit(model, sModelCheckpoint)
%ls -l
# load last model
if os.path.isfile(sModelCheckpoint)==True:
    print(" LOAD model: ",sModelCheckpoint)
    model =  load_model(sModelCheckpoint, custom_objects={'mean_iou': mean_iou,  'dice_coef': dice_coef,})
score = model.evaluate(X_train, Y_train, verbose=1, batch_size=16)
print('Test loss:', score[0],'Test accuracy:', score[1])
prc = 1 - validationSplit
val_count=int(X_train.shape[0]*prc)
score = model.evaluate(X_train[:val_count], Y_train[:val_count], verbose=1, batch_size=16)
print('Val loss:', score[0],'Val accuracy:', score[1])
# Predict on Valid
preds_train = model.predict(X_train[:val_count], verbose=1)
# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train))
imshow(X_train[ix]); plt.show()
imshow(np.squeeze(Y_train[ix]));plt.show()
imshow(np.squeeze(preds_train[ix]));plt.show()
val_count=int(X_train.shape[0]*prc)
# Predict on  val
preds_val = model.predict(X_train[val_count:], verbose=1)
# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val))
imshow(X_train[val_count:][ix])
plt.show()
imshow(np.squeeze(Y_train[val_count:][ix]))
plt.show()
imshow(np.squeeze(preds_val[ix]))
plt.show()

# Predict test
preds_test = model.predict(X_test, verbose=1)
# Create list of upsampled test masks
bPostProcessing=True
if bPostProcessing:
    preds_test_upsampled = makeUpsampledPostProcessingResizeMasks()
else:  
    preds_test_upsampled = makeUpsampledResizeMasks(preds_test)
new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))
submission('sub351.csv')
