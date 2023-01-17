import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import os

from tqdm import tqdm_notebook

import cv2



import keras

from keras.layers.convolutional import Conv2DTranspose

from keras.layers.merge import concatenate

from keras.layers import UpSampling2D, Conv2D, Activation, Input, Dropout, MaxPooling2D

from keras import Model

from keras import backend as K

from keras.layers.core import Lambda
tr = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')

print(len(tr))

tr.head()
df_train = tr[tr['EncodedPixels'].notnull()].reset_index(drop=True)

print(len(df_train))

df_train.head()
def rle2mask(rle, imgshape):

    width = imgshape[0]

    height= imgshape[1]

    

    mask= np.zeros( width*height ).astype(np.uint8)

    

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        mask[int(start):int(start+lengths[index])] = 1

        current_position += lengths[index]

        

    return np.flipud( np.rot90( mask.reshape(height, width), k=1 ) )
img_size = 256
def keras_generator(batch_size):

    while True:

        x_batch = []

        y_batch = []

        

        for i in range(batch_size):            

            fn = df_train['ImageId_ClassId'].iloc[i].split('_')[0]

            img = cv2.imread( '../input/severstal-steel-defect-detection/train_images/'+fn )

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            

            

            mask = rle2mask(df_train['EncodedPixels'].iloc[i], img.shape)

            

            img = cv2.resize(img, (img_size, img_size))

            mask = cv2.resize(mask, (img_size, img_size))

            

            x_batch += [img]

            y_batch += [mask]

                                    

        x_batch = np.array(x_batch)

        y_batch = np.array(y_batch)



        yield x_batch, np.expand_dims(y_batch, -1)
for x, y in keras_generator(16):

    break
from keras.applications.resnet50 import ResNet50

from keras.layers import Input

weights = '../input/resnet50-weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

img_shape = (256,256,3)

img_input = Input(shape=img_shape)

base_model = ResNet50(weights=weights, input_shape=img_shape, include_top=False,

                     input_tensor=img_input, classes=None)

base_out = base_model.output



up = Conv2D(1, (1,1), strides=(1, 1))(base_out)

up = UpSampling2D(size=(32, 32), interpolation='bilinear')(up)





model = Model(input=base_model.input, output=up)
base_model.summary()
best_w = keras.callbacks.ModelCheckpoint('resnet_best.h5',

                                monitor='val_loss',

                                verbose=0,

                                save_best_only=True,

                                save_weights_only=True,

                                mode='auto',

                                period=1)



last_w = keras.callbacks.ModelCheckpoint('resnet_last.h5',

                                monitor='val_loss',

                                verbose=0,

                                save_best_only=False,

                                save_weights_only=True,

                                mode='auto',

                                period=1)





callbacks = [best_w, last_w]







adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)





model.compile(adam, 'binary_crossentropy')
batch_size = 16

model.fit_generator(keras_generator(batch_size),

              steps_per_epoch=50,

              epochs=3,

              callbacks = callbacks)
"""

%%time





# Fit model

batch_size = 16

results = model.fit_generator(keras_generator(batch_size), 

                              steps_per_epoch=64,

                              epochs=2

                              )"""
pred = model.predict(x)

testfiles=os.listdir("../input/severstal-steel-defect-detection/test_images/")



test_img = []

for fn in tqdm_notebook(testfiles):

        img = cv2.imread( '../input/severstal-steel-defect-detection/test_images/'+fn )

        img = cv2.resize(img,(img_size,img_size))       

        test_img.append(img)

        



predict = model.predict(np.asarray(test_img))
def mask2rle(img):

    tmp = np.rot90( np.flipud( img ), k=3 )

    rle = []

    lastColor = 0;

    startpos = 0

    endpos = 0



    tmp = tmp.reshape(-1,1)   

    for i in range( len(tmp) ):

        if (lastColor==0) and tmp[i]>0:

            startpos = i

            lastColor = 1

        elif (lastColor==1)and(tmp[i]==0):

            endpos = i-1

            lastColor = 0

            rle.append( str(startpos)+' '+str(endpos-startpos+1) )

    return " ".join(rle)
pred_rle = []

for img in predict:      

    img = cv2.resize(img, (1600, 256))

    tmp = np.copy(img)

    tmp[tmp<np.mean(img)] = 0

    tmp[tmp>0] = 1

    pred_rle.append(mask2rle(tmp))
sub = pd.read_csv( '../input/severstal-steel-defect-detection/sample_submission.csv' )

for fn, rle in zip(testfiles, pred_rle):

    sub['EncodedPixels'][sub['ImageId_ClassId'].apply(lambda x: x.split('_')[0]) == fn] = rle

    

img_s = cv2.imread( '../input/severstal-steel-defect-detection/test_images/'+ sub['ImageId_ClassId'][16].split('_')[0])

mask_s = rle2mask(sub['EncodedPixels'][16], (256, 1600))
sub.to_csv('submission.csv', index=False)