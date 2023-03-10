import os



import numpy as np



import keras

from keras import backend as K

from keras.models import Model

from keras.layers import Input

from keras.layers.core import Dropout

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

import tensorflow as tf



from tqdm import tqdm



from astropy.io import fits

import matplotlib.pyplot as plt
### Image size

IMG_SIZE=32



### CNN path

MODEL_PATH='../input/semanticsegmentation/output/'



### Input path

INPUT_PATH='../input/starsegmentation'
image_data = fits.getdata(os.path.join(INPUT_PATH, 'A01-1-001-001.fts' ), ext=0)

plt.imshow(np.log10(image_data[300:400,500:600]), cmap='seismic')

plt.colorbar()
def ztransform(x):

    return (x-np.mean(x)) / np.std(x)



N,M = IMG_SIZE,IMG_SIZE

tiles = [ztransform(image_data[x:x+M,y:y+N]) for x in range(0,image_data.shape[0],M) for y in range(0,image_data.shape[1],N) if np.max(np.log10(image_data[x:x+M,y:y+N]))>4.3]
tiles = np.asarray(tiles)

tiles.shape, tiles[0].shape
def U_NET(img_size):

    inputs = Input((img_size, img_size, 1))



    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)

    c1 = Dropout(0.1)(c1)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)

    p1 = MaxPooling2D((2, 2))(c1)



    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)

    c2 = Dropout(0.1)(c2)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)

    p2 = MaxPooling2D((2, 2))(c2)



    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)

    c3 = Dropout(0.2)(c3)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)

    p3 = MaxPooling2D((2, 2))(c3)



    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)

    c4 = Dropout(0.2)(c4)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)

    p4 = MaxPooling2D(pool_size=(2, 2))(c4)



    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)

    c5 = Dropout(0.3)(c5)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)



    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)

    u6 = concatenate([u6, c4])

    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)

    c6 = Dropout(0.2)(c6)

    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)



    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)

    u7 = concatenate([u7, c3])

    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)

    c7 = Dropout(0.2)(c7)

    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)



    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)

    u8 = concatenate([u8, c2])

    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)

    c8 = Dropout(0.1)(c8)

    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)



    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)

    u9 = concatenate([u9, c1], axis=3)

    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)

    c9 = Dropout(0.1)(c9)

    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)



    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    

    model = Model(inputs=[inputs], outputs=[outputs])

    return model



model = U_NET(IMG_SIZE)

model.load_weights(os.path.join(MODEL_PATH,'weights.h5'))
for tile in tiles[120:150]:

    mask = model.predict(tile.reshape((1,IMG_SIZE,IMG_SIZE,1)))

    mask = (mask>0.4).astype(np.int8)

    ax=plt.subplot(1,2,1)

    ax.imshow(tile.reshape((IMG_SIZE,IMG_SIZE)))

    ax=plt.subplot(1,2,2)

    ax.imshow(mask.reshape((IMG_SIZE,IMG_SIZE)))

    plt.show()