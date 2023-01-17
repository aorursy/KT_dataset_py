import os



import numpy as np

import pandas as pd



import keras

from keras import backend as K

from keras.models import Model

from keras.layers import Input

from keras.layers.core import Dropout

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import Callback

import tensorflow as tf



from tqdm import tqdm



from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt
!mkdir output

!ls
### Random seed

SEED=100



### Image size

IMG_SIZE=32



### Output path

OUTPUT_PATH='./output'



### Input path

INPUT_PATH='../input/starsegmentation'
os.listdir(INPUT_PATH)
X = np.load(os.path.join(INPUT_PATH,'IMG.npy'))

y = np.load(os.path.join(INPUT_PATH,'MASK.npy'))

df = pd.read_csv(os.path.join(INPUT_PATH,'metainfo.csv'))

ids = np.arange(len(y))



y = (y>0).astype(np.int8)



X = np.expand_dims(X, axis=3) # N_images, naxis_1, naxis_2, n_bands

y = np.expand_dims(y, axis=3)



X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, ids, test_size=0.2, random_state=SEED)



X_train.shape, X_test.shape, y_train.shape, y_test.shape
for i in range(6):

    ax=plt.subplot(1,2,1)

    ax.imshow(X[i].reshape((IMG_SIZE,IMG_SIZE)), cmap='gray')

    ax=plt.subplot(1,2,2)

    ax.imshow(y[i].reshape((IMG_SIZE,IMG_SIZE)), cmap='gray')

    plt.show()
def set_tf(seed):

    K.set_image_data_format('channels_last')

    np.random.seed(seed)

    tf.random.set_seed(seed)



def dice(y_true, y_pred, smooth=1):

    """

    Dice = (2*|X & Y|)/ (|X|+ |Y|)

         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))

    ref: https://arxiv.org/pdf/1606.04797v1.pdf

    """

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)

    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)



def U_NET(img_size):

    """

    U-net (almost) original model

    ref: https://arxiv.org/abs/1505.04597

    """

    inputs = Input((img_size, img_size, 1))



    c1 = Conv2D(16, (3, 3), activation='elu', 

                kernel_initializer='he_normal', padding='same')(inputs)

    c1 = Dropout(0.1)(c1)

    c1 = Conv2D(16, (3, 3), activation='elu', 

                kernel_initializer='he_normal', padding='same')(c1)

    p1 = MaxPooling2D((2, 2))(c1)



    c2 = Conv2D(32, (3, 3), activation='elu', 

                kernel_initializer='he_normal', padding='same')(p1)

    c2 = Dropout(0.1)(c2)

    c2 = Conv2D(32, (3, 3), activation='elu', 

                kernel_initializer='he_normal', padding='same')(c2)

    p2 = MaxPooling2D((2, 2))(c2)



    c3 = Conv2D(64, (3, 3), activation='elu', 

                kernel_initializer='he_normal', padding='same')(p2)

    c3 = Dropout(0.2)(c3)

    c3 = Conv2D(64, (3, 3), activation='elu', 

                kernel_initializer='he_normal', padding='same')(c3)

    p3 = MaxPooling2D((2, 2))(c3)



    c4 = Conv2D(128, (3, 3), activation='elu', 

                kernel_initializer='he_normal', padding='same')(p3)

    c4 = Dropout(0.2)(c4)

    c4 = Conv2D(128, (3, 3), activation='elu', 

                kernel_initializer='he_normal', padding='same')(c4)

    p4 = MaxPooling2D(pool_size=(2, 2))(c4)



    c5 = Conv2D(256, (3, 3), activation='elu', 

                kernel_initializer='he_normal', padding='same')(p4)

    c5 = Dropout(0.3)(c5)

    c5 = Conv2D(256, (3, 3), activation='elu', 

                kernel_initializer='he_normal', padding='same')(c5)



    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)

    u6 = concatenate([u6, c4])

    c6 = Conv2D(128, (3, 3), activation='elu', 

                kernel_initializer='he_normal', padding='same')(u6)

    c6 = Dropout(0.2)(c6)

    c6 = Conv2D(128, (3, 3), activation='elu', 

                kernel_initializer='he_normal', padding='same')(c6)



    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)

    u7 = concatenate([u7, c3])

    c7 = Conv2D(64, (3, 3), activation='elu', 

                kernel_initializer='he_normal', padding='same')(u7)

    c7 = Dropout(0.2)(c7)

    c7 = Conv2D(64, (3, 3), activation='elu', 

                kernel_initializer='he_normal', padding='same')(c7)



    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)

    u8 = concatenate([u8, c2])

    c8 = Conv2D(32, (3, 3), activation='elu', 

                kernel_initializer='he_normal', padding='same')(u8)

    c8 = Dropout(0.1)(c8)

    c8 = Conv2D(32, (3, 3), activation='elu', 

                kernel_initializer='he_normal', padding='same')(c8)



    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)

    u9 = concatenate([u9, c1], axis=3)

    c9 = Conv2D(16, (3, 3), activation='elu', 

                kernel_initializer='he_normal', padding='same')(u9)

    c9 = Dropout(0.1)(c9)

    c9 = Conv2D(16, (3, 3), activation='elu', 

                kernel_initializer='he_normal', padding='same')(c9)



    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    

    model = Model(inputs=[inputs], outputs=[outputs])

    return model



class PlotValCallback(Callback):

    """

    Callback to show the segmentation on the validation images

    """

    def __init__(self, model, x_val, y_val):

        self.model = model

        self.x_val = x_val

        self.y_val = y_val



    def on_epoch_end(self, epoch, logs={}):

        mask_p = self.model.predict(self.x_val)

        mask = self.y_val

        n = len(self.x_val)

        plt.figure(figsize=(26, 6))

        for i in range(n):

            ax = plt.subplot(3, n, i + 1)

            plt.imshow(self.x_val[i].reshape(IMG_SIZE, IMG_SIZE))

            ax.get_xaxis().set_visible(False)

            ax.get_yaxis().set_visible(False)



            ax = plt.subplot(3, n, i + 1 + n)

            plt.imshow(mask_p[i,:,:,0], cmap='Greys')

            ax.get_xaxis().set_visible(False)

            ax.get_yaxis().set_visible(False)



            ax = plt.subplot(3, n, i + 1 + n + n)

            plt.imshow(mask[i,:,:,0], cmap='Greys')

            ax.get_xaxis().set_visible(False)

            ax.get_yaxis().set_visible(False)

        plt.show()



# Generator for image augmentations

datagen = ImageDataGenerator(horizontal_flip=True,

                             vertical_flip=True,

                             rotation_range=180.,

                             width_shift_range=0.1,

                             height_shift_range=0.1,

                             zoom_range=0.1)
set_tf(SEED)

model = U_NET(IMG_SIZE)

model.compile(optimizer='adam', loss='logcosh', metrics=[dice])
def lr_scheduler(epoch, lr):

    if epoch < 4:

        lr = 10**(-4.5 + epoch/2)

        return lr

    else:

        lr*=0.95

        return lr

    



lrscheduler     = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

pltvalmask      = PlotValCallback(model, X_test[60:68], y_test[60:68])

early_stop      = keras.callbacks.EarlyStopping(patience=5, verbose=1)

modelcheckpoint = keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_PATH,'weights.h5'), 

                                                  save_best_only=True,

                                                  save_weights_only=True, 

                                                  verbose=1,

                                                  monitor='val_loss', mode='min')

batch_size=16

initial_epochs=10



model.fit(X_train, y_train, batch_size=batch_size, epochs=initial_epochs,

          validation_data=[X_test, y_test],

          callbacks=[pltvalmask, lrscheduler, modelcheckpoint])
datagen.fit(X_train, seed=SEED)

image_generator = datagen.flow(X_train, batch_size=batch_size, seed=SEED)

mask_generator  = datagen.flow(y_train, batch_size=batch_size, seed=SEED)

train_generator = zip(image_generator, mask_generator)
batch_size=16

epochs=20



model.load_weights(os.path.join(OUTPUT_PATH,'weights.h5'))

model.fit_generator(train_generator,

                    steps_per_epoch=len(X_train) / batch_size, 

                    epochs=epochs,

                    validation_data=[X_test, y_test],

                    initial_epoch=initial_epochs,

                    callbacks=[pltvalmask, lrscheduler, modelcheckpoint, early_stop])