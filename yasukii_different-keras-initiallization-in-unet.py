# Version 1

# https://github.com/zhixuhao/unet/blob/master/model.py

import numpy as np

import tensorflow as tf

from tensorflow import keras

from keras.models import *

from keras.layers import *

from keras.optimizers import *

from keras import backend as K



def unet(HEIGHT=1024,WIDTH=1024):

    inputs = Input((HEIGHT,WIDTH,3))

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)

    conv1 = BatchNormalization(name='bn1')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)

    conv2 = BatchNormalization(name='bn2')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)

    conv3 = BatchNormalization(name='bn3')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)

    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)

    conv4 = BatchNormalization(name='bn4')(conv4)

    drop4 = Dropout(0.5)(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)



    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    conv5 = BatchNormalization(name='bn5')(conv5)

    drop5 = Dropout(0.5)(conv5)



    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))

    merge6 = concatenate([drop4,up6], axis = 3)

    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)

    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    conv6 = BatchNormalization(name='bn6')(conv6)



    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))

    merge7 = concatenate([conv3,up7], axis = 3)

    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)

    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    conv7 = BatchNormalization(name='bn7')(conv7)



    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))

    merge8 = concatenate([conv2,up8], axis = 3)

    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)

    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    conv8 = BatchNormalization(name='bn8')(conv8)



    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))

    merge9 = concatenate([conv1,up9], axis = 3)

    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)

    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv9 = BatchNormalization(name='bn9')(conv9)

    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = Conv2D(1, 1, activation = 'relu')(conv9)

    conv10 = BatchNormalization(name='bn10')(conv10)



    out = Activation('sigmoid')(conv10)

    predict = Reshape((HEIGHT ,WIDTH))(out)



    model = Model(input = inputs, output = predict,name = 'unet1')



    return model



if __name__=="__main__":

    model1 = unet()

    model1.summary()
# Version 2



def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)

    b = keras.layers.BatchNormalization()(c)

    d = keras.layers.Dropout(0.25)(b)

    p = keras.layers.MaxPool2D((2, 2), (2, 2))(d)

    return c, p



def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):

    us = keras.layers.UpSampling2D((2, 2))(x)

    concat = keras.layers.Concatenate()([us, skip])

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)

    c = keras.layers.BatchNormalization()(c)

    return c



def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)

    return c

def UNet():

    f = [64, 128, 256, 512, 1024]

    inputs = keras.layers.Input((1024, 1024, 3))

    

    p0 = inputs

    c1, p1 = down_block(p0, f[0]) #1024 -> 512

    c2, p2 = down_block(p1, f[1]) #512 -> 256

    c3, p3 = down_block(p2, f[2]) #256 -> 128

    c4, p4 = down_block(p3, f[3]) #128 -> 64

    

    bn = bottleneck(p4, f[4])

    

    u1 = up_block(bn, c4, f[3]) #64 -> 128

    u2 = up_block(u1, c3, f[2]) #128 -> 256

    u3 = up_block(u2, c2, f[1]) #256 -> 512

    u4 = up_block(u3, c1, f[0]) #512 -> 1024

    

    out = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)

    outputs = keras.layers.Reshape((1024,1024))(out)

    model = keras.models.Model(inputs, outputs,name = 'unet2')

    return model



if __name__=="__main__":

    model2 = UNet()

    model2.summary()
def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)

    y_pred = K.cast(y_pred, 'float32')

    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')

    intersection = y_true_f * y_pred_f

    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))

    return score



def dice_loss(y_true, y_pred):

    smooth = 1.

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = y_true_f * y_pred_f

    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return 1. - score



def bce_dice_loss(y_true, y_pred):

    return keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)



def get_iou_vector(A, B):

    # Numpy version    

    batch_size = A.shape[0]

    metric = 0.0

    for batch in range(batch_size):

        t, p = A[batch], B[batch]

        true = np.sum(t)

        pred = np.sum(p)

        

        # deal with empty mask first

        if true == 0:

            metric += (pred == 0)

            continue

        

        # non empty mask case.  Union is never empty 

        # hence it is safe to divide by its number of pixels

        intersection = np.sum(t * p)

        union = true + pred - intersection

        iou = intersection / union

        

        # iou metrric is a stepwise approximation of the real iou over 0.5

        iou = np.floor(max(0, (iou - 0.45)*20)) / 10

        

        metric += iou

        

    # teake the average over all images in batch

    metric /= batch_size

    return metric





def my_iou_metric(label, pred):

    # Tensorflow version

    return tf.py_function(get_iou_vector, [label, pred > 0.5], tf.float64)
import os, random

from PIL import Image

from sklearn.model_selection import train_test_split

class DataGen(keras.utils.Sequence):

    def __init__(self, path, batch_size=1, image_size=1024):

        self.path = path

        self.batch_size = batch_size

        self.image_size = image_size

        files = os.listdir(self.path)

        files = [os.path.join(self.path,x) for x in files]

        self.trains, self.vals = train_test_split(files, test_size=0.1, random_state=42)

    

    def generate(self,files): 

        random.shuffle(files)

        while True:

            image_batch = np.zeros([self.batch_size,self.image_size,self.image_size,3])

            label_batch = np.zeros([self.batch_size,self.image_size,self.image_size])

            index = random.randint(0,len(files)-self.batch_size)

            for i,img in enumerate(files[index:index+self.batch_size]):

                angle = random.randint(0,180)

                width = random.randint(0,100)

                hight = random.randint(0,100)

        

                ## Reading Image

                image = Image.open(img)

                image = image.rotate(angle)

                image = image.resize((self.image_size, self.image_size))

                image = np.array(image)

        

                _mask_image = Image.open(img.replace('raw','label'))

                _mask_image = _mask_image.rotate(angle)

                _mask_image = _mask_image.convert('L')

                _mask_image = _mask_image.resize((self.image_size, self.image_size)) 

                mask = np.array(_mask_image)

            

                ## Normalizaing 

                image = image/255.0

                mask = mask/255.0



                image_batch[i][width:,hight:,:]=image[:self.image_size-width,:self.image_size-hight,:]

                label_batch[i][width:,hight:]=mask[:self.image_size-width,:self.image_size-hight]

        

            yield image_batch, label_batch

train_path = '../input/train/img'

batch_size= 1

gen = DataGen( train_path, image_size=1024, batch_size=batch_size)

train_gen = gen.generate(gen.trains)

val_gen = gen.generate(gen.vals)





train_steps = len(gen.trains)//batch_size

valid_steps = len(gen.vals)//batch_size
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau



model1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=bce_dice_loss, metrics=['mse'])

model2.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=bce_dice_loss, metrics=['mse'])



model_checkpoint1 = ModelCheckpoint('unet1.h5', monitor='val_mse',mode='min',verbose=1, save_best_only=True)

model_checkpoint2 = ModelCheckpoint('unet2.h5', monitor='val_mse',mode='min',verbose=1, save_best_only=True)



changelr = ReduceLROnPlateau(monitor = 'val_mse',

                patience=3,mode = 'min',

                verbose = 1,

                factor = 0.6,

                min_lr = 0.000001)

h1 = model1.fit_generator(train_gen,steps_per_epoch=train_steps,epochs=30,

                    callbacks=[model_checkpoint1,changelr],

                    validation_data = val_gen,validation_steps = valid_steps)



h2 = model2.fit_generator(train_gen,steps_per_epoch=train_steps,epochs=30,

                    callbacks=[model_checkpoint2,changelr],

                    validation_data = val_gen,validation_steps = valid_steps)
import matplotlib.pyplot as plt

import cv2



def predict(model,image):

    image = np.array(image,np.float)/255.0

    image = np.expand_dims(image,axis=0)

    pred = model.predict(image)[0]

    pred = (pred-np.min(pred))/(np.max(pred)-np.min(pred))

    pred[pred<0.5]=0

    pred[pred!=0]=1

    pred = cv2.merge([pred,pred,pred])

    return pred



def plot_result(model,img):

    image = Image.open(img)

    h,w = image.size

    copy = image.resize((1024,1024))

    copy = np.array(copy,np.float)

    pred = predict(model,copy)

    pred = cv2.resize(pred,(h,w))

    blend = np.array(image)*pred

    blend = np.asarray(blend,np.uint8)

    return blend

    

images = gen.vals

model1.load_weights('./unet1.h5')

model2.load_weights('./unet2.h5')

for image in images:

    result1 = plot_result(model1,image)

    result2 = plot_result(model2,image)  

    _image = np.array(Image.open(image))

    

    plt.figure(figsize=(48,16))

    plt.subplot(131)

    plt.title('raw_image')

    plt.axis('off') 

    plt.imshow(_image)



    plt.subplot(132)

    plt.title('model1_result')

    plt.axis('off') 

    plt.imshow(result1)



    plt.subplot(133)

    plt.title('model2_result')

    plt.axis('off') 

    plt.imshow(result2)