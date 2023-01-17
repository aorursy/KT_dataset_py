import tensorflow as tf

import tensorflow_addons as tfa

from tensorflow.keras.preprocessing.image import load_img,img_to_array

from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.optimizers import *

from tensorflow.keras.models import Model,Sequential

from tensorflow.keras.layers import *

from tensorflow.keras.callbacks import *

from tensorflow.keras import backend as K

from tensorflow.keras.losses import *

from tensorflow.keras.mixed_precision import experimental as mixed_precision

from tensorflow.keras.regularizers import l1_l2

from tensorflow.python.client import device_lib

import numpy as np



from matplotlib import cm as CM

import matplotlib.pyplot as plt

import tensorflow as tf

from tqdm import tqdm

import scipy.io as io

from PIL import Image

import PIL

import h5py

import os

import glob

import cv2

import random

import math

import sys

import itertools 



print(tf.__version__)



policy = mixed_precision.Policy('mixed_float16')

mixed_precision.set_policy(policy)



print('Compute dtype: %s' % policy.compute_dtype)

print('Variable dtype: %s\n\n' % policy.variable_dtype)



for info in device_lib.list_local_devices():

    if (info.name.find('GPU') != -1):

        print(info)
def getPath(data_type):

    root = r'../input/shanghaitech-with-people-density-map/ShanghaiTech'

    print(root)

    if data_type.find('part_A')!=-1:

        target_train = os.path.join(root,r'part_A/train_data','images')

        target_test = os.path.join(root,r'part_A/test_data','images')



    elif data_type.find('part_B')!=-1:

        target_train = os.path.join(root,r'part_B/train_data','images')

        target_test = os.path.join(root,r'part_B/test_data','images')



    train_path = []

    for img_path in glob.glob(os.path.join(target_train, '*.jpg')):

        train_path.append(str(img_path))

        

    test_path = []

    for img_path in glob.glob(os.path.join(target_test, '*.jpg')):

        test_path.append(str(img_path))

        

    return train_path, test_path



def load_density(file_path):

    gt_file = h5py.File(file_path, 'r')

    groundtruth = np.asarray(gt_file['density'])

    groundtruth = np.expand_dims(groundtruth, axis=-1)

    return groundtruth



# Get input image path

train_paths, test_paths = getPath('part_B')

# Get ground truth path

gth_train = [path.replace('jpg', 'h5').replace('images', 'ground-truth-h5') for path in train_paths]

gth_test = [path.replace('jpg', 'h5').replace('images', 'ground-truth-h5') for path in test_paths]



print('train len:', len(train_paths))

print('test len:', len(test_paths))



print(train_paths[0])

print(gth_train[0])



train_labels=np.array([load_density(path) for path in gth_train]).astype('float16')

print(train_labels.shape)

test_labels=np.array([load_density(path) for path in gth_test]).astype('float16')

print(test_labels.shape)
target_type=tf.dtypes.float16

def load_img(path):

    image_string=tf.io.read_file(path)

    image=tf.image.decode_jpeg(image_string,channels=3)

    image=tf.image.convert_image_dtype(image, target_type)

    return image



def gen_translate_func(translate_range):

    def translate_function(img, gth):

        ratio=tf.random.uniform((2,), minval=translate_range[1], maxval=translate_range[0], dtype=tf.dtypes.int32)

        ratio=tf.cast(ratio, target_type)



        out_gth=tfa.image.translate(gth, ratio, 'BILINEAR')

        out_img=tfa.image.translate(img, ratio, 'BILINEAR')



        return out_img, out_gth

    return translate_function



def gen_downsampling_func(downsampling, method='nearest', batch=True):

    batchadd=1

    if batch==False:

        batchadd=0

    @tf.function

    def _downsampling_function_(img, gth):

        down_ratio=downsampling

        before_resize=tf.reduce_sum(gth)

        gth_shape=tf.shape(gth)

        out_gth=tf.image.resize(gth, (gth_shape[0+batchadd]//down_ratio, gth_shape[1+batchadd]//down_ratio), method=method,antialias=False)

        out_gth=tf.cast(out_gth, dtype=target_type)

        after_resize=tf.reduce_sum(out_gth)

        if (before_resize >= 0.3) and (after_resize > 0.01):

            out_gth=out_gth * before_resize / after_resize

        return img, out_gth

    return _downsampling_function_



def gen_randomcrop_func(crop_size, batch=True):

    batchadd=1

    if batch==False:

        batchadd=0

    @tf.function

    def _random_crop_(img, gth):

        

        output_shape=tf.constant(crop_size)

        img_shape=tf.shape(img)



        ratio_x=tf.random.uniform((1,), minval=0, maxval=img_shape[1+batchadd]-output_shape[1], dtype=tf.dtypes.int32)[0]

        ratio_y=tf.random.uniform((1,), minval=0, maxval=img_shape[0+batchadd]-output_shape[0], dtype=tf.dtypes.int32)[0]



        out_img=tf.image.crop_to_bounding_box(img, ratio_y, ratio_x, output_shape[0], output_shape[1])

        out_gth=tf.image.crop_to_bounding_box(gth, ratio_y, ratio_x, output_shape[0], output_shape[1])



        return out_img, out_gth

    return _random_crop_



def gen_coarsedrop_func(hole_count, hole_size, hole_prob=0.75):

    def _dropout_(image, gth):

        shape = tf.shape(image)

        BATCH = shape[0]

        IMG_WIDTH = shape[2]

        IMG_HEIGHT = shape[1]

        IMG_WIDTH_F = tf.cast(IMG_WIDTH, tf.float32)

        IMG_HEIGHT_F = tf.cast(IMG_HEIGHT, tf.float32)

        PROBABILITY = hole_prob

        CT = hole_count

        CROP_SIZE = tf.cast(hole_size,tf.int32)

        # DO DROPOUT WITH PROBABILITY DEFINED ABOVE

        P = tf.cast( tf.random.uniform([], 0.0, 1.0) < PROBABILITY, tf.int32)

        if (P==0): return image, gth

        

        mask = tf.ones((IMG_HEIGHT,IMG_WIDTH,1))

        for k in range(CT):

            # CHOOSE RANDOM LOCATION

            x = tf.cast( tf.random.uniform([],0.0,IMG_WIDTH_F),tf.int32)

            y = tf.cast( tf.random.uniform([],0.0,IMG_HEIGHT_F),tf.int32)

            # COMPUTE SQUARE

            ya = tf.math.maximum(0,y-CROP_SIZE//2)

            yb = tf.math.minimum(IMG_HEIGHT,y+CROP_SIZE//2)

            xa = tf.math.maximum(0,x-CROP_SIZE//2)

            xb = tf.math.minimum(IMG_WIDTH,x+CROP_SIZE//2)

            # DROPOUT IMAGE

            one = mask[ya:yb,0:xa,:]

            two = tf.zeros([yb-ya,xb-xa,1]) 

            three = mask[ya:yb,xb:IMG_WIDTH,:]

            middle = tf.concat([one,two,three],axis=1)

            mask = tf.concat([mask[0:ya,:,:],middle,mask[yb:IMG_HEIGHT,:,:]],axis=0)

        

        # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR 

        mask = tf.cast(mask, dtype='float16')

        return image*mask, gth*mask

    return _dropout_



def basic_dataset(img_data, label_data, batch_size=1, flip=False, downsampling=1, buffer_size=32, shuffle=False):

    input_data=tf.data.Dataset.from_tensor_slices(img_data)

    input_data = input_data.map(load_img)

    output_data=tf.data.Dataset.from_tensor_slices(label_data)



    if flip:

        input_data=input_data.concatenate(input_data.map(tf.image.flip_left_right))

        output_data=output_data.concatenate(output_data.map(tf.image.flip_left_right))



    dataset=tf.data.Dataset.zip((input_data, output_data))

    if shuffle:

        dataset = dataset.batch(batch_size).repeat().shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    else:

        dataset = dataset.batch(batch_size).repeat()



    if downsampling!=1:

        dataset=dataset.map(gen_downsampling_func(downsampling=downsampling))

    return dataset



def crop_dataset(img_data, label_data, crop_size, flip, downsampling, batch_size=1, sample_method='bicubic', buffer_size=16, 

hole_count=0, hole_size=100, hole_prob=0.75):

    dataset = basic_dataset(img_data, label_data, flip=flip, batch_size=batch_size, downsampling=1, shuffle=True, buffer_size=buffer_size)

    dataset=dataset.map(gen_randomcrop_func(crop_size))

    if hole_count!=0:

        dataset=dataset.map(gen_coarsedrop_func(hole_count=hole_count, hole_size=hole_size, hole_prob=hole_prob))

    if downsampling!=1:

        dataset=dataset.map(gen_downsampling_func(downsampling=downsampling, method=sample_method))

    return dataset



def translate_dataset(img_data, label_data, translate_range, flip, downsampling, batch_size=1, sample_method='bicubic', buffer_size=16, 

hole_count=0, hole_size=100, hole_prob=0.75):

    dataset = basic_dataset(img_data, label_data, flip=flip, batch_size=batch_size, downsampling=1, shuffle=True, buffer_size=buffer_size)

    dataset=dataset.map(gen_translate_func(translate_range))

    if hole_count!=0:

        dataset=dataset.map(gen_coarsedrop_func(hole_count=hole_count, hole_size=hole_size, hole_prob=hole_prob))

    if downsampling!=1:

        dataset=dataset.map(gen_downsampling_func(downsampling=downsampling, method=sample_method))

    return dataset

            

def show_images(images, cols = 2, titles = None, padding=1, axis="off", channel1=CM.jet):

    assert((titles is None)or (len(images) == len(titles)))

    n_images = len(images)

    # if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]

    if titles is None: titles = [None for i in range(1,n_images + 1)]

    fig = plt.figure()

    

    for n, (image, title) in enumerate(zip(images, titles)):

        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)



        plt.axis(axis)

        plt.subplots_adjust(wspace=padding, hspace=padding)



        if (image.shape[2] == 1):

            image = image[:,:,0]

            plt.imshow(image, cmap=channel1)

        elif np.any(image > 1.0):

            plt.imshow(image / 255.0)

        else:

            plt.imshow(image)

        a.set_title(title, fontsize=20)

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)

    plt.show()

def density_mae(y_truth, y_pred):

    return tf.reduce_mean(tf.abs(tf.reduce_sum(y_truth, axis=(1,2,3))-tf.reduce_sum(y_pred, axis=(1,2,3))))
def CSRNet(batch_norm=False, middle_activation='relu', final_activation='sigmoid', reg=(0.0,0.0), dtype=tf.dtypes.float32):

    init=RandomNormal(stddev=0.01)

    

    activationDict = {

        'sigmoid': tf.math.sigmoid,

        'relu': tf.nn.relu,

        'tanh': tf.math.tanh,

        'leakyrelu': tf.nn.leaky_relu,

        'elu': tf.nn.elu,

        'softsign':tf.nn.softsign

    }

    vgg16 = VGG16(weights='imagenet', include_top=False)

    vgg16.trainable=False

    

    input_layer = Input(shape=(None, None, 3))

    x = input_layer

    x = Lambda(lambda batch: (batch - tf.constant([0.485,0.456,0.406], dtype=dtype)) / tf.constant([0.229,0.224,0.225], dtype=dtype))(x)

    count = 0

    for layer in vgg16.layers:

        layer.trainable = False

        x = layer(x)

        if 'conv' in layer.name:

            count+=1

        if count == 10:

            break;

            

    

    if batch_norm:

        x = BatchNormalization()(x)

    

    if batch_norm:

        x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=2, padding='same', use_bias=False, kernel_initializer=init, kernel_regularizer=l1_l2(l1=reg[0], l2=reg[1]))(x)

        x = BatchNormalization()(x)

    else:

        x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=2, padding='same', use_bias=True, kernel_initializer=init, kernel_regularizer=l1_l2(l1=reg[0], l2=reg[1]))(x)

    x = Activation(activationDict[middle_activation])(x)

    

    if batch_norm:

        x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=2, padding='same', use_bias=False, kernel_initializer=init, kernel_regularizer=l1_l2(l1=reg[0], l2=reg[1]))(x)

        x = BatchNormalization()(x)

    else:

        x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=2, padding='same', use_bias=True, kernel_initializer=init, kernel_regularizer=l1_l2(l1=reg[0], l2=reg[1]))(x)

    x = Activation(activationDict[middle_activation])(x)

    

    if batch_norm:

        x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=2, padding='same', use_bias=False, kernel_initializer=init, kernel_regularizer=l1_l2(l1=reg[0], l2=reg[1]))(x)

        x = BatchNormalization()(x)

    else:

        x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=2, padding='same', use_bias=True, kernel_initializer=init, kernel_regularizer=l1_l2(l1=reg[0], l2=reg[1]))(x)

    x = Activation(activationDict[middle_activation])(x)

    

    

    if batch_norm:

        x = Conv2D(filters=256, kernel_size=(3, 3), dilation_rate=2, padding='same', use_bias=False, kernel_initializer=init, kernel_regularizer=l1_l2(l1=reg[0], l2=reg[1]))(x)

        x = BatchNormalization()(x)

    else:

        x = Conv2D(filters=256, kernel_size=(3, 3), dilation_rate=2, padding='same', use_bias=True, kernel_initializer=init, kernel_regularizer=l1_l2(l1=reg[0], l2=reg[1]))(x)

    x = Activation(activationDict[middle_activation])(x)

    

    

    if batch_norm:

        x = Conv2D(filters=128, kernel_size=(3, 3), dilation_rate=2, padding='same', use_bias=False, kernel_initializer=init, kernel_regularizer=l1_l2(l1=reg[0], l2=reg[1]))(x)

        x = BatchNormalization()(x)

    else:

        x = Conv2D(filters=128, kernel_size=(3, 3), dilation_rate=2, padding='same', use_bias=True, kernel_initializer=init, kernel_regularizer=l1_l2(l1=reg[0], l2=reg[1]))(x)

    x = Activation(activationDict[middle_activation])(x)

    

    

    if batch_norm:

        x = Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=2, padding='same', use_bias=False, kernel_initializer=init, kernel_regularizer=l1_l2(l1=reg[0], l2=reg[1]))(x)

        x = BatchNormalization()(x)

    else:

        x = Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=2, padding='same', use_bias=True, kernel_initializer=init, kernel_regularizer=l1_l2(l1=reg[0], l2=reg[1]))(x)

    x = Activation(activationDict[middle_activation])(x)

        

    ###############################    

    x = Conv2D(filters=1, kernel_size=(1, 1), dilation_rate=1, padding='same', use_bias=True, kernel_initializer=init)(x)

    x = Activation(activationDict[final_activation])(x)

    

    model = Model(input_layer, x)

    return model
class SPANet(tf.keras.Model):

    def __init__(self, model, branches=4, sigma=0.0, with_vanilla=False, dtype=tf.dtypes.float32):

        super(SPANet, self).__init__()

        self.model = model

        self.branches = branches

        self.sigma = sigma

        self.with_vanilla = with_vanilla

        self.parameter_type = dtype

    @tf.function

    def train_step(self, data):

        # function with custom configuration train loss, with out vanilla loss

        x, y = data

        

        output_shape = tf.shape(y)

        out_height = output_shape[1] // self.branches

        out_width = output_shape[2] // self.branches

        

        rank_loss = tf.constant(0.0, dtype=self.parameter_type)

        SK = tf.zeros(tf.shape(y), dtype=self.parameter_type)

        SK_previous = tf.zeros(tf.shape(y), dtype=self.parameter_type)

        

        for b in range(1, self.branches):

            with tf.GradientTape() as tape:

                y_pred = self.model(x, training=True) * (1.0 - SK_previous)  # Forward pass

                

                M1_pred_sum = tf.reduce_sum(y_pred[:, -out_height*b:,     -out_height*b:], axis=(1,2,3))

                M2_pred_sum = tf.reduce_sum(y_pred[:, -out_height*(b+1):, -out_height*(b+1):], axis=(1,2,3))

                

                M1_y_sum = tf.reduce_sum(y[:, -out_height*b:,     -out_width*b:], axis=(1,2,3))

                M2_y_sum = tf.reduce_sum(y[:, -out_height*(b+1):, -out_width*(b+1):], axis=(1,2,3))

                

                Rank_ratio = tf.reduce_mean(tf.abs((M1_pred_sum-M2_pred_sum)-(M1_y_sum-M2_y_sum)))

                # Rank_ratio = tf.reduce_mean(tf.math.maximum(M1_pred-M2_pred+(M2_y-M1_y), 0.0))

                # Rank_loss = tf.reduce_mean(M1_pred-M1_y - (M2_pred-M2_y))

            # Compute gradients

            trainable_vars = self.model.trainable_variables

            rank_gradients = tape.gradient(Rank_ratio, trainable_vars)

            if (Rank_ratio > 0.0):

                # Update weights

                self.optimizer.apply_gradients(zip(rank_gradients, trainable_vars))



              

            # Get New Density map

            SK_previous = tf.math.sigmoid(

                tf.nn.relu(tf.abs(model(x) - y_pred) - tf.constant(0.001, dtype=self.parameter_type))

            ) - 0.5

            SK = tf.math.maximum(SK, SK_previous)

            rank_loss+=Rank_ratio





        with tf.GradientTape() as tape2:

            y_final = self.model(x, training=True)

            MEP_loss = tf.abs(y_final-y) * SK

            MEP_loss = tf.reduce_mean(tf.reduce_mean(MEP_loss, axis=(1,2,3)))

            

        trainable_vars = self.model.trainable_variables

        MEP_gradients = tape2.gradient(MEP_loss, trainable_vars)

        self.optimizer.apply_gradients(zip(MEP_gradients, trainable_vars))

        

        if self.with_vanilla:

            with tf.GradientTape() as tape3:

                y_final = self.model(x, training=True)

                Vanilla_loss = self.compiled_loss(y, y_final)

              

            trainable_vars = self.model.trainable_variables    

            Vanilla_gradients = tape3.gradient(Vanilla_loss, trainable_vars)

            self.optimizer.apply_gradients(zip(Vanilla_gradients, trainable_vars))

        

        # Update metrics (includes the metric that tracks the loss)

        # self.model.compiled_metrics.update_state(y, y_final)

        self.compiled_metrics.update_state(y, y_final)

        # Return a dict mapping metric names to current value

        result = {'rank_loss': rank_loss, 'MEP': MEP_loss}

        result.update({m.name: m.result() for m in self.metrics})

        return result

    

    @tf.function

    def __call__(self, x, **kwargs):

        return self.model(x)



sgd = SGD(lr = 1e-6, momentum = 0.2, nesterov=True)

rms = RMSprop(lr=1e-3, momentum=0.7)

nadam = Nadam(lr=1e-2)



optimizer = rms

loss = 'binary_crossentropy'

batch_size=8

train_size=320

pre_train_epoch=20



val_gen = basic_dataset(train_paths[train_size:], train_labels[train_size:], batch_size=1, flip=False, downsampling=8, buffer_size=16, shuffle=False)

test_gen = basic_dataset(test_paths, test_labels, batch_size=1, flip=False, downsampling=8, buffer_size=16, shuffle=False)

train_gen = crop_dataset(train_paths[:train_size], train_labels[:train_size], crop_size=(512,512), flip=True, downsampling=8, batch_size=batch_size, sample_method='bicubic', buffer_size=8)



sub_model = CSRNet(middle_activation='relu', final_activation='sigmoid', batch_norm=True, dtype=tf.dtypes.float16)

sub_model.compile(optimizer=RMSprop(lr=1e-3, momentum=0.7), loss=loss, metrics=[density_mae])



filepath = r'/content/CSRNet/CSRNet'

reduceLR=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_density_mae', factor=0.5, patience=10, verbose=1, min_delta=1e-8)

earlyStop=tf.keras.callbacks.EarlyStopping(monitor='val_density_mae', patience=40, verbose=1, restore_best_weights=True)

monitor=tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_density_mae', verbose=0, save_best_only=True, save_weights_only=True)



print('train baseline: ', np.mean( np.sum(train_labels[:train_size], axis=(1,2,3))))

print('val baseline: ', np.mean( np.sum(train_labels[train_size:], axis=(1,2,3))))



sub_model.fit(train_gen, steps_per_epoch=int(train_size*2/batch_size), epochs=pre_train_epoch, verbose=1, 

              max_queue_size=32, workers=6, validation_data=val_gen, validation_steps=80, callbacks=[earlyStop, monitor])



sub_model.load_weights(filepath)

sub_model.compile(optimizer=RMSprop(lr=1e-3, momentum=0.7), loss=loss, metrics=[density_mae])

sub_model.evaluate(val_gen, steps=80, verbose=1, workers=4)

sub_model.evaluate(test_gen, steps=316, verbose=1, workers=4)
batch_size = 1

training_epoch=1

spa_filepath = r'/content/CSRNet/SPANet'

spa_monitor=tf.keras.callbacks.ModelCheckpoint(spa_filepath, monitor='val_density_mae', verbose=0, save_best_only=True, save_weights_only=True)

train_gen = crop_dataset(train_paths[:train_size], train_labels[:train_size], crop_size=(512,512), flip=True, downsampling=8, batch_size=batch_size, sample_method='bicubic', buffer_size=8)





model = SPANet(sub_model, branches=32, with_vanilla=False, dtype=tf.dtypes.float16)

model.compile(optimizer=RMSprop(lr=1e-4, momentum=0.5), loss=loss, metrics=[density_mae])

# int(train_size/batch_size)

history=model.fit(train_gen, steps_per_epoch=int(train_size/batch_size), epochs=training_epoch, verbose=1, max_queue_size=32, workers=6,

                  validation_data=val_gen, validation_steps=80, callbacks=[earlyStop, reduceLR, spa_monitor])



model.load_weights(spa_filepath)

model.evaluate(val_gen, steps=80, verbose=1, workers=4)

model.evaluate(test_gen, steps=316, verbose=1, workers=4)
for a,b in val_gen.take(4):

    c = model.predict(a)

    show_images([a[0], b[0], c[0]],1, ['',np.sum(b[0]),np.sum(c[0])])
for a,b in test_gen.take(4):

    c = model.predict(a)

    show_images([a[0], b[0], c[0]],1, ['',np.sum(b[0]),np.sum(c[0])])