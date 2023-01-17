# packages

## system

import os, warnings, sys, pathlib

from pathlib import Path

warnings.filterwarnings('ignore')



## data structure

import pandas as pd

import numpy as np

import tensorflow as tf

from tensorflow import keras

import tensorflow.keras.backend as K

from tensorflow.python.keras.utils.data_utils import Sequence



## utils

import math

import scipy

import scipy.ndimage as ndimage

import scipy.ndimage.filters as filters

import random

import datetime

import time

from tqdm import tqdm

from sklearn.pipeline import Pipeline

from sklearn.utils import class_weight



## image 

import cv2

from PIL import Image

from skimage.transform import rotate

import imgaug.augmenters as iaa



## audio

import soundfile as sf

import librosa

from pydub import AudioSegment



## data cleaning

from sklearn.preprocessing import StandardScaler, normalize, LabelEncoder, OneHotEncoder, scale

from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay



## graphing

import librosa.display

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.collections import PatchCollection

from matplotlib.patches import Rectangle

%matplotlib inline
# print(tf.test.is_gpu_available(), tf.config.list_physical_devices('GPU'))
# get image path

# define audio dataset path



DATASET_AUDIO = "train_audio/"

DATASET_SPEC = "/kaggle/input/" + "birdsong-recognition/" + DATASET_AUDIO



print(DATASET_SPEC)

# get index of label and freq 

label_names = sorted(set(os.listdir(DATASET_SPEC)))



# onehot encode label and freq

label_to_onehot = OneHotEncoder(sparse = False)

label_to_onehot.fit(np.array(label_names).reshape(-1, 1))

label_dict = dict(zip(label_names, label_to_onehot.fit_transform(np.array(label_names).reshape(-1, 1))))

pd.DataFrame(label_dict)
freq_bin = np.arange(0,5000)

def split(word): 

    return [char for char in word] 

freq_bin_encode = [split(str(freq).zfill(4)) for freq in freq_bin]

freq_bin_encode = [[int(digit) for digit in freq]  for freq in freq_bin_encode]

freq_dict = dict(zip(list(freq_bin), freq_bin_encode))

pd.DataFrame(freq_dict)
# define audio dataset path



# DATASET_AUDIO = "train_audio_small_sort"

# DATASET_AUDIO = "train_audio_medium_sort"

# DATASET_AUDIO = "train_audio_large_sort"

DATASET_AUDIO = "train_audio_sort"



# DATASET_SPEC = "../input" + "/birdsong-recognition" + "/img_" + DATASET_AUDIO # local

DATASET_SPEC = "../input" + "/img-train-audio-sort" + "/img_" + DATASET_AUDIO # kaggle  



print(DATASET_SPEC)

data_dir = DATASET_SPEC

data_root = pathlib.Path(data_dir)

all_image_path = data_root.rglob('*.jpg')

all_image_path = [str(pathlib.Path(path)) for path in all_image_path]



all_label = [path.split('/')[-2] for path in all_image_path]
class DataGenerator(Sequence):

    """

    https://bbs.cvmart.net/topics/1545

    """

    def __init__(self, filepath, label_dict, freq_dict, batch_size=8, imgshape=(256, 472),

                 n_channels=3, n_classes=13, shuffle=True, ):

        # initiation method

        self.filepath=filepath

        self.batch_size = batch_size

        self.imgshape = imgshape

        self.n_channels = n_channels

        self.shuffle = shuffle

        

        self.pathlist= [str(pathlib.Path(path)) for path in pathlib.Path(self.filepath).rglob('*.jpg')]

        self.on_epoch_end()

        self.label_dict = label_dict

        self.freq_dict = freq_dict



    def __getitem__(self, index):

        # generate batch index

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # generate list of batch index

        batch_pathlist = [self.pathlist[k] for k in indexes]

        # generate data

        images = self._generate_images(batch_pathlist)

        freqs = self._generate_freqs(batch_pathlist)

        labels = self._generate_labels(batch_pathlist)

        return (images, freqs), labels



    def __len__(self):

        # return the number of batch

        return int(np.floor(len(self.pathlist) / self.batch_size))

    

    def _load_image(self, image_path):

        def gasuss_noise(image, mean=0, var=0.01):

            noise = np.random.normal(mean, var ** 0.5, image.shape)

            out = image + noise

            if out.min() < 0:

                low_clip = -1.

            else:

                low_clip = 0.

            out = np.clip(out, low_clip, 1.0)

            return out

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)/255 # grey 1 channel

        img = cv2.flip(img, flipCode= random.randint(-1,1)) # flip up or down

#         img = rotate(img, angle=random.randint(-5,5), mode='reflect') # rotate

        img = cv2.warpAffine(img,

                             M = np.float32([[1, 0, random.randint(-28,28)],

                                             [0, 1, random.randint(-28,28)]]),

                             dsize = img.shape)

        img = gasuss_noise(img, var = random.randint(1,10)/1000)

        if self.imgshape != img.shape:

            img = cv2.resize(img, self.imgshape)

            

        return np.expand_dims(img, -1)

    

    def _generate_images(self, batch_pathlist):

        # generate images for a batach

        images = np.empty((self.batch_size, *self.imgshape, self.n_channels))

        for i, path in enumerate(batch_pathlist):

            images[i,] = self._load_image(path)

        return images



    def _generate_labels(self, batch_pathlist):

        # generate labels for a batch

        labels = np.empty((self.batch_size, len(self.label_dict) ), dtype=int)

        for i, path in enumerate(batch_pathlist):

            # Store sample

            labels[i,] = label_dict.get(path.split('/')[-2])

        return labels

    

    def _generate_freqs(self, batch_pathlist):

        # generatre freqs for a batch

        freqs = np.empty((self.batch_size, len(self.freq_dict.get(0))), dtype=int)

        # Generate data

        for i, path in enumerate(batch_pathlist):

            # Store sample

            freqs[i,]= freq_dict.get(int(path.split('/')[-1].split('_')[1]))

        return freqs



    def on_epoch_end(self):

        # update index at the end of each epoch

        self.indexes = np.arange(len(self.pathlist))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)

# Parameters

batch_size = 32

imgshape = (128,128)

n_channels = 1

params = {'batch_size': batch_size,

        'n_channels': n_channels,

        'shuffle': True,

        'label_dict': label_dict, 

        'freq_dict': freq_dict,

        'imgshape': imgshape}

train_filepath = DATASET_SPEC + "/train" # for local

valid_filepath = DATASET_SPEC + "/val" # for local

all_filepath = DATASET_SPEC

# Generators

train_generator = DataGenerator(train_filepath, **params)

valid_generator = DataGenerator(valid_filepath, **params)

all_generator = DataGenerator(all_filepath, **params)
# getting weight for the unbalance data

cw = class_weight.compute_class_weight('balanced',

                                                 np.unique(all_label),

                                                 all_label)

cw = dict(enumerate(cw))

def getkeybyval(my_dict, val):

    key_list = list(my_dict.keys()) 

    val_list = list(my_dict.values())

    return key_list[val_list.index(list(val))]



def show_batch(image_batch, freq_batch, label_batch, label_to_onehot, freq_dict):

    """

    https://stackoverflow.com/questions/60129658/def-show-batch-not-showing-my-train-images

    """

    plt.figure(figsize=(20,20))

    for n in range(25):

        ax = plt.subplot(5,5,n+1)

        plt.imshow(image_batch[n,:,:,0])

        plt.title("class: {} ({})".format(

            label_to_onehot.inverse_transform([label_batch[n]]), 

            getkeybyval(freq_dict, freq_batch[n])))

        plt.axis('off')
i = 0

for image_freq, label in train_generator:    

    print("[batch {}/{}] shape: [image={}, freq = {}], label = {}".

          format(i+1,len(train_generator), image_freq[0].shape, image_freq[1].shape, label.shape))

    show_batch(image_freq[0], image_freq[1], label, label_to_onehot, freq_dict)

    

    i += 1

    if i > 0:

        break 

        
initializer = tf.keras.initializers.VarianceScaling()

images_shape = [128,128,1]

freqs_shape = len(freq_dict.get(0))

output_size = len(label_dict)



print("initializer: ", initializer)

print("images_shape: ", images_shape)

print("freqs_shape: ", freqs_shape)

print("output_size: ", output_size)
# batchnormalization and activation

def BatchActivate(x):

    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Activation('relu')(x)

    return x



def convolution2D_block(x, filters, kernel_size = (3,3), strides=(1,1),

                        padding='same', BatchAct=True, initializer = tf.keras.initializers.VarianceScaling()):

    x = keras.layers.Conv2D(filters = filters, strides=strides, padding=padding,

                            kernel_size = kernel_size, kernel_initializer = initializer)(x)

    if BatchAct == True:

        x = BatchActivate(x)

    return x

# conv block

def convolution1D_block(x, filters, size, strides=(1,1), padding='same', BatchAct=True, initializer = tf.keras.initializers.VarianceScaling()):

    x = keras.layers.Conv1D(filters, size, strides=strides, padding=padding, kernel_initializer = initializer)(x)

    if BatchAct == True:

        x = BatchActivate(x)

    return x



def residual1D_block(x, filters, conv_num=3, activation="relu", padding="same"):

    # Shortcut

    s = keras.layers.Conv1D(filters, kernel_size = 1, padding=padding)(x)

    for i in range(conv_num - 1):

        x = keras.layers.Conv1D(filters, kernel_size= 3, padding=padding)(x)

        x = keras.layers.Activation(activation)(x)

    x = keras.layers.Conv1D(filters, 3, padding=padding)(x)

    x = keras.layers.Add()([x, s])

    x = keras.layers.Activation(activation)(x)

    return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)





def squeeze_excite_block(x, ratio=8):

    '''

    https://github.com/titu1994/keras-squeeze-excite-network

    '''

    init = x

    filters = init.shape[3]

    se_shape = (1, 1, filters)



    se = keras.layers.GlobalAveragePooling2D()(x)

    se = keras.layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)

    se = keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    se = keras.layers.Reshape(se_shape)(se)

    x = keras.layers.multiply([x, se])

    return x



def squeeze_excite_block1d(x, ratio=8):

    '''

    https://github.com/titu1994/keras-squeeze-excite-network

    '''

    init = x

    filters = init.shape[1]

    se = keras.layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(x)

    se = keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = keras.layers.multiply([x, se])

    return x





def residual2D_block(x, filters = 128, conv_num=2, downscale_filters = 32, kernel_size = (3,1), batch_activate = True, initializer = tf.keras.initializers.VarianceScaling()):

    # dimension reduction

    if filters != x.shape[3]:

        x = keras.layers.Conv2D(filters, kernel_size = (1,1), padding="same", kernel_initializer = initializer)(x) 

        c = convolution2D_block(x, downscale_filters, kernel_size = kernel_size, initializer = initializer)

    else:

        c = convolution2D_block(x, downscale_filters, kernel_size = kernel_size, initializer = initializer)

    # convolution2D block series

    for i in range(conv_num - 2): 

        c = convolution2D_block(c, downscale_filters, kernel_size = kernel_size, initializer = initializer)

    # restore dimension

    c = keras.layers.Conv2D(filters, (1,1), padding="same", kernel_initializer = initializer)(c) #

    c = squeeze_excite_block(c)

    x = keras.layers.Add()([c, x])

    return x





def resNet_model(image_inputs, initializer = initializer, DropoutRatio = 0.5):

    

    # expanding receptive field

    x = convolution2D_block(image_inputs, filters = 64, kernel_size = (3,3))

    x = convolution2D_block(x, filters = 256, kernel_size = (3,3))

    

    # residual_block_sequence with short

    x = keras.layers.MaxPooling2D((2, 2))(x)

    s1 = x

    x = residual2D_block(x, filters = 256, downscale_filters = 128, conv_num=3, kernel_size = (3,1), batch_activate = True, initializer = initializer)    

#     x = keras.layers.MaxPooling2D((2, 2))(x)

    s1 = keras.layers.Conv2D(256, (1,1), padding="same", kernel_initializer = initializer)(s1)

#     s1 = keras.layers.MaxPooling2D((2, 2))(s1)

    x = residual2D_block(x, filters = 256, downscale_filters = 128, conv_num=3, kernel_size = (3,1), batch_activate = True, initializer = initializer)

    x = keras.layers.Add()([s1, x])

    

    x = keras.layers.MaxPooling2D((2, 2))(x)

    s2 = x

    x = residual2D_block(x, filters = 256, downscale_filters = 128, conv_num=4, kernel_size = (3,3), batch_activate = True, initializer = initializer)

#     x = keras.layers.MaxPooling2D((2, 2))(x)

    s2 = keras.layers.Conv2D(256, (1,1), padding="same", kernel_initializer = initializer)(s2)

#     s2 = keras.layers.MaxPooling2D((2, 2))(s2)

    x = residual2D_block(x, filters = 256, downscale_filters = 128, conv_num=4, kernel_size = (3,3), batch_activate = True, initializer = initializer)

    x = keras.layers.Add()([s2, x])

    

    x = keras.layers.MaxPooling2D((2, 2))(x)

    s3 = x

    x = residual2D_block(x, filters = 256, downscale_filters = 128, conv_num=3, kernel_size = (3,1), batch_activate = True, initializer = initializer)

#     x = keras.layers.MaxPooling2D((2, 2))(x)

    s3 = keras.layers.Conv2D(256, (1,1), padding="same", kernel_initializer = initializer)(s3)

#     s3 = keras.layers.MaxPooling2D((2, 2))(s3)

    x = residual2D_block(x, filters = 256, downscale_filters = 128, conv_num=3, kernel_size = (3,1), batch_activate = True, initializer = initializer)

    x = keras.layers.Add()([s3, x])



    x = keras.layers.MaxPooling2D((2, 2))(x)

    s4 = x

    x = residual2D_block(x, filters = 256, downscale_filters = 128, conv_num=3, kernel_size = (3,1), batch_activate = True, initializer = initializer)

#     x = keras.layers.MaxPooling2D((2, 2))(x)

    s4 = keras.layers.Conv2D(256, (1,1), padding="same", kernel_initializer = initializer)(s4)

#     s4 = keras.layers.MaxPooling2D((2, 2))(s4)

    x = residual2D_block(x, filters = 256, downscale_filters = 128, conv_num=3, kernel_size = (3,1), batch_activate = True, initializer = initializer)

    x = keras.layers.Add()([s4, x])



    # output

    x = keras.layers.AveragePooling2D(pool_size=(2, 2))(x)

    x = keras.layers.Flatten()(x)

    x = keras.layers.Dropout(DropoutRatio)(x)

    model=tf.keras.models.Model(inputs=image_inputs,outputs=x)

    return model



def build_model(input_shape, num_classes, DropoutRatio, initializer = tf.keras.initializers.VarianceScaling()):

    # image decoding

    image_inputs = keras.layers.Input(shape = input_shape[0], name = "image_input_layer")

    resNet = resNet_model(image_inputs = image_inputs, initializer = initializer, DropoutRatio = DropoutRatio)

    x = resNet.output

    x = keras.layers.Dense(1024, activation='relu')(x)

    x = squeeze_excite_block1d(x)

#     x = keras.layers.Dropout(DropoutRatio)(x)

    # merge freq

    freq_inputs = keras.layers.Input(shape = input_shape[1], name = "freq_input_layer")

    x = keras.layers.Concatenate(axis= 1)([x, freq_inputs])

    # multilayer perceptron

    x = keras.layers.Dropout(DropoutRatio)(x)

    # output

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax",

                                    kernel_initializer = initializer, name="output")(x) # or sigmoid

    return tf.keras.Model(inputs=[image_inputs, freq_inputs], outputs=outputs)





model = build_model(input_shape = (images_shape, freqs_shape), num_classes = output_size, initializer = initializer, DropoutRatio = 0)

model.summary()
# Adaptive Moment Estimation (Adam) 

adam_learning_rate = 0.0001  # initial learning rate

adam = keras.optimizers.Adam(learning_rate=adam_learning_rate)



# RMSprop

rmsprop_learning_rate = 0.01

rmsprop = tf.keras.optimizers.RMSprop(learning_rate=rmsprop_learning_rate)



# Adagrad

adagrad_learning_rate = 0.01

adagrad = tf.keras.optimizers.Adagrad(learning_rate=adagrad_learning_rate)



# Stochastic gradient descent (sgd)

sgd_learning_rate = 0.01 # initial learning rate

sgd_decay_rate = 0.1

sgd_momentum = 0.8

sgd = keras.optimizers.SGD(learning_rate = sgd_learning_rate )



# Adadelta

adadelta_learning_rate=0.002

adadelta_rho=0.99

adadelta_epsilon=1e-07

adadelta = tf.keras.optimizers.Adadelta(learning_rate = adadelta_learning_rate,

                                        rho = adadelta_rho, 

                                        epsilon = adadelta_epsilon)
# compilation configuration

optimizer = adadelta



# loss = "sparse_categorical_crossentropy" # for integer encoding output

loss = "categorical_crossentropy" # for onehot encoding output



metrics = ["acc"]
# compile

model.compile(optimizer = optimizer, 

              loss = loss, 

              metrics = metrics)
# ModelCheckpoint

model_path = "./model/"

if not os.path.exists(model_path):

    os.mkdir(model_path)

mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(model_save_filename = "model.h5", monitor = "val_accuracy", 

                                                   filepath = model_path, save_best_only = True)



# learning schedule

logdir = './logs'

if not os.path.exists(logdir):

    os.mkdir(logdir)

logdir = logdir + "/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if not os.path.exists(logdir):

    os.mkdir(logdir)

file_writer_path = logdir + "/metrics"

if not os.path.exists(file_writer_path):

    os.mkdir(file_writer_path)

file_writer = tf.summary.create_file_writer(file_writer_path)

file_writer.set_as_default()

def exp_decay(epoch, lr):

    decay_rate = 0.07

    decay_step = 40

    if epoch % decay_step == 0 and epoch:

        return lr * decay_rate

    #tf.summary.scalar('learning rate', data=lr, step=epoch)

    return lr

learningrate_cb = keras.callbacks.LearningRateScheduler(exp_decay)



# TensorBoard

tensorboard_cb = keras.callbacks.TensorBoard(logdir)



# EarlyStopping

earlystopping_cb = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)

# traning configuration

NUM_EPOCHS = 15

callbacks_list = [#earlystopping_cb,

                  mdlcheckpoint_cb, 

                  tensorboard_cb,

#                   learningrate_cb

                 ]

model.load_weights("../input/model-v2-e45/model_v2_e45.h5", by_name=False)
# Training

!rm -rf ./logs/ # Clear any logs from previous runs

history = model.fit(

    x = train_generator,

    epochs = NUM_EPOCHS,

    validation_data = valid_generator,

    callbacks = callbacks_list,

    verbose = 1,

    class_weight = cw

)

model.save(model_path + "model_v2_e60.h5")
# load dataset and preprocess images（the preprocess function in map  can preprocess all previous pictures)

def load_and_preprocess_image(img_path):

    # read pictures

    img_raw = tf.io.read_file(img_path)

    # decode pictures

    img_tensor = tf.image.decode_jpeg(img_raw, channels=1) # change channels to 3 result in 3-channel image

    # resize

#     img_tensor = tf.image.resize(img_tensor, [128, 128])

    #tf.cast() function is a type conversion function that converts the data format of x into dtype

    img_tensor = tf.cast(img_tensor, tf.float32)

    # normalization

    img_tensor = img_tensor / 255.0

    # flip left or right

#     img_tensor = tf.image.random_flip_left_right(img_tensor)

    return img_tensor



# define valid path

valid_root = pathlib.Path(valid_filepath)

valid_image_path = valid_root.rglob('*.jpg')

valid_image_path = [str(pathlib.Path(path)) for path in valid_image_path]

print("total valid: ", len(valid_image_path))



# get index of label and freq 

valid_freq = [int(path.split('/')[-1].split('_')[1]) for path in valid_image_path]

valid_label = [path.split('/')[-2] for path in valid_image_path]

oh_valid_freq = [freq_dict.get(freq) for freq in valid_freq]

oh_valid_label = label_to_onehot.transform(np.array(valid_label).reshape(-1, 1))



# convert to tensor and zip

valid_image_ds = tf.data.Dataset.from_tensor_slices(valid_image_path).map(load_and_preprocess_image)

valid_freq_ds = tf.data.Dataset.from_tensor_slices(oh_valid_freq)

valid_label_ds = tf.data.Dataset.from_tensor_slices(oh_valid_label)

valid_ds = tf.data.Dataset.zip(((valid_image_ds, valid_freq_ds), valid_label_ds))

valid_ds = valid_ds.batch(len(valid_image_path)).repeat()

iteration_valid = iter(valid_ds)
pred_prob = model.predict(iteration_valid.get_next())

pred_class_prob = np.amax(pred_prob, 1)

pred_class_integer = np.argmax(pred_prob, 1)

pred_class = [list(label_dict)[i] for i in pred_class_integer]

cm = confusion_matrix(valid_label, pred_class)
plt.figure(figsize=(15, 5))

ConfusionMatrixDisplay(cm, display_labels = list(label_dict.keys())).plot(cmap = "BuGn", 

                                                                          include_values= False)

plt.xticks(rotation=45)
df = pd.DataFrame(data=pred_prob, columns=list(label_dict.keys()))

df["actual_class"] = valid_label

df = df.reset_index()

df = df.melt(id_vars = ["index", "actual_class"], var_name = "pred_class", value_name = "probability")

df = df.sort_values(by = ['index', 'actual_class']).reset_index(drop = True)

grouped_max = df[['index', 'probability']].groupby('index').max().reset_index().rename(columns={"probability": "max_probability"})

df = pd.merge(df, grouped_max, on='index')

df['max_probability'] = np.where(df['probability'] == df['max_probability'], True, False)

df = df[['index', 'actual_class', 'pred_class', 'max_probability', 'probability']]



df['correctly_pred'] = np.where((df['pred_class'] == df['actual_class']) & (df['max_probability'] == True), True, False)

df = df[(df['correctly_pred']==True) | (df['max_probability']==True) ]
plt.figure(figsize=(15, 5))

sns.boxplot(x="actual_class", y="probability", 

            hue = "correctly_pred", data = df)

plt.legend(loc='upper left')

plt.xticks(rotation=45)
plt.figure(figsize=(15, 5))

sns.countplot(x="actual_class", hue = "correctly_pred", data = df)

plt.legend(loc='upper left')

plt.xticks(rotation=45)
f1_score(valid_label, pred_class, average= "weighted")
# %reload_ext tensorboard

# %tensorboard --logdir {logdir}
# test_loss, test_accuracy = model.evaluate(test_ds)

# print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))
from tensorflow.keras.models import Model

from tensorflow.python.framework import ops



class GradCAM:

    # Adapted with some modification from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/

    def __init__(self, model, layerName=None):

        """

        model: pre-softmax layer (logit layer)

        """

        self.model = model

        self.layerName = layerName



    def compute_heatmap(self, input_ds, classIdx, upsample_size, eps=1e-5):

        gradModel = Model(inputs=[self.model.inputs],

                          outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        # record operations for automatic differentiation

        with tf.GradientTape() as tape:

            (convOuts, preds) = gradModel(input_ds)  # preds after softmax

            loss = preds[:, classIdx]

        # compute gradients with automatic differentiation

        grads = tape.gradient(loss, convOuts)

        

        # discard batch

        convOuts = convOuts[0]

        grads = grads[0]

        

        # normalize grads

        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))

        

        # compute weights

        weights = tf.reduce_mean(norm_grads, axis=(0, 1))

        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)

        

        # Apply reLU

        cam = np.maximum(cam, 0)

        cam = cam / np.max(cam)

        cam = cv2.resize(cam, upsample_size, cv2.INTER_LINEAR)



#         print("[convOuts] shape: {}, max: {}, min: {}".format(convOuts.numpy().shape, convOuts.numpy().max(), convOuts.numpy().min()))

#         print("[loss] shape: {}, max: {}, min: {}".format(loss.numpy().shape, loss.numpy().max(), loss.numpy().min()))

#         print("[grads] shape: {}, max: {}, min: {}".format(grads.numpy().shape, grads.numpy().max(), grads.numpy().min()))

        

        return cam

def overlay_gradCAM(img, cam):

    def remap(x, out_min, out_max):

        return (x - x.min()) * (out_max - out_min) / (x.max() - x.min()) + out_min

    cam = remap(cam, 0, 255)

    cam = np.uint8(cam)

    new_img = 0.3 * cam + 0.5 * img

    new_img = np.array(new_img)

    new_img = remap(new_img, 0, 255)

    

    return (new_img * 255.0 / new_img.max()).astype("uint8")



def guidedRelu(x):

    def grad(dy):

        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy

    return tf.nn.relu(x), grad



# Reference: https://github.com/eclique/keras-gradcam with adaption to tensorflow 2.0  

class GuidedBackprop:

    def __init__(self,model, layerName=None):

        self.model = model

        self.layerName = layerName

        self.gbModel = self.build_guided_model()



    def build_guided_model(self):

        gbModel = Model(

            inputs = [self.model.inputs],

            outputs = [self.model.get_layer(self.layerName).output]

        )

        layer_dict = [layer for layer in gbModel.layers[1:] if hasattr(layer,"activation")]

        for layer in layer_dict:

            if layer.activation == tf.keras.activations.relu:

                layer.activation = guidedRelu        

        return gbModel

    

    def guided_backprop(self, input_ds, upsample_size):

        """Guided Backpropagation method for visualizing input saliency."""

        with tf.GradientTape() as tape:

            tape.watch(input_ds)

            outputs = self.gbModel(input_ds)

        grads = tape.gradient(outputs, input_ds)[0][0,:,:,:]

        saliency = cv2.resize(src = np.float32(grads), dsize = upsample_size, interpolation = cv2.INTER_AREA)

#         saliency = grads

        return saliency





def deprocess_image(x):

    """Same normalization as in:

    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py

    """

    # normalize tensor: center on 0., ensure std is 0.25

    x = x.copy()

    x -= x.mean()

    x /= (x.std() + K.epsilon())

    x *= 0.25



    # clip to [0, 1]

    x += 0.5

    x = np.clip(x, 0, 1)



    # convert to RGB array

    x *= 255

    if K.image_data_format() == 'channels_first':

        x = x.transpose((1, 2, 0))

    x = np.clip(x, 0, 255).astype('uint8')

#     def remap(x, out_min, out_max):

#         return (x - x.min()) * (out_max - out_min) / (x.max() - x.min()) + out_min

#     x = remap(x, 0, 255).astype('uint8')

    

    return x
def show_gradCAMs(model, gradCAM, GuidedBP, image_inputs, aux_inputs, classIdx , upsample_size,n=3, decode={}):

    """

    this work for tensorflow 2.x for input image with shape (none, x, x, 1)

    modified from

    https://github.com/nguyenhoa93/GradCAM_and_GuidedGradCAM_tf2/blob/master/src/guidedBackprop.py

    """

    plt.subplots(figsize=(25, 10*n))

    k=1

    for j in range(n):

        # define instance

        i = random.randint(0,n)

        image_input = load_and_preprocess_image(image_inputs[i])

        aux_input = aux_inputs[i]

        input_ds = [tf.cast(np.expand_dims(image_input, axis=0), tf.float32), 

                    tf.cast(np.expand_dims(aux_input, axis=0), tf.float32)]

#         input_ds = tf.concat([x, y], axis = 1)

        image_input = image_input[:,:,0]

        # record the image size

        upsample_size = (image_input.shape[1], image_input.shape[0])

        

        # Show original image

        plt.subplot(n,3,k)

        plt.imshow(image_input)

#         plt.title("class: {}".format(classIdx), fontsize=20)

        plt.axis("off")

        

        # Show overlayed grad

        plt.subplot(n,3,k+1)

        preds = model.predict(input_ds)

        idx = preds.argmax()

        

        # decode result in form of [class, prob]

        if len(decode)==0:

            res = tf.keras.applications.imagenet_utils.decode_predictions(preds)#[0][0][1:]

        else:

            res = [list(decode)[idx], preds.max()]    

            

        # compute cam and gb

        cam = gradCAM.compute_heatmap(input_ds = input_ds, classIdx=idx, upsample_size=upsample_size)

        gb = GuidedBP.guided_backprop(input_ds = input_ds, upsample_size = upsample_size)

        

#         print("[image_input] shape: {}, max: {}, min: {}".format(image_input.shape, image_input.max(), image_input.min()))

#         print("[cam] shape: {}, max: {}, min: {}".format(cam.shape, cam.max(), cam.min()))

#         print("[gb] shape: {}, max: {}, min: {}".format(gb.shape, gb.max(), gb.min()))

        

        # Show Gradient CAM 

        gradCAM_img = overlay_gradCAM(image_input, cam)

#         new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

        plt.imshow(gradCAM_img)

        plt.title("GradCAM - Pred: {}. Prob: {}".format(res[0],res[1]), fontsize=20)

        plt.axis("off")

        

        # Show guided GradCAM

        plt.subplot(n,3,k+2)

        guided_gradcam = deprocess_image(gb* cam)

        plt.imshow(guided_gradcam)

#         plt.imshow(gb, cmap=plt.cm.BuGn)

        plt.title("Guided GradCAM", fontsize=20)

        plt.axis("off")

        

        k += 3

    plt.show()

# layer and class

layerName = "activation_19"

actual_class = "aldfly"



# index

class_indices = label_dict

classIdx = label_dict.get(actual_class)



# input image path

class_root = pathlib.Path(valid_filepath + "/" +actual_class)

class_image_path = class_root.rglob('*.jpg')

class_image_path = [str(pathlib.Path(path)) for path in class_image_path]



# get index of label and freq 

class_freq = [int(path.split('/')[-1].split('_')[1]) for path in class_image_path]

class_label = [path.split('/')[-2] for path in class_image_path]

encode_class_freq = [freq_dict.get(freq) for freq in class_freq]

oh_class_label = label_to_onehot.transform(np.array(class_label).reshape(-1, 1))



image_inputs = class_image_path

aux_inputs = list(encode_class_freq)
# launch

gradCAM = GradCAM(model = model, layerName = layerName)

guidedBP = GuidedBackprop(model = model, layerName = layerName)

print("actual class: ",actual_class)

show_gradCAMs(model, gradCAM, guidedBP, 

              image_inputs = image_inputs, aux_inputs = aux_inputs, 

              decode= class_indices, classIdx = classIdx, 

              upsample_size = (128, 128), n=20)
