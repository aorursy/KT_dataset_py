# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import math, re, os

import tensorflow as tf

from matplotlib import pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from sklearn.utils import shuffle

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

from keras import applications

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras import layers

from keras.callbacks import ModelCheckpoint

from keras.models import Sequential, Model

from keras.layers import Dense, GlobalAveragePooling2D,Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization

from keras import backend as K

from keras import losses

from scipy import io

from keras.utils import plot_model

from keras.utils import np_utils

print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         #print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
PATH = "/kaggle/input/CamVid"
def normalize(input_image, input_mask):

  input_image = tf.cast(input_image, tf.float32) / 255.0

  input_mask = tf.cast(input_mask, tf.float32) / 255.0

  return input_image, input_mask
def load_image(inputImage,MaskImage , dsize):

    input_image = cv2.resize(inputImage, dsize, interpolation = cv2.INTER_AREA)

    input_mask = cv2.resize(MaskImage, dsize, interpolation = cv2.INTER_AREA)



    input_image =  cv2.normalize(input_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    input_mask =  cv2.normalize(input_mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)





    return input_image, input_mask
def load_image_train():

  

  train_data = PATH + "/train/"

  val_data = PATH + "/val/"



  train_label = PATH + "/train_labels/"

  val_label  = PATH + "/val_labels/"



  train_batch = sorted(os.listdir(train_data))

  val_batch = sorted(os.listdir(val_data))



  train_label_batch = sorted(os.listdir(train_label))

  val_label_batch = sorted(os.listdir(val_label))



  X_train = []

  y_train = []



  X_val = []

  y_val = []

  for image_idx in range(len(train_batch)):

    tempTrainImage = cv2.cvtColor(cv2.imread(train_data + train_batch[image_idx],-1),cv2.COLOR_BGR2RGB)

    tempTrainLabelImage = cv2.cvtColor(cv2.imread(train_label + train_label_batch[image_idx],-1),cv2.COLOR_BGR2RGB)

    

    inputImage,MaskImage = load_image(tempTrainImage,tempTrainLabelImage, dsize)

    X_train.append(inputImage)

    y_train.append(MaskImage)





  

  for image_idx in range(len(val_batch)):

    tempValImage = cv2.cvtColor(cv2.imread(val_data + val_batch[image_idx],-1),cv2.COLOR_BGR2RGB)

    tempValLabelImage = cv2.cvtColor(cv2.imread(val_label + val_label_batch[image_idx],-1),cv2.COLOR_BGR2RGB)

    

    

    inputImage,MaskImage = load_image(tempValImage,tempValLabelImage, dsize)

    X_val.append(inputImage)

    y_val.append(MaskImage)



  return X_train,y_train,X_val,y_val

def load_image_test():

  

  test_data = PATH + "/test/"



  test_label = PATH + "/test_labels/"



  test_batch = os.listdir(test_data)

  test_label_batch = os.listdir(test_label)



  X_test = []

  y_test = []





  for image_idx in range(len(test_batch)):

    tempTestImage = cv2.cvtColor(cv2.imread(test_data + test_batch[image_idx],-1),cv2.COLOR_BGR2RGB)

    tempTestLabelImage = cv2.cvtColor(cv2.imread(test_label + test_label_batch[image_idx],-1),cv2.COLOR_BGR2RGB)

    

    inputImage,MaskImage = load_image(tempTestImage,tempTestLabelImage, dsize)

    X_test.append(inputImage)

    y_test.append(MaskImage)





  return X_test,y_test
# try:

#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

#     print('Running on TPU ', tpu.master())

# except ValueError:

#     tpu = None



# if tpu:

#     tf.config.experimental_connect_to_cluster(tpu)

#     tf.tpu.experimental.initialize_tpu_system(tpu)

#     strategy = tf.distribute.experimental.TPUStrategy(tpu)

# else:

#     strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



# print("REPLICAS: ", strategy.num_replicas_in_sync)
tf.test.gpu_device_name()
dsize = (256,256)

X_train,y_train,X_val,y_val = load_image_train()
print(np.shape(X_train))

print(np.shape(y_train))

print(np.shape(X_val))

print(np.shape(y_val))
def get_label_values():

  df = pd.read_csv(PATH + "/class_dict.csv")

  label_values = []

  class_names_list = df.name.to_list()

  df.index = df.name

  df = df.drop(columns= ['name'])

  for i in range(len(df)):

    label_values.append(np.array(df.iloc[i]))

  num_classes = len(label_values)

  return label_values, class_names_list, num_classes
label_values, class_names_list, num_classes = get_label_values()
def BNET(output_channel):



  size = 4

  init_kernel = 'he_uniform'

  model = Sequential()

  model.add(Conv2D(filters=64,kernel_size=size,strides=2,padding='same' , kernel_initializer=init_kernel,use_bias = False,input_shape = (256,256,3)))

  model.add(Conv2D(filters=64,kernel_size=size,strides=2,padding='same' , kernel_initializer=init_kernel,use_bias = False))

  model.add(BatchNormalization())

  model.add(Conv2D(filters=128,kernel_size=size,strides=2,padding='same' , kernel_initializer=init_kernel,use_bias = False))

  model.add(BatchNormalization())

  model.add(Conv2D(filters=128,kernel_size=size,strides=2,padding='same' , kernel_initializer=init_kernel,use_bias = False))

  model.add(BatchNormalization())

  model.add(Conv2D(filters=256,kernel_size=size,strides=2,padding='same' , kernel_initializer=init_kernel,use_bias = False))

  model.add(BatchNormalization())

  model.add(Conv2D(filters=256,kernel_size=size,strides=2,padding='same' , kernel_initializer=init_kernel,use_bias = False))

  model.add(BatchNormalization())

  model.add(Conv2D(filters=512,kernel_size=size,strides=2,padding='same' , kernel_initializer=init_kernel,use_bias = False))

  model.add(BatchNormalization())  



  model.add(layers.Conv2DTranspose(filters=512,kernel_size=size,strides=2,padding='same',kernel_initializer=init_kernel))

  model.add(layers.Dropout(0.2))

  model.add(BatchNormalization())

  model.add(layers.Conv2DTranspose(filters=256,kernel_size=size,strides=2,padding='same',kernel_initializer=init_kernel))

  model.add(layers.Dropout(0.2))

  model.add(BatchNormalization())

  model.add(layers.Conv2DTranspose(filters=256,kernel_size=size,strides=2,padding='same',kernel_initializer=init_kernel))

  model.add(BatchNormalization())

  model.add(layers.Conv2DTranspose(filters=128,kernel_size=size,strides=2,padding='same',kernel_initializer=init_kernel))

  model.add(BatchNormalization())

  model.add(layers.Conv2DTranspose(filters=128,kernel_size=size,strides=2,padding='same',kernel_initializer=init_kernel))

  model.add(BatchNormalization())  

  model.add(layers.Conv2DTranspose(filters=64,kernel_size=size,strides=2,padding='same',kernel_initializer=init_kernel))

  model.add(layers.Conv2DTranspose(output_channel,4,strides=2,padding='same' , kernel_initializer=init_kernel,activation='tanh'))



 

  print(model.summary())

  return model

  
model = BNET(output_channel = 3)

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics = ['accuracy'])
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
print('# Fit model on training data')

epochs = 160

batch_size =128



checkpoint = ModelCheckpoint('bnet_membrane.hdf5',

                                 monitor='val_loss',

                                 verbose=0,

                                 save_best_only=True,

                                 mode='auto')

with tf.device('/device:GPU:0'):

    history = model.fit(np.array(X_train), np.array(y_train),

                        batch_size=batch_size,

                        epochs=80,

                        callbacks = [checkpoint],

                        validation_data=(np.array(X_val), np.array(y_val)))
X_test,y_test = load_image_test()

print('\n# Evaluate on test data')

results = model.evaluate(np.array(X_test),np.array(y_test),batch_size=64)

print('test loss, test acc:', results)
model = BNET(3)

model.compile(optimizer='adam',

              loss=losses.CategoricalCrossentropy(from_logits = True),

              metrics=['accuracy'])
plot_model(model, show_shapes=True)
print('# Fit model on training data')

epochs = 150

batch_size =128



checkpoint2 = ModelCheckpoint('bnet_categorical.hdf5',

                                 monitor='val_loss',

                                 verbose=1,

                                 save_best_only=True,

                                 mode='auto')

with tf.device('/device:GPU:0'):

    history2 = model.fit(np.array(X_train), np.array(y_train),

                        batch_size=batch_size,

                        epochs=epochs,

                        callbacks = [checkpoint2],

                        validation_data=(np.array(X_val), np.array(y_val)))
X_test,y_test = load_image_test()

print('\n# Evaluate on test data')

results = model.evaluate(np.array(X_test),np.array(y_test),batch_size=128)

print('test loss, test acc:', results)
plt.subplots(figsize=(10,10), facecolor='#F0F0F0')

plt.tight_layout()

ax = plt.subplot(211)

ax.set_facecolor('#F8F8F8')

ax.plot(history2.history['loss'])

ax.plot(history2.history['val_loss'])

ax.set_title('model')

ax.set_ylabel('BNET')

#ax.set_ylim(0.28,1.05)

ax.set_xlabel('epoch')

ax.legend(['train', 'valid.'])





ax2 = plt.subplot(212)

ax2.set_facecolor('#F8F8F8')

ax2.plot(history2.history['accuracy'])

ax2.plot(history2.history['val_accuracy'])

ax2.set_title('model ')

ax2.set_ylabel('BNET')

#ax.set_ylim(0.28,1.05)

ax2.set_xlabel('epoch')

ax2.legend(['train', 'valid.'])
plt.subplots(figsize=(10,10), facecolor='#F0F0F0')

plt.tight_layout()

ax = plt.subplot(211)

ax.set_facecolor('#F8F8F8')

ax.plot(history2.history['loss'])

ax.plot(history2.history['val_loss'])

ax.set_title('model')

ax.set_ylabel('BNET')

#ax.set_ylim(0.28,1.05)

ax.set_xlabel('epoch')

ax.legend(['train', 'valid.'])





ax2 = plt.subplot(212)

ax2.set_facecolor('#F8F8F8')

ax2.plot(history2.history['accuracy'])

ax2.plot(history2.history['val_accuracy'])

ax2.set_title('model ')

ax2.set_ylabel('BNET')

#ax.set_ylim(0.28,1.05)

ax2.set_xlabel('epoch')

ax2.legend(['train', 'valid.'])
def display(display_list):

  plt.figure(figsize=(15, 15))



  title = ['Input Image', 'True Mask', 'Predicted Mask']



  for i in range(len(display_list)):

    plt.subplot(1, len(display_list), i+1)

    plt.title(title[i])

    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))

    plt.axis('off')

  plt.show()
def create_mask(pred_mask):

  pred_mask = tf.argmax(pred_mask, axis=-1)

  pred_mask = pred_mask[..., tf.newaxis]

  return pred_mask[0]
pred_mask = model.predict(np.expand_dims(X_train[2], axis=0))

display([X_train[2], y_train[2], create_mask(pred_mask)])