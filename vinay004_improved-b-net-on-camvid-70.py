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
from keras.layers import Dense, GlobalAveragePooling2D,Conv2D, MaxPooling2D,Conv2DTranspose
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization,LeakyReLU,Concatenate
from keras import backend as K
from keras import losses
from scipy import io

from scipy import ndimage
from keras.utils import plot_model
from keras.utils import np_utils

print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
PATH = "/kaggle/input/camvid/CamVid"
def equaliseHistogram(image):
    h,s,v = cv2.split(image)
    v = cv2.equalizeHist(v)
    image = cv2.merge((h,s,v))
    return image
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

  train_batch = os.listdir(train_data)
  val_batch = os.listdir(val_data)

  train_label_batch = os.listdir(train_label)
  val_label_batch = os.listdir(val_label)

  X_train = []
  y_train = []

  X_val = []
  y_val = []
  for image_idx in range(len(train_batch)):
    tempTrainImage = equaliseHistogram(cv2.cvtColor(cv2.imread(train_data + train_batch[image_idx],-1),cv2.COLOR_BGR2HSV))
    tempTrainLabelImage = equaliseHistogram(cv2.cvtColor(cv2.imread(train_label + train_label_batch[image_idx],-1),cv2.COLOR_BGR2HSV))
    
    cv2.cvtColor(tempTrainImage,cv2.COLOR_HSV2RGB)
    cv2.cvtColor(tempTrainLabelImage,cv2.COLOR_HSV2RGB)

    
    inputImage,MaskImage = load_image(tempTrainImage,tempTrainLabelImage, dsize)
    X_train.append(inputImage)
    y_train.append(MaskImage)


  
  for image_idx in range(len(val_batch)):
    tempValImage = equaliseHistogram(cv2.cvtColor(cv2.imread(val_data + val_batch[image_idx],-1),cv2.COLOR_BGR2HSV))
    tempValLabelImage = equaliseHistogram(cv2.cvtColor(cv2.imread(val_label + val_label_batch[image_idx],-1),cv2.COLOR_BGR2HSV))
    
    tempValImage = cv2.cvtColor(tempValImage,cv2.COLOR_HSV2RGB)
    tempValLabelImage = cv2.cvtColor(tempValLabelImage,cv2.COLOR_HSV2RGB)
    
    inputImage,MaskImage = load_image(tempValImage,tempValLabelImage, dsize)
    X_val.append(inputImage)
    y_val.append(MaskImage)

  return X_train,y_train,X_val,y_val

def load_image_train():
  
  train_data = PATH + "/train/"
  val_data = PATH + "/val/"

  train_label = PATH + "/train_labels/"
  val_label  = PATH + "/val_labels/"

  train_batch = os.listdir(train_data)
  val_batch = os.listdir(val_data)

  train_label_batch = os.listdir(train_label)
  val_label_batch = os.listdir(val_label)

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
base_model =  applications.mobilenet_v2.MobileNetV2(input_shape=[256,256,3], include_top=False, weights='imagenet')
base_model.summary()

output_layers = [   'block_1_expand_relu' ,   #128x128
                    'block_2_expand_relu' ,   #64x64
                    'block_4_expand_relu' ,   #32x32
                    'block_8_expand_relu' ,   #16x16
                    'block_16_expand_relu']   #8x8

layer_outputs = []
for count_output in range(len(output_layers)):
    layer_outputs.append(base_model.get_layer(output_layers[count_output]).output)
    
down_stack = Model(inputs = base_model.input,outputs = layer_outputs)
down_stack.trainable = False
def upsample(filters , size,strides = 2, Batch_Norm = False, apply_dropout = False):
    result = Sequential()
    result.add(Conv2DTranspose(
    filters = filters, kernel_size = size, strides=2, padding='same',
    kernel_initializer='glorot_uniform'))
    
    if Batch_Norm:
        result.add(BatchNormalization())
    if apply_dropout:
        result.add(Dropout(0.4))
    
    result.add(LeakyReLU(alpha=0.01))
    return result
def BNETv2(output_channels = 3):
    inputs = layers.Input(shape = [256,256,3])
    x = inputs
    
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])
    Batch_Norm = True
    apply_dropout = True
    up_stack = [upsample(512,4,Batch_Norm,apply_dropout) ,
                upsample(256,4,Batch_Norm,apply_dropout) ,
                upsample(128,4,Batch_Norm,apply_dropout) ,
                upsample(64,4,Batch_Norm,apply_dropout) ]
    
    for up,skip in zip(up_stack,skips):
        x = up(x)
        concat = Concatenate()
        x = concat([x,skip])
        
    last = Conv2DTranspose(output_channels, 4, strides=2,padding='same' , )
    x = last(x)
    
    return Model(inputs = inputs,outputs = x)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
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
def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    
    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
    (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss

def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
            y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_bce_loss(y_true, y_pred, weight) + \
    weighted_dice_loss(y_true, y_pred, weight)
    return loss
model2 = BNETv2(output_channels = 3)
model2.compile(optimizer=optimizers.Adam(lr=0.0003), loss=weighted_bce_dice_loss, metrics=['accuracy'])
plot_model(model2, show_shapes=True)
print('# Fit model on training data')
epochs = 200
batch_size =128

checkpoint2 = ModelCheckpoint('bnetv2_Loss.hdf5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')
with tf.device('/device:GPU:0'):
    
    history_loss = model2.fit(np.array(X_train), np.array(y_train),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks = [checkpoint2],
                        validation_data=(np.array(X_val), np.array(y_val)))
model2.load_weights('bnetv2_Loss.hdf5')
X_test,y_test = load_image_test()
print('\n# Evaluate on test data')
results = model2.evaluate(np.array(X_test),np.array(y_test),batch_size=128)
print('test loss, test acc:', results)
plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
plt.tight_layout()
ax = plt.subplot(211)
ax.set_facecolor('#F8F8F8')
ax.plot(history_loss.history['loss'] )
ax.plot(history_loss.history['val_loss'] )
ax.set_title('model')
ax.set_ylabel('BNET')
#ax.set_ylim(0.28,1.05)
ax.set_xlabel('epoch')
ax.legend(['train', 'valid.'])


ax2 = plt.subplot(212)
ax2.set_facecolor('#F8F8F8')
ax2.plot(history_loss.history['accuracy'])
ax2.plot(history_loss.history['val_accuracy'] )
ax2.set_title('model ')
ax2.set_ylabel('BNET')
#ax.set_ylim(0.28,1.05)
ax2.set_xlabel('epoch')
ax2.legend(['train', 'valid.'])
for i in range(5):
    pred_mask = model2.predict(np.expand_dims(X_train[i], axis=0))
    display([X_train[i], y_train[i], create_mask(pred_mask)])

