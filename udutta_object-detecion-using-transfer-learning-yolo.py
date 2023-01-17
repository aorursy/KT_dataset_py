# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from utils import WeightReader, decode_netout, draw_boxes

from yolo1_preprocessing import parse_annotation, BatchGenerator

import sys

print(sys.version)

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#lets check what all images we have to check the performance of the model. 

import os

folder=os.listdir('/kaggle/input/yolo-weights-inputs/downloaded_images/downloaded_images')

folder
from keras.models import Sequential, Model

from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda

from keras.layers.advanced_activations import LeakyReLU

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from keras.optimizers import SGD, Adam, RMSprop

from keras.layers.merge import concatenate

import matplotlib.pyplot as plt

import keras.backend as K

import tensorflow as tf

import imgaug as ia

from tqdm import tqdm

from imgaug import augmenters as iaa

import numpy as np

import pickle

import os, cv2

# from preprocessing import parse_annotation, BatchGenerator

# from utils import WeightReader, decode_netout, draw_boxes #normalize

# The above two imports are imported seperately by first creating a script and then adding te scipt to our notebook

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = ""



%matplotlib inline
LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']



IMAGE_H, IMAGE_W = 416, 416

GRID_H,  GRID_W  = 13 , 13

BOX              = 5

CLASS            = len(LABELS)

CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')

OBJ_THRESHOLD    = 0.3#0.5

NMS_THRESHOLD    = 0.3#0.45

ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]



NO_OBJECT_SCALE  = 1.0

OBJECT_SCALE     = 5.0

COORD_SCALE      = 1.0

CLASS_SCALE      = 1.0



BATCH_SIZE       = 16

WARM_UP_BATCHES  = 0

TRUE_BOX_BUFFER  = 50
print(F"Total number of classes :{len(LABELS)}")

print(f"some of the labels: {LABELS[10:20]}")
wt_path = '../input/yolo-weights-inputs/yolo.weights'     

# the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)

def space_to_depth_x2(x):

    import tensorflow as tf

    return tf.nn.space_to_depth(x, block_size=2)
import sys

input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))

true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))



# Layer 1

x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)

x = BatchNormalization(name='norm_1')(x)

x = LeakyReLU(alpha=0.1)(x)

x = MaxPooling2D(pool_size=(2, 2))(x)



# Layer 2

x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)

x = BatchNormalization(name='norm_2')(x)

x = LeakyReLU(alpha=0.1)(x)

x = MaxPooling2D(pool_size=(2, 2))(x)



# Layer 3

x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)

x = BatchNormalization(name='norm_3')(x)

x = LeakyReLU(alpha=0.1)(x)



# Layer 4

x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)

x = BatchNormalization(name='norm_4')(x)

x = LeakyReLU(alpha=0.1)(x)



# Layer 5

x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)

x = BatchNormalization(name='norm_5')(x)

x = LeakyReLU(alpha=0.1)(x)

x = MaxPooling2D(pool_size=(2, 2))(x)



# Layer 6

x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)

x = BatchNormalization(name='norm_6')(x)

x = LeakyReLU(alpha=0.1)(x)



# Layer 7

x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)

x = BatchNormalization(name='norm_7')(x)

x = LeakyReLU(alpha=0.1)(x)



# Layer 8

x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)

x = BatchNormalization(name='norm_8')(x)

x = LeakyReLU(alpha=0.1)(x)

x = MaxPooling2D(pool_size=(2, 2))(x)



# Layer 9

x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)

x = BatchNormalization(name='norm_9')(x)

x = LeakyReLU(alpha=0.1)(x)



# Layer 10

x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)

x = BatchNormalization(name='norm_10')(x)

x = LeakyReLU(alpha=0.1)(x)



# Layer 11

x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)

x = BatchNormalization(name='norm_11')(x)

x = LeakyReLU(alpha=0.1)(x)



# Layer 12

x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)

x = BatchNormalization(name='norm_12')(x)

x = LeakyReLU(alpha=0.1)(x)



# Layer 13

x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)

x = BatchNormalization(name='norm_13')(x)

x = LeakyReLU(alpha=0.1)(x)



skip_connection = x



x = MaxPooling2D(pool_size=(2, 2))(x)



# Layer 14

x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)

x = BatchNormalization(name='norm_14')(x)

x = LeakyReLU(alpha=0.1)(x)



# Layer 15

x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)

x = BatchNormalization(name='norm_15')(x)

x = LeakyReLU(alpha=0.1)(x)



# Layer 16

x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)

x = BatchNormalization(name='norm_16')(x)

x = LeakyReLU(alpha=0.1)(x)



# Layer 17

x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)

x = BatchNormalization(name='norm_17')(x)

x = LeakyReLU(alpha=0.1)(x)



# Layer 18

x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)

x = BatchNormalization(name='norm_18')(x)

x = LeakyReLU(alpha=0.1)(x)



# Layer 19

x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)

x = BatchNormalization(name='norm_19')(x)

x = LeakyReLU(alpha=0.1)(x)



# Layer 20

x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)

x = BatchNormalization(name='norm_20')(x)

x = LeakyReLU(alpha=0.1)(x)



# Layer 21

skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)

skip_connection = BatchNormalization(name='norm_21')(skip_connection)

skip_connection = LeakyReLU(alpha=0.1)(skip_connection)

skip_connection = Lambda(space_to_depth_x2)(skip_connection)



x = concatenate([skip_connection, x])



# Layer 22

x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)

x = BatchNormalization(name='norm_22')(x)

x = LeakyReLU(alpha=0.1)(x)



# Layer 23

x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)

output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)



# small hack to allow true_boxes to be registered when Keras build the model 

# for more information: https://github.com/fchollet/keras/issues/2790

output = Lambda(lambda args: args[0])([output, true_boxes])



model = Model([input_image, true_boxes], output)
model.summary()
weight_reader = WeightReader(wt_path)
weight_reader.reset()

nb_conv = 23



for i in range(1, nb_conv+1):

    conv_layer = model.get_layer('conv_' + str(i))

    

    if i < nb_conv:

        norm_layer = model.get_layer('norm_' + str(i))

        

        size = np.prod(norm_layer.get_weights()[0].shape)



        beta  = weight_reader.read_bytes(size)

        gamma = weight_reader.read_bytes(size)

        mean  = weight_reader.read_bytes(size)

        var   = weight_reader.read_bytes(size)



        weights = norm_layer.set_weights([gamma, beta, mean, var])       

        

    if len(conv_layer.get_weights()) > 1:

        bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))

        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))

        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))

        kernel = kernel.transpose([2,3,1,0])

        conv_layer.set_weights([kernel, bias])

    else:

        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))

        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))

        kernel = kernel.transpose([2,3,1,0])

        conv_layer.set_weights([kernel])
dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
images=os.listdir('../input/yolo-weights-inputs/downloaded_images/downloaded_images')

images=images[4:]
images=os.listdir('../input/yolo-weights-inputs/downloaded_images/downloaded_images')

images=images[3:]

for file in images:

#     image=os.path.join()

    image = cv2.imread(os.path.join('../input/yolo-weights-inputs/downloaded_images/downloaded_images' ,file))

    dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))



    plt.figure(figsize=(10,10))



    input_image = cv2.resize(image, (416, 416))

    input_image = input_image / 255.

    input_image = input_image[:,:,::-1]

    input_image = np.expand_dims(input_image, 0)



    netout = model.predict([input_image, dummy_array])



    boxes = decode_netout(netout[0], 

                          obj_threshold=OBJ_THRESHOLD,

                          nms_threshold=NMS_THRESHOLD,

                          anchors=ANCHORS, 

                          nb_class=CLASS)



    image = draw_boxes(image, boxes, labels=LABELS)



    plt.imshow(image[:,:,::-1]); plt.show()