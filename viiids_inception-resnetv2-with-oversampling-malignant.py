import numpy as np
import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import load_img
from keras import regularizers
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications import imagenet_utils
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.callbacks.callbacks import EarlyStopping
from keras.callbacks.callbacks import ReduceLROnPlateau
from keras.callbacks.callbacks import ModelCheckpoint
import pydot
from IPython.display import SVG
from keras.utils import plot_model
from keras.utils import to_categorical
from random import randint

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import random
import cv2

import tensorflow as tf

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
# Setting up TPU.
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() 
print("REPLICAS: ", strategy.num_replicas_in_sync)
ROOT = '/kaggle/input/siim-isic-melanoma-classification/'
TRAIN_IMAGES_PATH = ROOT + 'jpeg/train/'
TEST_IMAGES_PATH = ROOT + 'jpeg/test/'
EXTENSION = '.jpg'
IMAGE_DIM_X = 299
IMAGE_DIM_Y = 299
BATCH_SIZE = 8

inception_preprocess = preprocess_input

train_data = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
test_data = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
sample_submission = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
TEST_SIZE = len(test_data)

train_data.rename(columns={
    'anatom_site_general_challenge': 'area', 
    'benign_malignant': 'class',
    'target': 'label'
}, inplace=True)

img_path = TRAIN_IMAGES_PATH + train_data.iloc[0].image_name + EXTENSION
img = load_img(img_path)
np_img = img_to_array(img).astype(int)
np_img.shape
RGB = np_img.shape[2]
# Output: (4000, 6000, 3) = Very high resolution RGB images

train_data['image_name'] = train_data['image_name'].apply(
    lambda image_name : TRAIN_IMAGES_PATH + image_name + EXTENSION)
test_data['image'] = test_data['image_name']
test_data['image_name'] = test_data['image_name'].apply(
    lambda image_name : TEST_IMAGES_PATH + image_name + EXTENSION)

print(train_data.iloc[1].image_name)
print(test_data.iloc[1].image_name)
print(len(train_data))
train_data.head()
print(test_data.iloc[0].image_name)
test_data.head()
class_weights = class_weight.compute_class_weight(
    'balanced',
    train_data['label'].unique(),
    train_data[['label']].to_numpy().reshape(-1)
)

print('benign weight: ' + str(class_weights[0]))
print('malignant weight: ' + str(class_weights[1]))
X_train, X_val = train_test_split(train_data, test_size=0.1, shuffle=True)

X_train.head()
X_val.head()

TRAIN_SIZE = len(X_train)

def image_walker(input_data, image_name_col, label_cols, is_test=False):
    """
    Creates an ImageDataGenerator which serves two purposes:
    1. Loads images from a predefined directory and prepares batches for Gradient Descent.
    2. Performs data augmentation on those images.
    
    Parameters
    ----------
    input_paths : Dataframe containing a columns "paths" representing different paths from which an image
    is expected to be loaded.
    It is assumed that all the images are in the input/siim-isic-melanoma-classification/jpeg directory
    
    Returns
    -------
    iterator to the created ImageDataGenerator
    """
    data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    iterator = data_generator.flow_from_dataframe(
        input_data, 
        directory=None, 
        x_col=image_name_col, 
        y_col=label_cols,                                
        target_size=(IMAGE_DIM_X, IMAGE_DIM_Y),
        seed=42, 
        class_mode='raw',
        batch_size=BATCH_SIZE,
        shuffle=False,
        subset='training'
    )
    return iterator
print("[INFO] loading {}...".format("inception"))
base_model = InceptionResNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMAGE_DIM_X, IMAGE_DIM_Y, RGB))
CLASSES = 1
X = base_model.output
X = GlobalAveragePooling2D(name='avg_pool')(X)
predictions = Dense(CLASSES, activation='sigmoid')(X)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']
             )

early = EarlyStopping(monitor='accuracy', min_delta=0.1, patience=5, verbose=0, mode='auto', restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.0001, mode='auto')

model_checkpoint = ModelCheckpoint('incepresnetv2_ol.model', monitor='loss', save_best_only=True)
EPOCHS = 2
MODEL_FILE = 'filename_ol_dense.model'
def train():
    history = model.fit_generator(
        image_walker(X_train, 'image_name', 'label'),
        shuffle=True,
        epochs=EPOCHS,
        steps_per_epoch=int(TRAIN_SIZE/BATCH_SIZE),
        callbacks=[model_checkpoint, reduce_lr],
        validation_data=image_walker(X_val, 'image_name', 'label'),
        validation_steps=18,
        class_weight=class_weights
    )
    model.save(MODEL_FILE)
    return history

# Enable this if training is needed
# history = train()

# Use this if trained and ready for predictions
model = keras.models.load_model('/kaggle/input/incepresnet-v2/filename_ol_dense.model')
print('Finished loading/training model')
# REMOVE THIS LINE if you are using this notebook to submit.
test_data = test_data.head(100)

print(len(test_data))
TEST_BATCH_SIZE = 1

test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
test_predictions = model.predict_generator(
    test_generator.flow_from_dataframe(
        test_data, 
        x_col='image_name', 
        y_col=None,
        class_mode=None,
        target_size=(IMAGE_DIM_X, IMAGE_DIM_Y),
        batch_size=TEST_BATCH_SIZE, 
        shuffle=False), 
    steps=len(test_data)/TEST_BATCH_SIZE,
    verbose=1
    )
print(test_predictions.shape)
sample_submission.head()
test_data['target'] = test_predictions[:, 0]
test_data = test_data[['image', 'target']]
test_data.rename(columns={'image': 'image_name'}, inplace=True)
test_data.head()
test_data.to_csv('submission.csv', index=False)