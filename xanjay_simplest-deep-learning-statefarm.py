import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visulization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os

import gc # garbage collection

import glob # extract path via pattern matching

from tqdm.notebook import tqdm # progressbar

import random

import math

import cv2 # read image

# store to disk

import pickle

import h5py # like numpy array





from sklearn.model_selection import train_test_split

from keras.utils import to_categorical



from keras.models import Sequential, Model

from keras.models import load_model

from keras.layers import Input, Dense, Conv2D, MaxPool2D, AveragePooling2D

from keras.layers import Flatten, Dropout, BatchNormalization, Activation

from keras.layers import Add

from keras.optimizers import SGD, RMSprop, Adam

from keras import regularizers



from keras.callbacks import ModelCheckpoint, EarlyStopping

# from keras.applications.vgg16 import VGG16
# from IPython.display import FileLink

# FileLink('../input/pretrained-models/my_model.h5')

# keras config file

# !sudo $HOME/.keras/keras.json
# which keras backend

from keras import backend as K

print("backend:", K.backend())

print("image_format:", K.image_data_format())

import tensorflow as tf

tf.test.is_gpu_available() # True/False

# Or only check for gpu's with cuda support

tf.test.is_gpu_available(cuda_only=True) 

TRAIN_DIR = '../input/state-farm-distracted-driver-detection/imgs/train/'

TEST_DIR = '../input/state-farm-distracted-driver-detection/imgs/test/'
driver_imgs_list = pd.read_csv('../input/state-farm-distracted-driver-detection/driver_imgs_list.csv')

sample_submission = pd.read_csv('../input/state-farm-distracted-driver-detection/sample_submission.csv')
driver_imgs_list.head()
sample_submission.head()
def get_image(path, img_height=None, img_width=None, rotate=False, color_type=0):

    img = cv2.imread(path, color_type)

    if img_width and img_height:

        img = cv2.resize(img, (img_width, img_height))

    if rotate is True:

        rows, cols = img.shape

        rotation_angle = random.uniform(10,-10)

        M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation_angle, 1)

        img = cv2.warpAffine(img, M, (cols,rows))

    return img
def plot_images(image_paths):

    image_count = len(image_paths)

    cols = 3

    rows = math.ceil(image_count/cols)

    fig = plt.figure(figsize=(16, 4.5*rows))



    for i in range(1,image_count+1):

        fig.add_subplot(rows, cols, i)

        image = get_image(image_paths[i-1])

        plt.imshow(image, cmap="gray")

        plt.xticks([])

        plt.yticks([])

    plt.show()
# display_unique_drivers

driver_count = driver_imgs_list.subject.nunique()

image_paths = [TRAIN_DIR+row.classname+'/'+row.img 

               for (index, row) in driver_imgs_list.groupby('subject').head(1).iterrows()]

print(driver_count)

# plot images

plot_images(image_paths)

# class distribution

driver_imgs_list.classname.value_counts().plot(kind='bar')
# display_unique_classes

image_paths = [TRAIN_DIR+row.classname+'/'+row.img 

               for (index, row) in driver_imgs_list.groupby('classname').head(1).iterrows()]

# plot images

plot_images(image_paths)

random_list = np.random.permutation(len(driver_imgs_list))[:50]

df_copy = driver_imgs_list.iloc[random_list]

image_paths = [TRAIN_DIR+row.classname+'/'+row.img 

                   for (index, row) in df_copy.iterrows()]

image_shapes = [get_image(path).shape for path in image_paths]

print(set(image_shapes))
IMG_HEIGHT = 240

IMG_WIDTH = 320
def preprocess_data(X, Y, train = True):

    # normalize

    X = X/np.float32(255)

#     X = X.astype('float32')

    # reshape for grayscale image

    if len(X.shape) == 3:

        X = np.expand_dims(X, axis=-1)

    if train:

        # one hot encode target

        Y = to_categorical(Y, num_classes=10)

        return X, Y

    else:

        return X
# # load 1000 images per class from all 10 classes

# small_df = driver_imgs_list.groupby('classname').head(1000)

# image_paths = [TRAIN_DIR+row.classname+'/'+row.img 

#                for (index, row) in small_df.iterrows()]

# trainx = []

# trainy= []

# for item in image_paths:

#     trainx.append(get_image(item,img_height=IMG_HEIGHT,img_width=IMG_WIDTH))

#     trainy.append(item.split('/')[-2][-1]) # as c0,c1,etc.

# trainx = np.array(trainx)

# trainy = np.array(trainy,'uint8')

# #after preprocess function loaded

# Xtrain, Ytrain = preprocess_data(trainx, trainy)
def load_train_data(train_dir):

    data = []

    targets = []

    target_classes = os.listdir(train_dir)



    print("Image Size:", IMG_HEIGHT, IMG_WIDTH)

    print(target_classes)

    

    for c in tqdm(target_classes):

        class_dir = os.path.join(train_dir,c)

        items = glob.glob(os.path.join(class_dir,'*g'))

        for item in items:

            data.append(get_image(item,img_height=IMG_HEIGHT,img_width=IMG_WIDTH))

            targets.append(c[1])

    return np.array(data), np.array(targets, dtype='uint8')
Xtrain, Ytrain = load_train_data(TRAIN_DIR)
# del Xtrain, Ytrain

gc.collect()
Xtrain, Ytrain = preprocess_data(Xtrain, Ytrain)
print(Xtrain.shape,Ytrain.shape)
# load test data

def load_test_data(test_dir):

    data = []

    ids = []

    items = glob.glob(os.path.join(test_dir,'*g'))

    print(len(items))

    for item in tqdm(items):

        data.append(get_image(item,img_height=240,img_width=320))

        ids.append(os.path.basename(item))

    return np.array(data), ids

    
# run after memory cleanup and finished training

# Xtest, TEST_IDS = load_test_data(TEST_DIR)
# Xtest = preprocess_data(Xtest,None,train=False)
# dir()

# globals()

# locals()
gc.collect()
def save_pickle_file(filename, data):

    with open(filename, "wb") as f:

        pickle.dump(data, f)



def load_pickle_file(filename):

    with open(filename, "rb") as f:

        data = pickle.load(f)

    return data
X_train, X_valid, Y_train, Y_valid = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state=42)
# free up memory

del Xtrain, Ytrain

gc.collect()
EPOCHS=10

BATCH_SIZE=32
MODELS_DIR = "saved_models"

if not os.path.exists(MODELS_DIR):

    os.makedirs(MODELS_DIR)
from keras.callbacks.callbacks import ModelCheckpoint, EarlyStopping



filepath = MODELS_DIR+'/epoch{epoch:02d}-loss{loss:.2f}-val_loss{val_loss:.2f}.hdf5'

# checkpoint

model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 

                                   save_best_only=True, save_weights_only=False, mode='min', period=1)



# early stopping: patience = epocs

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1,

                               mode='min', baseline=None, restore_best_weights=True)
# No need for adaptive learning alogrithms like: RmsProp, Adam

from keras.callbacks.callbacks import LearningRateScheduler, ReduceLROnPlateau

# This function keeps the learning rate at 0.01 for the first five epochs

# and decreases it exponentially after that.

def learning_rate_scheduler(epoch):

  if epoch < 5:

    return 0.01

  else:

    return 0.01 * math.exp(0.1 * (5 - epoch))



lr_scheduler = LearningRateScheduler(learning_rate_scheduler, verbose=1)

# OR

# lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, 

#                                             verbose=1, mode='min', min_delta=0.0001, min_lr=0.0001)
def plot_model_loss(history):

    plt.plot(history['loss'])

    plt.plot(history['val_loss'])

    plt.title('Model Loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['train', 'valid'], loc='upper left')

    plt.show()

    
# use stride 2 in the middle to reduce size and increase no. of filters

# use avgpool at the end (not maxpool)

def seq_conv_block(model, filters=32):

    model.add(Conv2D(filters=filters, kernel_size=(3,3), strides=2, padding="same"))

    model.add(BatchNormalization(axis=-1))

    model.add(Activation("relu"))

    return model



# all conv layers  with strides=1

model = Sequential(name="seq_conv_rmsprop")



model.add(Conv2D(input_shape=(IMG_HEIGHT,IMG_WIDTH,1), filters=16, kernel_size=(3,3), padding="same"))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(BatchNormalization(axis=-1))

model.add(Activation("relu"))



model = seq_conv_block(model, filters=32)

model = seq_conv_block(model, filters=64)

model = seq_conv_block(model, filters=128)



model.add(Flatten())

model.add(Dropout(0.5))



model.add(Dense(500))

model.add(Dropout(0.5))

model.add(BatchNormalization(axis=-1))

model.add(Activation("relu"))



model.add(Dense(10, activation="softmax"))



# sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)

# optimizer = RMSprop(learning_rate=0.001)

# adad = Adam(learning_rate=0.001)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()
# history1 = model.fit(x=X_train,y=Y_train, validation_data=(X_valid, Y_valid),

#                           epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,

#                           callbacks=[model_checkpoint, early_stopping])
# Plot Model History

# plot_model_loss(history_v1.history)
# del X_train, X_valid, Y_train, Y_valid

gc.collect()
# Saving/loading whole models (architecture + weights + optimizer state)

# creates a HDF5 file 'my_model.h5'

# model.save('saved_models/my_model_4conv_block.h5')

# del model  # deletes the existing model
!ls saved_models
# returns a compiled model

model = load_model('../input/pretrained-models/my_model_4conv_block.h5')
# # save only architecture

# json_string = model.to_json()

# from keras.models import model_from_json

# model = model_from_json(json_string)

# #only weights

# model.save_weights('my_model_weights.h5')

# model.load_weights('my_model_weights.h5')
loss, accuracy = model.evaluate(X_valid, Y_valid)

print("Loss:", loss)

print("Accuracy:", accuracy)
def prepare_submission_df(predictions, ids):

    result_df = pd.DataFrame(predictions, columns=["c0","c1","c2","c3","c4","c5","c6","c7","c8","c9"])

    result_df['img'] = ids

    return result_df
test_items = glob.glob(os.path.join(TEST_DIR,'*g'))

print(len(test_items))



# pedict block-wise due to memory issue

block_size = 1000

total_test_size = len(test_items)

result_df = []

data = []

ids = []

count = 0

for item in tqdm(test_items):

    data.append(get_image(item, img_height=IMG_HEIGHT, img_width=IMG_WIDTH))

    ids.append(os.path.basename(item))

    if len(ids) == block_size or count == total_test_size-1:

        data = np.array(data)

        data = preprocess_data(data, None, train=False)

        df = prepare_submission_df(model.predict(data), ids)

        result_df.append(df)

        data = []

        ids = []

    

    count += 1

final_df = None

for df in result_df:

    if final_df is None:

        final_df = df

    else:

        final_df = final_df.append(df)



print(len(final_df))
final_df.to_csv("submission.csv", index=False)
# # Reordering DF columns

# cols = df.columns.tolist()

# cols = cols[-1] + cols[:-1]

# df = df[cols]
# !pip install kaggle

# !kaggle competitions submit -c state-farm-distracted-driver-detection -f submission.csv -m "First Submission."


def conv_layer(inputs, filters=16, num_strides=1):

    return Conv2D(filters=filters, kernel_size=(3,3), strides=num_strides, padding='same')(inputs)



def conv_block(inputs, filters=16, num_strides=1):

    '''

    conv>batch_norm>relu

    '''

    x = conv_layer(inputs, filters, num_strides)

    x = BatchNormalization(axis=-1)(x)

    x = Activation('relu')(x)

    return x

    

def resnet_block(inputs, filters=16):

    x_shortcut = inputs

    x = conv_block(inputs, filters)

    x = BatchNormalization(axis=-1)(x)

    x = Add()([x,x_shortcut]) # skip connection

    x = Activation('relu')(x)

    return x

    



inputs = Input(shape=(IMG_HEIGHT,IMG_WIDTH,1))



output_0 = conv_block(inputs=inputs, filters=16)

output_0 = MaxPool2D(pool_size=(2,2), strides=(2,2))(output_0)



output_1 = conv_block(output_0, filters=32)

output_1 = resnet_block(output_1, filters=32)

output_1 = MaxPool2D(pool_size=(2,2), strides=(2,2))(output_1)



output_2 = conv_block(output_1, filters=64)

output_2 = resnet_block(output_2, filters=64)

output_2 = AveragePooling2D(pool_size=(2,2), strides=(2,2))(output_2)



output_3 = Flatten()(output_2)

output_3 = Dropout(0.5)(output_3)



output_4 = Dense(500)(output_3)

output_4 = Dropout(0.5)(output_4)

output_4 = BatchNormalization(axis=-1)(output_4)

output_4 = Activation('relu')(output_4)



output_5 = Dense(10, activation='softmax')(output_4)



res_model = Model(inputs=inputs, outputs=output_5, name="res_model")



res_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

res_model.summary()