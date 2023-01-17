import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

from PIL import Image

from tqdm.notebook import tqdm
import tensorflow as tf

from keras import backend as K

from sklearn.metrics import roc_auc_score

def AUC(y_true, y_pred):

    return tf.numpy_function(roc_auc_score, (y_true, y_pred), tf.double)
input_path = '../input/1056lab-defect-detection-extra/'

_train_path = os.path.join(input_path, 'train').replace("\\", os.sep)

_test_path = os.path.join(input_path, 'test')

train_fold_list = sorted(os.listdir(_train_path))

test_file_list = sorted(os.listdir(_test_path))
from PIL import ImageOps



def img_preprocess(img, resize=None, h_flip=False, v_flip=False, angle=0):

    if resize is not None:

        img = img.resize(resize)

    

    if h_flip:

        img = ImageOps.flip(img)

        

    if v_flip:

        img = ImageOps.mirror(img)

        

    img = img.rotate(angle)

        

    return img
import re



def load_images(fold_name, fold_path=None, train_mode=True, img_size=(256, 256), h_flip=False, v_flip=False, verbose=True,

               rate=1, defect=True):

    images = [] # 画像データを格納する配列

    file_list = sorted(os.listdir(os.path.join(fold_path, fold_name))) # ファイルのリストを取得

    

    labels = [] # 正解ラベル

    if train_mode:

        if defect:

            label = 0 if fold_name.endswith("_def") else 1

        else:

            label = int(re.search(pattern='[0-9]', string=fold_name).group()) # フォルダ名からラベルを設定

        labels = [label for _ in range(len(file_list))]

    

    if verbose:

        f_list = tqdm(file_list)

    else:

        f_list = file_list

    

    for file in f_list:

        img = Image.open(os.path.join(fold_path, fold_name, file))

        img = img.convert('L')

        img = img_preprocess(img, resize=img_size, h_flip=h_flip, v_flip=v_flip)

        images.append(img)

        

#         for _ in range(rate - 1):

#             if fold_name.endswith("_def"):

#                 img = Image.open(os.path.join(fold_path, fold_name, file))

#                 img = img.convert('L')

#                 h_flip_temp = np.random.randint(0, 1) // 2 == 0

#                 v_flip_temp = np.random.randint(0, 1) // 2 == 0

#                 angle_temp = np.random.randint(-90, 91)

#                 img = img_preprocess(img, resize=img_size, h_flip=h_flip_temp, v_flip=v_flip_temp, angle=angle_temp)

#                 images.append(img)

#                 labels.append(label)



    return images, labels
rate = round(len(os.listdir(os.path.join(_train_path, train_fold_list[0]))) / len(os.listdir(os.path.join(_train_path, train_fold_list[1]))))
train_images = []

labels = []

img_size = (128, 128)



defect = False



for fold in train_fold_list:

    print('{} Loading...'.format(fold))



    imgs, label = load_images(fold, _train_path, img_size=img_size, verbose=True, rate=rate, defect=defect)

    train_images.extend(imgs)

    labels.extend(label)



    # h_flip = True

    imgs, label = load_images(fold, _train_path, img_size=img_size, h_flip=True, verbose=False, rate=rate, defect=defect)

    train_images.extend(imgs)

    labels.extend(label)



    # v_flip = True

    imgs, label = load_images(fold, _train_path, img_size=img_size, v_flip=True, verbose=False, rate=rate, defect=defect)

    train_images.extend(imgs)

    labels.extend(label)



    # h_flip, v_flip = True

    imgs, label = load_images(fold, _train_path, img_size=img_size, h_flip=True, v_flip=True, verbose=False, rate=rate, defect=defect)

    train_images.extend(imgs)

    labels.extend(label)



    del imgs, label

test_images, _ = load_images('test', input_path, train_mode=False, img_size=img_size)
from tensorflow.keras.preprocessing.image import img_to_array



def imgs2arr(img_array):

    arr = []

    for data in tqdm(img_array):

        arr.append(img_to_array(data))

    arr = np.array(arr, dtype=np.float16) / 255.

    return arr
X = imgs2arr(train_images)
X_test = imgs2arr(test_images)
import gc

del train_images, test_images

gc.collect()
from tensorflow.keras.utils import to_categorical



# y_1d = np.array(labels).astype(int)

y = to_categorical(labels)[:, 1:]
p = np.random.permutation(len(X)).astype(int)

X = X[p]

y = y[p]
from tensorflow.keras.layers import (add, Activation, Input, BatchNormalization, MaxPooling2D,

                                     GlobalAveragePooling2D, Conv2D, Dense)

from tensorflow.keras.models import Model



def _shortcut(inputs, residual):

#     print(residual.shape)

    n_filters = residual.shape[3]

    

    shortcut = Conv2D(filters=n_filters, kernel_size=(1, 1),

                     padding='valid')(inputs)

    return add([shortcut, residual])



def _resblock(n_filters, strides=(1, 1)):

    def f(inputs):

        x = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=strides,

                  kernel_initializer='he_normal', padding='same')(inputs)

        x = BatchNormalization()(x)

        x = Activation('relu')(x)

        

        x = Conv2D(filters=n_filters, kernel_size=(3, 3), strides=strides,

                  kernel_initializer='he_normal', padding='same')(x)

        x = BatchNormalization()(x)

        

#         print(type(inputs), type(x))

        return _shortcut(inputs, x)

    return f



def resnet(input_shape=(256, 256, 1), out_shape=6):

    inputs = Input(shape=input_shape)

    x = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1),

              kernel_initializer='he_normal', padding='same')(inputs)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = MaxPooling2D((3, 3), strides=(2,2), padding='same')(x)



    x = _resblock(n_filters=64)(x)

    x = _resblock(n_filters=64)(x)

    x = _resblock(n_filters=64)(x)

    x = MaxPooling2D(strides=(2,2))(x)  

    x = _resblock(n_filters=128)(x)

    x = _resblock(n_filters=128)(x)

    x = _resblock(n_filters=128)(x)





    x = GlobalAveragePooling2D()(x)

    x = Dense(units=out_shape, kernel_initializer='he_normal', activation='softmax')(x)

    

    model = Model(inputs=inputs, outputs=x)

    model.compile(optimizer='SGD', loss='categorical_crossentropy',

                 metrics=["acc"])

    

    return model
input_shape = (X.shape[1], X.shape[2], 1)

out_shape = len(np.unique(labels))

model = resnet(input_shape=input_shape, out_shape=out_shape)
from tensorflow.keras.callbacks import EarlyStopping

es_cb = EarlyStopping(monitor='val_acc', patience=5, mode="max")



model.fit(X, y, batch_size=32, epochs=5, validation_split=0.2, callbacks=[es_cb], verbose=2)
p = model.predict(X_test)
(np.argmax(p, axis=1) + 1)
df_submit = pd.read_csv(os.path.join(input_path, 'sampleSubmission.csv')).set_index('name')

df_submit["Class"] = np.argmax(p, axis=1) + 1

del df_submit["defect"]

df_submit.to_csv("./TestData_Class.csv")