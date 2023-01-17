import pandas as pd

import numpy as np

import os

from PIL import Image

from tqdm.notebook import tqdm

import tensorflow as tf

import sys
print("Python version :", sys.version)

print("Pandas version :", pd.__version__)

print("NumPy version", np.__version__)

import PIL

print("Pillow version", PIL.__version__); del PIL

import tqdm as tqdm_v

print("tqdm version :", tqdm_v.__version__); del tqdm_v

print("TensorFlow version :", tf.__version__)
input_path = '../input/1056lab-defect-detection-extra/'

_train_path = os.path.join(input_path, 'train').replace("\\", os.sep)

_test_path = os.path.join(input_path, 'test')

train_fold_list = sorted(os.listdir(_train_path))

test_file_list = sorted(os.listdir(_test_path))
train_fold_list
test_file_list[: 10]
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

               rate=1, file_list=None):

    images = [] # 画像データを格納する配列

    if file_list is None:

        file_list = sorted(os.listdir(os.path.join(fold_path, fold_name))) # ファイルのリストを取得

    

    labels = [] # 正解ラベル

    if train_mode:

#         label = int(re.search(pattern='[0-9]', string=fold_name).group()) # フォルダ名からラベルを設定

        label = 0 if fold_name.endswith("_def") else 1

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

        

        for _ in range(rate - 1):

            if fold_name.endswith("_def"):

                img = Image.open(os.path.join(fold_path, fold_name, file))

                img = img.convert('L')

                h_flip_temp = np.random.randint(0, 1) // 2 == 0

                v_flip_temp = np.random.randint(0, 1) // 2 == 0

                angle_temp = np.random.randint(-90, 91)

                img = img_preprocess(img, resize=img_size, h_flip=h_flip_temp, v_flip=v_flip_temp, angle=angle_temp)

                images.append(img)

                labels.append(label)



    return images, labels
rate = round(len(os.listdir(os.path.join(_train_path, train_fold_list[0]))) / len(os.listdir(os.path.join(_train_path, train_fold_list[1]))))
import gc

import joblib



for i in range(1, 7):

    exec("train_images_{} = []".format(i))

    exec("labels_{} = []".format(i))

    img_size = (128, 128)



    for fold in train_fold_list:

        if str(i) in fold:

            print('{} Loading...'.format(fold))



            imgs, label = load_images(fold, _train_path, img_size=img_size, verbose=True, rate=rate)

            exec("train_images_{}.extend(imgs)".format(i))

            exec("labels_{}.extend(label)".format(i))



            # h_flip = True

            imgs, label = load_images(fold, _train_path, img_size=img_size, h_flip=True, verbose=False, rate=rate)

            exec("train_images_{}.extend(imgs)".format(i))

            exec("labels_{}.extend(label)".format(i))



            # v_flip = True

            imgs, label = load_images(fold, _train_path, img_size=img_size, v_flip=True, verbose=False, rate=rate)

            exec("train_images_{}.extend(imgs)".format(i))

            exec("labels_{}.extend(label)".format(i))



            # h_flip, v_flip = True

            imgs, label = load_images(fold, _train_path, img_size=img_size, h_flip=True, v_flip=True, verbose=False, rate=rate)

            exec("train_images_{}.extend(imgs)".format(i))

            exec("labels_{}.extend(label)".format(i))



            del imgs, label



    #     gc.collect()
df_test_class = pd.read_csv("../input/classification-testdata/TestData_Class.csv", index_col="name")
for i in range(1, 7):

    exec("test_files_{} = df_test_class[df_test_class['Class'] == {}].index".format(i, i))

    

    exec("test_images_{}, _ = load_images('test', input_path, train_mode=False, img_size=img_size, file_list=test_files_{})".format(i, i))
from tensorflow.keras.preprocessing.image import img_to_array



def imgs2arr(img_array):

    arr = []

    for data in tqdm(img_array):

        arr.append(img_to_array(data))

    arr = np.array(arr, dtype=np.float16) / 255.

    return arr
for i in range(1, 7):

    exec("X_{} = imgs2arr(train_images_{})".format(i, i))
for i in range(1, 7):

    exec("X_test_{} = imgs2arr(test_images_{})".format(i, i))
from tensorflow.keras.utils import to_categorical



for i in range(1, 7):

    exec("y_{} = to_categorical(labels_{})".format(i, i))
import tensorflow as tf

from tensorflow.keras.layers import (add, Activation, Input, 

                                     GlobalAveragePooling2D,

                                    Conv2D, Dropout, Dense,

                                    BatchNormalization, Flatten, MaxPooling2D)

from tensorflow.keras.models import Model, Sequential



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

    model.compile(optimizer='adam', loss='categorical_crossentropy',

                 metrics=["AUC"])

    

    return model
input_shape = (X_1.shape[1], X_1.shape[2], 1)

out_shape = len(np.unique(labels_1))



from tensorflow.keras.callbacks import EarlyStopping

es_cb = EarlyStopping(monitor='val_AUC', patience=30, mode="max")
for i in range(1, 7):

    exec("model_{} = resnet(input_shape=input_shape, out_shape=out_shape)".format(i))

    exec("model_{}.fit(X_{}, y_{}, batch_size=32, epochs=100, validation_split=0.2, callbacks=[es_cb])".format(i,i,i))
df_predict = df_test_class.copy()

df_predict["defect"] = 0

for i in range(1, 7):

    exec("df_predict.loc[df_predict['Class'] == {}, 'defect'] = model_{}.predict(X_test_{})[:, 0]".format(i, i, i))

df_predict.drop(["Class"], axis=1).to_csv("ResNet.csv")