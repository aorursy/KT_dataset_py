import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

from PIL import Image

from tqdm.notebook import tqdm

# from tqdm import  tqdm
input_path = '../input/1056lab-defect-detection-extra/'

_train_path = os.path.join(input_path, 'train').replace("\\", os.sep)

_test_path = os.path.join(input_path, 'test')

train_fold_list = sorted(os.listdir(_train_path))

test_file_list = sorted(os.listdir(_test_path))
train_fold_list
test_file_list[: 10]
from PIL import ImageOps



def img_preprocess(img, resize=None, h_flip=False, v_flip=False, angle=0, noise=False):

    if resize is not None:

        img = img.resize(resize)

    

    if h_flip:

        img = ImageOps.flip(img)

        

    if v_flip:

        img = ImageOps.mirror(img)

        

    img = img.rotate(angle)

        

    return img
def load_images(fold_name, fold_path=None, train_mode=True, img_size=(256, 256), h_flip=False, v_flip=False, verbose=True,

               rate=1, angle=0):

    images = [] # 画像データを格納する配列

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

        img = img_preprocess(img, resize=img_size, h_flip=h_flip, v_flip=v_flip, angle=angle)

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

import time

import joblib



try:

    train_images = joblib.load("./train_images.pkl")

    labels = joblib.load("./label.pkl")

    print("Loaded Local dump files.")



except:

    train_images = []

    labels = []

    img_size = (128, 128)

    n_aug = 5



    for fold in train_fold_list:

        print('{} Loading...'.format(fold))

        time.sleep(0.3)

        

        imgs, label = load_images(fold, _train_path, img_size=img_size, verbose=True, rate=rate)

        train_images.extend(imgs)

        labels.extend(label)

        

        # h_flip = True

        imgs, label = load_images(fold, _train_path, img_size=img_size, h_flip=True, verbose=False, rate=rate)

        train_images.extend(imgs)

        labels.extend(label)



        # v_flip = True

        imgs, label = load_images(fold, _train_path, img_size=img_size, v_flip=True, verbose=False, rate=rate)

        train_images.extend(imgs)

        labels.extend(label)

        

        # h_flip, v_flip = True

        imgs, label = load_images(fold, _train_path, img_size=img_size, h_flip=True, v_flip=True, verbose=False, rate=rate)

        train_images.extend(imgs)

        labels.extend(label)

        

        # random data augment

        for _ in range(n_aug):

            hflip = True if np.random.randint(1, 3) // 2 == 0 else False

            vflip = True if np.random.randint(1, 3) // 2 == 0 else False

            angle = np.random.randint(-90, 91)

            

            imgs, label = load_images(fold, _train_path, img_size=img_size, h_flip=True, v_flip=True, verbose=False, rate=rate, angle=angle)

            train_images.extend(imgs)

            labels.extend(label)



        del imgs, label



#     gc.collect()

#     joblib.dump(train_images, "./train_images.pkl")

#     joblib.dump(labels, "./label.pkl")
len(labels)
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
del train_images, test_images

gc.collect()
from tensorflow.keras.utils import to_categorical



y = to_categorical(labels)
y[:, 1].mean()
X = np.array(X)

X_test = np.array(X_test)
p = np.random.permutation(len(X)).astype(int)

X = X[p]

y = y[p]
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dropout, Dense, BatchNormalization, Flatten, MaxPooling2D

from tensorflow.keras.models import Sequential
def dnn_model(input_shape=(512, 512, 1), out_shape=1):

    model = Sequential()

    

    # 入力/中間層

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

#     model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))

    model.add(MaxPooling2D(3, 3))

    model.add(Dropout(0.2))

    model.add(BatchNormalization())

    

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(3, 3))

    model.add(Dropout(0.2))

    model.add(BatchNormalization())

    

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D(3, 3))

    model.add(Dropout(0.2))

    model.add(BatchNormalization())

    

#     model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))

#     model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))

#     model.add(MaxPooling2D(3, 3))

#     model.add(Dropout(0.2))

#     model.add(BatchNormalization())

    

    # 出力層

    model.add(Flatten())

    model.add(Dense(units=10, activation='relu'))

    model.add(Dense(units=out_shape, activation='softmax'))

    

    # コンパイル

    model.compile(loss='categorical_crossentropy',

#     model.compile(loss='binary_crossentropy',

                 optimizer='adam', metrics=[AUC])

    

    return model
from tensorflow.keras.layers import add, Activation, Input, GlobalAveragePooling2D

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

    model.compile(optimizer='adam', loss='categorical_crossentropy',

                 metrics=["AUC"])

    

    return model
from tensorflow.keras.layers import (Input, Dense, Conv2D, BatchNormalization, Activation, 

                                     MaxPooling2D, GlobalAveragePooling2D, add)

from tensorflow.keras.models import Model



def _resnextblock(n_filters1, n_filters2, strides=(1,1)):

    def f(input):    

        x = Conv2D(n_filters1, (1,1), strides=strides,

                                          kernel_initializer='he_normal', padding='same')(input)

        x = BatchNormalization()(x)

        x = Activation('relu')(x)

        x = Conv2D(n_filters1, (3,3), strides=strides,

                                          kernel_initializer='he_normal', padding='same')(x)

        x = BatchNormalization()(x)

        x = Activation('relu')(x)

        x = Conv2D(n_filters2, (1,1), strides=strides,

                                          kernel_initializer='he_normal', padding='same')(x)

        x = BatchNormalization()(x)



        return x



    return f



def resnext(input_shape=(256, 256, 1), out_shape=2):

    inputs = Input(shape=input_shape)

    x = Conv2D(32, (7,7), strides=(1,1),

                    kernel_initializer='he_normal', padding='same')(inputs)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = MaxPooling2D((3, 3), strides=(2,2), padding='same')(x)



    residual = Conv2D(256, 1, strides=1, padding='same')(x)

    residual = BatchNormalization()(residual)



    x1 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x2 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x3 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x4 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x5 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x6 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x7 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x8 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x9 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x10 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x11 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x12 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x13 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x14 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x15 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x16 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x17 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x18 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x19 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x20 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x21 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x22 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x23 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x24 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x25 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x26 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x27 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x28 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x29 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x30 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x31 = _resnextblock(n_filters1=4, n_filters2=256)(x)

    x32 = _resnextblock(n_filters1=4, n_filters2=256)(x)





    x_all = add([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,

               x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,

               x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,

               x31,x32])





    x = add([x_all, residual])



    x = MaxPooling2D((3, 3), strides=(2,2), padding='same')(x)

    residual = Conv2D(512, 1, strides=1, padding='same')(x)

    residual = BatchNormalization()(residual)



    x1 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x2 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x3 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x4 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x5 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x6 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x7 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x8 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x9 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x10 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x11 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x12 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x13 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x14 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x15 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x16 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x17 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x18 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x19 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x20 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x21 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x22 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x23 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x24 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x25 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x26 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x27 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x28 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x29 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x30 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x31 = _resnextblock(n_filters1=8, n_filters2=512)(x)

    x32 = _resnextblock(n_filters1=8, n_filters2=512)(x)





    x_all = add([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,

               x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,

               x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,

               x31,x32])



    x = add([x_all, residual])



    x = GlobalAveragePooling2D()(x)

    x = Dense(out_shape, kernel_initializer='he_normal', activation='softmax')(x)





    model = Model(inputs=inputs, outputs=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy',

                 metrics=["AUC"])

    return model
from tensorflow.keras.applications import ResNet50

from tensorflow.keras.layers import Input, Flatten, Dense, Dropout

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.optimizers import Adam, SGD



def resnet50(input_shape=(256, 256, 1), out_shape=1):

#     input_shape = reversed(input_shape)

    

    input_tensor = Input(shape=input_shape)

    resnet_50 = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)

    

    # FC層

    top_model = Sequential()

    top_model.add(Flatten(input_shape=resnet_50.output_shape[1:]))

    top_model.add(Dense(units=256, activation="relu"))

    top_model.add(Dropout(0.5))

    top_model.add(Dense(units=out_shape, activation="softmax"))

    

    model = Model(input=resnet_50.input, output=top_model(resnet_50.output))

    

    loss = "binary_crossentropy" if out_shape <= 2 else "categorical_crossentropy"

    model.compile(loss=loss, optimizer=Adam(0.01), metrics=["acc"])

    

    return model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense

from tensorflow.keras.models import Model

from tensorflow.keras.applications import DenseNet121



def densenet(input_shape=(256, 256, 1), out_shape=1):

    input_tensor = Input(shape=input_shape)

    base_model = DenseNet121(include_top=False, # 出力層にするかどうか

                             weights=None,

                             input_tensor=input_tensor

                            )

    x = GlobalAveragePooling2D()(base_model.output)

    x = Dense(1024, activation="relu")(x)

    

    x = Dense(out_shape, activation="softmax")(x)

    model = Model(input_tensor, x)

    

    loss = "binary_crossentropy" if out_shape <= 2 else "categorical_crossentropy"

    model.compile(loss=loss, optimizer="adam", metrics=["AUC"])

    

    return model
# input_shape = (X.shape[1], X.shape[2], 1)

# out_shape = len(np.unique(labels))

# model = resnet50(input_shape=input_shape, out_shape=out_shape)



# from tensorflow.keras.callbacks import EarlyStopping



# es_cb = EarlyStopping(monitor='val_acc', patience=5, mode="max")

# # es_cb = EarlyStopping(monitor='val_acc', patience=5)



# model.fit(X, y, batch_size=32, epochs=100, validation_split=0.2, callbacks=[es_cb], verbose=2)

# # model.fit(X, y_1d, batch_size=32, epochs=100, validation_split=0.2, callbacks=[es_cb])

# # model.fit_generator(gene.flow(X, y, batch_size=32), steps_per_epoch=100, epochs=10, verbose=2,

# #                     validation_data=val_gene.flow(X, y, batch_size=32),validation_steps=100)



# predict = model.predict(X_test)



# predict[:, 1].mean()



# df_submit = pd.read_csv(os.path.join(input_path, 'sampleSubmission.csv')).set_index('name')

# df_submit['defect'] = predict[:, 0]

# df_submit.to_csv('./resnet.csv')
input_shape = (X.shape[1], X.shape[2], 1)

out_shape = len(np.unique(labels))

model = resnet(input_shape=input_shape, out_shape=out_shape)



from tensorflow.keras.callbacks import EarlyStopping

es_cb = EarlyStopping(monitor='val_AUC', patience=25, mode="max")



model.fit(X, y, batch_size=32, epochs=1000, validation_split=0.2, callbacks=[es_cb], verbose=1)
predict = model.predict(X_test)

df_submit = pd.read_csv(os.path.join(input_path, 'sampleSubmission.csv')).set_index('name')

df_submit['defect'] = predict[:, 0]

df_submit.to_csv('./ResNet.csv')
# from tensorflow.keras.callbacks import EarlyStopping

# es_cb = EarlyStopping(monitor='val_AUC', patience=25, mode="max")



# input_shape = (X.shape[1], X.shape[2], 1)

# out_shape = len(np.unique(labels))

# myModel = dnn_model(input_shape=input_shape, out_shape=out_shape)



# myModel.fit(X, y, batch_size=32, epochs=5, validation_split=0.2, callbacks=[es_cb], verbose=1)
# predict = myModel.predict(X_test)

# df_submit = pd.read_csv(os.path.join(input_path, 'sampleSubmission.csv')).set_index('name')

# df_submit['defect'] = predict[:, 0]

# df_submit.to_csv('./my_model.csv')
# from tensorflow.keras.callbacks import EarlyStopping

# es_cb = EarlyStopping(monitor='val_AUC', patience=25, mode="max")



# input_shape = (X.shape[1], X.shape[2], 1)

# out_shape = len(np.unique(labels))

# resnext_model = resnext(input_shape=input_shape, out_shape=out_shape)



# resnext_model.fit(X, y, batch_size=32, epochs=1000, validation_split=0.2, callbacks=[es_cb], verbose=1)
# predict = resnext_model.predict(X_test)

# df_submit = pd.read_csv(os.path.join(input_path, 'sampleSubmission.csv')).set_index('name')

# df_submit['defect'] = predict[:, 0]

# df_submit.to_csv('./resnext.csv')
# input_shape = (X.shape[1], X.shape[2], 1)

# out_shape = len(np.unique(labels))

# dense_model = densenet(input_shape=input_shape, out_shape=out_shape)



# from tensorflow.keras.callbacks import EarlyStopping

# es_cb = EarlyStopping(monitor='val_AUC', patience=25, mode="max")



# dense_model.fit(X, y, batch_size=32, epochs=1000, validation_split=0.2, callbacks=[es_cb], verbose=1)
# predict = dense_model.predict(X_test)

# df_submit = pd.read_csv(os.path.join(input_path, 'sampleSubmission.csv')).set_index('name')

# df_submit['defect'] = predict[:, 0]

# df_submit.to_csv('./DenseNet.csv')