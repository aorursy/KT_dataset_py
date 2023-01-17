# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os

import sys

import numpy as np

import pandas as pd

import zipfile

import keras

from PIL import Image

from keras_applications.mobilenet_v2 import MobileNetV2

from keras.preprocessing.image import array_to_img, img_to_array, load_img

from keras.models import Sequential

from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tqdm import tqdm

from sklearn.model_selection import train_test_split
print(os.listdir("../input/train/train")[0])

path = os.path.join('../input/train/train', os.listdir("../input/train/train")[0])

print(path)

print(load_img(path))
class DataHandler():

    def __init__(self, csv_name):

        self.dataset_path = '../input'

        print(os.listdir(self.dataset_path))

        self.train_data = pd.read_csv(os.path.join(self.dataset_path, csv_name))

        # zipfiles = ['test.zip', 'train.zip']

        # for datas in zipfiles:

        #     target_dir = os.path.join(self.dataset_path, datas.split('.')[0])

        #     os.makedirs(target_dir, exist_ok=True)

        #     with zipfile.ZipFile(os.path.join(self.dataset_path, datas)) as f:

        #         f.extractall(target_dir)



    def get_from_columns(self, *args):

        for column in args:

            yield self.train_data[column]



    def get_only_data(self, folname):

        fol_path = os.path.join(self.dataset_path, folname)

        img_paths = list(map(lambda fname: os.path.join(fname), os.listdir(fol_path)))

        return img_paths



    def get_data_label(self, folname='train/train', val_rate=0.2):

        fol_path = os.path.join(self.dataset_path, folname)

        params = self.train_data.columns.values

        datas = []

        labels = []

        if folname.split('/')[0] == 'train':

            for idx ,row in tqdm(self.train_data.iterrows()):

                for data_or_label, param in enumerate(params):

                    if data_or_label == 0:

                        img_name = row[param]

                        img_data = img_to_array(load_img(os.path.join(fol_path, img_name)))

                        datas.append(img_data)

                    else:

                        labels.append(float(row[param]))

        else:

            img_names = os.listdir(fol_path)

            self.test_names = img_names

            for img_name in tqdm(img_names):

                img_data = img_to_array(load_img(os.path.join(fol_path, img_name)))

                datas.append(img_data)

            datas = np.asarray(datas).astype(np.float)

            datas /= 255

            print(len(datas))

            return datas



        datas = np.asarray(datas).astype(np.float)

        datas /= 255

        labels = np.asarray(labels).astype(np.float)

        datas, val_datas, labels, val_labels = train_test_split(datas, labels, test_size=val_rate)

        return (datas, labels, val_datas, val_labels)



    def get_shape(self, folname='train/train'):

        from random import randint

        fol_path = os.path.join(self.dataset_path, folname)

        sample_img_path = os.path.join(fol_path ,os.listdir(fol_path)[randint(0,20)])

        img_bin = img_to_array(load_img(sample_img_path)) # return ndarray

        return img_bin.shape
class Train():

    def __init__(self, use_imgnt=False):

        pass



    def build_model(self, img_shape, do_cb=True):

        self.model = Sequential()

        self.model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=img_shape))

        self.model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu'))

        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Dropout(0.1))

        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))

        self.model.add(Conv2D(filters=64, kernel_size=(5, 3), padding='same', activation='relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Dropout(0.1))

        self.model.add(Flatten())

        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(

            optimizer=keras.optimizers.Adam(),

            loss='binary_crossentropy',

            metrics=['accuracy']

        )

        if do_cb == True:

            self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1)

            self.early_stopping = EarlyStopping(monitor='val_loss', verbose=1, mode='auto')

        return self.model.summary



    def train(self, x_train, y_train, x_test, y_test, epochs=20, batch_size=32):

        hist = self.model.fit(

            x=x_train, y=y_train,

            # steps_per_epoch=x_train.shape[0] // batch_size,

            # validation_steps=x_test.shape[0] // batch_size,

            epochs=epochs,

            batch_size=batch_size,

            verbose=1,

            validation_data=(x_test, y_test),

        )

        return hist

    

    def predict(self, test):

        pred = self.model.predict(

            test

        )

        return pred
data_handler = DataHandler('train.csv')

datagen = data_handler.get_from_columns('id', 'has_cactus')

img_shape = data_handler.get_shape()

test = data_handler.get_data_label('test/test')

test_names = data_handler.test_names

print(len(test), len(test_names))

x_train, y_train, x_test, y_test = data_handler.get_data_label('train/train')

model = Train()

summary = model.build_model(img_shape=img_shape)

summary()

hist = model.train(x_train, y_train, x_test, y_test)
predict = model.predict(test)
submission_df = pd.DataFrame(predict, columns=['has_cactus'])

submission_df['id'] = ''

cols = submission_df.columns.tolist()

cols = cols[-1:] + cols[:-1]

submission_df = submission_df[cols]

for i, name in enumerate(test_names):

    submission_df.set_value(i, 'id', name)

submission_df.head(10)
submission_df.to_csv('submission.csv', index=False)