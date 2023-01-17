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

['images', 'sample_submission.csv', 'train.csv', 'test.csv']

#CNN for classifier



import os

import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split



class Data_Clean(object):

    def __init__(self):

        self.numerical_data, self.num_test_data = self.read_numerical_data()

        self.id, self.species, self.num_train, self.test_id, self.test_num = self.split_numerical_data()

        self.image_data = self.read_image_data()

        self.image_test_data = self.read_image_test_data()



    def split_numerical_data(self):

        #将numerical data划分为id, species,和训练集

        id = self.numerical_data.pop('id')

        species = self.numerical_data.pop('species')

        species = LabelEncoder().fit(species).transform(species)

        species = to_categorical(species, num_classes=99)

        self.numerical_data = StandardScaler().fit(self.numerical_data).transform(self.numerical_data)



        test_id = self.num_test_data.pop('id')

        num_test_data = StandardScaler().fit(self.num_test_data).transform(self.num_test_data)

        return id, species, self.numerical_data, test_id, num_test_data



    def read_numerical_data(self):

        root = "../input"

        data = pd.read_csv('%s/train.csv' %root)

        test_data = pd.read_csv('%s/test.csv' %(root))

        return data, test_data



    def read_image_test_data(self):

        #读取图片，将图片等比例缩放到96*96，最长边为96， 图片放置到中心位置

        max_dim = 96

        #设定一个初始的output,全白背景

        X = np.empty((len(self.test_id), max_dim, max_dim, 1))

        #开始读图

        root = "../input"

        for index, id in enumerate(self.test_id):

            x = self.resize_img(load_img(os.path.join(root, 'images', str(id) + '.jpg'), grayscale=True), max_dim = max_dim)

            #将图片格式转化为矩阵

            x = img_to_array(x)

            #获得缩放后图片的长和宽

            length = x.shape[0]

            width = x.shape[1]

            #将图片放置到X的中心位置

            h1 = int((max_dim - length) / 2)

            h2 = h1 + length

            w1 = int((max_dim - width) / 2)

            w2 = w1 + width

            #放置到X中

            X[index, h1:h2, w1:w2, 0:1] = x

        return np.around(X / 255.0)



    def read_image_data(self):

        #读取图片，将图片等比例缩放到96*96，最长边为96， 图片放置到中心位置

        max_dim = 96

        #设定一个初始的output,全白背景

        X = np.empty((len(self.id), max_dim, max_dim, 1))

        #开始读图

        root = "../input"

        for index, id in enumerate(self.id):

            x = self.resize_img(load_img(os.path.join(root, 'images', str(id) + '.jpg'), grayscale=True), max_dim = max_dim)

            #将图片格式转化为矩阵

            x = img_to_array(x)

            #获得缩放后图片的长和宽

            length = x.shape[0]

            width = x.shape[1]

            #将图片放置到X的中心位置

            h1 = int((max_dim - length) / 2)

            h2 = h1 + length

            w1 = int((max_dim - width) / 2)

            w2 = w1 + width

            #放置到X中

            X[index, h1:h2, w1:w2, 0:1] = x

        return np.around(X / 255.0)



    def resize_img(self, image, max_dim):

        #对图片进行缩放

        max_length = max(image.size[0], image.size[1])

        #确定缩放比例

        scale = max_dim / max_length

        return image.resize((int(image.size[0] * scale), int(image.size[1] * scale)))



    def run(self):

        ##Check the data

        # print('id:', self.id.loc[0])

        # print('species:', self.species[0])

        # print('feature:', self.num_train[0])

        # plt.imshow(self.image_data[0].reshape(960, 960), cmap='gray')

        # plt.show()

        return self.species, self.num_train, self.image_data, self.test_num, self.image_test_data, self.test_id



from keras.models import Model

from keras.layers import Convolution2D, LeakyReLU, BatchNormalization, Dense, MaxPool2D, Flatten, Dropout, Input, merge

from keras.optimizers import RMSprop

from keras.callbacks import ReduceLROnPlateau



class CNN(object):

    def __init__(self, image_train, image_vali, num_train, num_vali, species_train, species_vali, num_test, image_test, test_id):

        self.image_train = image_train

        self.image_vali = image_vali

        self.num_train = num_train

        self.num_vali = num_vali

        self.species_train = species_train

        self.species_vali = species_vali

        self.num_test = num_test

        self.image_test = image_test

        self.test_id = test_id



    def define_CNN(self):

        image = Input(shape=(96, 96, 1), name='image')

        x = Convolution2D(filters=32, kernel_size=(3,3), padding='same', use_bias=False)(image)

        x = LeakyReLU(alpha=0.1)(x)

        x = BatchNormalization()(x)



        x = Convolution2D(filters=32, kernel_size=(3,3), padding='same', use_bias=False)(x)

        x = LeakyReLU(alpha=0.1)(x)

        x = BatchNormalization()(x)

        x = MaxPool2D(pool_size=(2,2))(x)



        x = Convolution2D(filters=64, kernel_size=(3, 3), padding='same', use_bias=False)(x)

        x = LeakyReLU(alpha=0.1)(x)

        x = BatchNormalization()(x)



        x = Convolution2D(filters=64, kernel_size=(3, 3), padding='same', use_bias=False)(x)

        x = LeakyReLU(alpha=0.1)(x)

        x = BatchNormalization()(x)

        x = MaxPool2D(pool_size=(2, 2))(x)



        x = Convolution2D(filters=96, kernel_size=(3, 3), padding='same', use_bias=False)(x)

        x = LeakyReLU(alpha=0.1)(x)

        x = BatchNormalization()(x)



        x = Convolution2D(filters=96, kernel_size=(3, 3), padding='same', use_bias=False)(x)

        x = LeakyReLU(alpha=0.1)(x)

        x = BatchNormalization()(x)

        x = MaxPool2D(pool_size=(2, 2))(x)



        x = Convolution2D(filters=128, kernel_size=(3, 3), padding='same', use_bias=False)(x)

        x = LeakyReLU(alpha=0.1)(x)

        x = BatchNormalization()(x)



        x = Convolution2D(filters=128, kernel_size=(3, 3), padding='same', use_bias=False)(x)

        x = LeakyReLU(alpha=0.1)(x)

        x = BatchNormalization()(x)

        x = MaxPool2D(pool_size=(2, 2))(x)



        x = Convolution2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False)(x)

        x = LeakyReLU(alpha=0.1)(x)

        x = BatchNormalization()(x)



        x = Convolution2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False)(x)

        x = LeakyReLU(alpha=0.1)(x)

        x = BatchNormalization()(x)

        x = MaxPool2D(pool_size=(2, 2))(x)



        x = Convolution2D(filters=512, kernel_size=(3, 3), padding='same', use_bias=False)(x)

        x = LeakyReLU(alpha=0.1)(x)

        x = BatchNormalization()(x)



        x = Convolution2D(filters=512, kernel_size=(3, 3), padding='same', use_bias=False)(x)

        x = LeakyReLU(alpha=0.1)(x)

        x = BatchNormalization()(x)

        x = MaxPool2D(pool_size=(2, 2))(x)



        x = Flatten()(x)



        numerical = Input(shape=(192,), name='numerical')

        concatenated = merge.concatenate([x, numerical])



        x = Dense(512, activation='relu')(concatenated)

        x = Dropout(0.1)(x)

        x = Dense(99, activation='softmax')(x)

        self.model = Model(inputs=[image, numerical], outputs=x)



    def RMSprop(self, batch_size, epoch):

        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)

        self.model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])

        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

        self.model.fit([self.image_train, self.num_train], self.species_train, batch_size=batch_size, epochs=epoch,

                       validation_data=([self.image_vali, self.num_vali], self.species_vali), verbose=0, callbacks=[learning_rate_reduction])



    def make_predict(self):

        ypred_prob = self.model.predict([self.image_test, self.num_test])

        root = "../input"

        labels = sorted(pd.read_csv(os.path.join(root, 'train.csv')).species.unique())

        ypred = pd.DataFrame(ypred_prob, index = self.test_id, columns=labels)

        fp = open('submit.csv', 'w')

        fp.write(ypred.to_csv())

        print('Done!')



    def run(self):

        self.define_CNN()

        self.RMSprop(batch_size = 128, epoch=1600)

        self.make_predict()





if __name__ == '__main__':

    data_clean = Data_Clean()

    species, num, image, num_test, image_test, test_id = data_clean.run()

    #划分训练集和测试集

    species_train, species_vali, num_train, num_vali, image_train, image_vali = train_test_split(species, num, image, test_size=0.15, random_state=93)



    #搭建CNN模型，看看CNN对图像的分类情况

    cnn = CNN(image_train, image_vali, num_train, num_vali,species_train, species_vali, num_test, image_test, test_id)

    cnn.run()