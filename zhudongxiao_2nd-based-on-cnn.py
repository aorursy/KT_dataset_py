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

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPool2D, BatchNormalization, LeakyReLU

from keras.models import Sequential

import matplotlib.pyplot as plt



class Data_Clean(object):

    def __init__(self, trainPath, testPath):

        self.train = pd.read_csv(trainPath)

        self.test = pd.read_csv(testPath)

        self.data_assemble = []

        self.column = self.train.columns.drop('Image')



    def seperate_data(self):

        for column_name in self.train.columns.drop('Image'):

            data = self.train[[column_name, 'Image']].dropna()

            self.data_assemble.append([data[column_name], data['Image']])

        self.test = self.test['Image']



    def reshape_image(self):

        for data in self.data_assemble:

            data[1] = data[1].apply(lambda im: np.fromstring(im, sep=' ', dtype=float))

            values = data[1].values / 255.0

            values = np.vstack(values).reshape(-1, 96, 96, 1)

            data[1] = values

        self.test = self.test.apply(lambda img: np.fromstring(img, sep=' ', dtype=float))

        self.test = self.test.values / 255.0

        self.test = np.vstack(self.test).reshape(-1, 96, 96, 1)



    def run(self):

        self.seperate_data()

        self.reshape_image()

        return self.data_assemble, self.test, self.column



class CNN(object):

    def __init__(self, train, test, columns_list):

        self.train = train

        self.test = test

        self.columns_list = columns_list



    def define_CNN(self):

        model = Sequential()

        model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', use_bias=False, input_shape=(96, 96, 1)))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization())



        model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', use_bias=False))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization())

        model.add(MaxPool2D(pool_size=(2, 2)))



        model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', use_bias=False))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization())



        model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', use_bias=False))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization())

        model.add(MaxPool2D(pool_size=(2, 2)))



        model.add(Convolution2D(filters=96, kernel_size=(3, 3), padding='same', use_bias=False))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization())



        model.add(Convolution2D(filters=96, kernel_size=(3, 3), padding='same', use_bias=False))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization())

        model.add(MaxPool2D(pool_size=(2, 2)))



        model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same', use_bias=False))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization())



        model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same', use_bias=False))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization())

        model.add(MaxPool2D(pool_size=(2, 2)))



        model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization())



        model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same', use_bias=False))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization())

        model.add(MaxPool2D(pool_size=(2, 2)))



        model.add(Convolution2D(filters=512, kernel_size=(3, 3), padding='same', use_bias=False))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization())



        model.add(Convolution2D(filters=512, kernel_size=(3, 3), padding='same', use_bias=False))

        model.add(LeakyReLU(alpha=0.1))

        model.add(BatchNormalization())



        model.add(Flatten())

        model.add(Dense(512, activation='relu'))

        model.add(Dropout(0.1))

        model.add(Dense(1))

        self.model = model



    def Adam(self, epochs, batchSize, xtrain, xvalidation, ytrain, yvalidation):

        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        self.model.fit(xtrain, ytrain, batch_size=batchSize, epochs=epochs, validation_data=(xvalidation, yvalidation),

                       verbose=0)



    def show_a_result(self, ypred, ytest):

        import copy

        plt.imshow(ytest.reshape(96, 96))

        ypred = copy.deepcopy(ypred)

        xy = np.split(ypred, 30)

        predx, predy = [], []

        for i in range(0, 30, 2):

            predx.append(xy[i])

            predy.append(xy[i+1])

        plt.plot(predx, predy, 'o', color='red')

        plt.show()



    def make_output(self, ypred):

        pred = ypred

        dataPath = '../input/IdLookupTable.csv'

        lookid_data = pd.read_csv(dataPath)

        lookid_list = list(lookid_data['FeatureName'])

        imageID = list(lookid_data['ImageId'] - 1)

        rowid = lookid_data['RowId']

        rowid = list(rowid)

        feature = []

        for f in lookid_list:

            feature.append(f)

        preded = []

        for x, y in zip(imageID, feature):

            preded.append(pred[y].loc[x])

        rowid = pd.Series(rowid, name='RowId')

        loc = pd.Series(preded, name='Location')

        submission = pd.concat([rowid, loc], axis=1)

        submission.to_csv('Utkarsh.csv', index=False)





    def run(self):

        ypred = pd.DataFrame(index = [i for i in range(1783)] ,columns=self.columns_list)

        for index, data in enumerate(self.train):

            label = data[0]

            columns_name = self.columns_list[index]

            train = data[1]

            xtrain, xvalidation, ytrain, yvalidation = train_test_split(train, label, test_size=0.1, random_state=9)

            self.define_CNN()

            print(columns_name, ' training started:')

            self.Adam(epochs=100, batchSize=128, xtrain=xtrain, xvalidation=xvalidation, ytrain=ytrain, yvalidation=yvalidation)

            ypred[columns_name] = self.model.predict(self.test)



        self.make_output(ypred)

        #show a results

        self.show_a_result(ypred.loc[159], self.test[159])



if __name__ == '__main__':

    trainPath = '../input/training/training.csv'

    testPath = '../input/test/test.csv'



    data_clean = Data_Clean(trainPath, testPath)

    train, test, columns_list = data_clean.run()



    cnn = CNN(train, test, columns_list)

    cnn.run()
