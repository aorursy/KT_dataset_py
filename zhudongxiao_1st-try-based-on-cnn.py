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
#原始数据将image的数据变成96*96的形状，缺失数据删除，用DNN模型

#将Image缩进[0, 1]范围内

#采用CNN，参数为这个Case种，Best Score的参数, 不适用BatchNormalization

#Convolution2D改为Conv2D, LeakyRelu改进Conv2D里面

#使用adam算法



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPool2D, BatchNormalization, LeakyReLU

from keras.models import Sequential

from keras.optimizers import RMSprop

from keras.callbacks import ReduceLROnPlateau



class Data_Clean(object):

    def __init__(self, trainPath, testPath):

        self.train = pd.read_csv(trainPath)

        self.test = pd.read_csv(testPath)



    def check_null(self):

        print(self.train.isnull().any().value_counts())

        print(len(self.train['left_eye_center_x']))



    def drop_na_row(self):

        self.train = self.train.dropna()



    def reshape_image(self):

        self.train = self.train.apply(lambda im: np.fromstring(im, sep=' ', dtype=float))

        # scaler the data to [0, 1]

        self.train = self.train.values / 255.0

        self.train = np.vstack(self.train).reshape(-1, 96, 96, 1)

        self.test = self.test.apply(lambda img: np.fromstring(img, sep=' ', dtype=float))

        # scaler the data to [0, 1]

        self.test = self.test.values / 255.0

        self.test = np.vstack(self.test).reshape(-1, 96, 96, 1)



    def split_train_label(self):

        self.label = self.train.drop(['Image'], axis=1)

        y_label = []

        for i in range(len(self.label)):

            y_val = self.label.iloc[i, :]

            y_label.append(y_val)

        self.label = np.array(y_label, dtype='float')

        self.train = self.train['Image']

        self.test = self.test['Image']



    def show_a_picture(self):

        feature = np.split(self.label[9], 30)

        x, y = [], []

        for i in range(0, 30, 2):

            x.append(feature[i])

            y.append(feature[i+1])

        plt.imshow(self.train[9].reshape(96, 96))

        #plt.imshow(self.train[9].values)

        plt.plot(x, y, 'o', color='red')

        plt.show()



    def run(self):

        # Delete the Null data

        self.drop_na_row()

        # Split the train and label

        self.split_train_label()

        #reshape

        self.reshape_image()

        # up to here, we finished cleaning our data

        # show a picture to see if it is okay

        #self.show_a_picture()

        return self.train, self.label, self.test



class CNN(object):

    def __init__(self, xtrain, xtest, ytrain, ytest, test):

        self.xtrain = xtrain

        self.xtest = xtest

        self.ytrain = ytrain

        self.ytest = ytest

        self.test = test



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

        model.add(Dense(30))

        self.model = model



    def Adam(self, epochs, batchSize):

        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        self.model.fit(self.xtrain, self.ytrain, batch_size=batchSize, epochs=epochs, validation_data=(self.xtest, self.ytest),

                       verbose=2)



    def prediction(self):

        pred = self.model.predict(self.test)

        lookid_data = pd.read_csv('../input/IdLookupTable.csv')

        lookid_list = list(lookid_data['FeatureName'])

        imageID = list(lookid_data['ImageId'] - 1)

        pred_list = list(pred)

        rowid = lookid_data['RowId']

        rowid = list(rowid)

        feature = []

        for f in list(lookid_data['FeatureName']):

            feature.append(lookid_list.index(f))

    

        preded = []

        for x, y in zip(imageID, feature):

            preded.append(pred_list[x][y])

        rowid = pd.Series(rowid, name='RowId')

        loc = pd.Series(preded, name='Location')

        submission = pd.concat([rowid, loc], axis=1)

        submission.to_csv('Utkarsh.csv', index=False)



    def show_any_result(self):

        self.pred = self.model.predict(self.xtest)

        import copy

        xtest, ypred, ytest = self.xtest[37], copy.deepcopy(self.pred[37]), copy.deepcopy(self.ytest[37])

        predx, predy, testx, testy = [], [], [], []

        xypred = np.split(ypred, 30)

        xytest = np.split(ytest, 30)

        for i in range(0, 30, 2):

            predx.append(xypred[i])

            predy.append(xypred[i+1])

            testx.append(xytest[i])

            testy.append(xytest[i + 1])

        plt.imshow(xtest.reshape(96, 96))

        #plt.imshow(self.train[9].values)

        plt.plot(predx, predy, 'o', color='red')

        plt.plot(testx, testy, 'o', color='blue')

        plt.show()





    def run(self):

        self.define_CNN()

        self.Adam(epochs=200, batchSize= 128)

        self.show_any_result()

        self.prediction()



import os

if __name__ == '__main__':

    dataPath =  '../input/training/training.csv'

    testPath = '../input/test/test.csv'

    data_clean = Data_Clean(dataPath, testPath)

    train, label, test = data_clean.run()



    global random_seed

    random_seed = 9

    xtrain, xtest, ytrain, ytest = train_test_split(train, label, test_size=0.1, random_state=random_seed)



    cnn = CNN(xtrain, xtest, ytrain, ytest, test)

    model = cnn.run()