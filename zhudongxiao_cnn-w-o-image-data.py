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
#CNN for classifier， Do not use image Data



import os

import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler

from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split



class Data_Clean(object):

    def __init__(self):

        self.numerical_data, self.num_test_data = self.read_numerical_data()

        self.id, self.species, self.num_train, self.test_id, self.test_num = self.split_numerical_data()



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



    def run(self):

        ##Check the data

        # print('id:', self.id.loc[0])

        # print('species:', self.species[0])

        # print('feature:', self.num_train[0])

        # plt.imshow(self.image_data[0].reshape(960, 960), cmap='gray')

        # plt.show()

        return self.species, self.num_train, self.test_num, self.test_id



from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout

from keras.optimizers import RMSprop

from keras.callbacks import ReduceLROnPlateau



class CNN(object):

    def __init__(self, num_train, num_vali, species_train, species_vali, num_test, test_id):

        self.num_train = num_train

        self.num_vali = num_vali

        self.species_train = species_train

        self.species_vali = species_vali

        self.num_test = num_test

        self.test_id = test_id



    def define_CNN(self):

        model = Sequential()  #0.9798

        model.add(Dense(512, activation='relu', input_dim=192))

        model.add(Dropout(0.25))

        model.add(Dense(512, activation='relu'))

        model.add(Dropout(0.25))

        model.add(Dense(99, activation='softmax'))

        self.model = model





    def RMSprop(self, batch_size, epoch):

        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)

        self.model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])

        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

        self.model.fit(self.num_train, self.species_train, batch_size=batch_size, epochs=epoch,

                       validation_data=(self.num_vali, self.species_vali), verbose=2, callbacks=[learning_rate_reduction])



    def make_predict(self):

        ypred_prob = self.model.predict(self.num_test)

        root = "../input"

        labels = sorted(pd.read_csv(os.path.join(root, 'train.csv')).species.unique())

        ypred = pd.DataFrame(ypred_prob, index = self.test_id, columns=labels)

        print(ypred.head(2))

        fp = open('submit.csv', 'w')

        fp.write(ypred.to_csv())



    def run(self):

        self.define_CNN()

        self.RMSprop(batch_size = 128, epoch=100)

        self.make_predict()





if __name__ == '__main__':

    data_clean = Data_Clean()

    species, num, num_test,  test_id = data_clean.run()

    #划分训练集和测试集

    species_train, species_vali, num_train, num_vali = train_test_split(species, num, test_size=0.1, random_state=26)



    #搭建CNN模型，看看CNN对图像的分类情况

    cnn = CNN(num_train, num_vali,species_train, species_vali, num_test, test_id)

    cnn.run()






