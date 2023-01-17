# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Mis librerias

from keras.models import Sequential

from keras.layers import Dense

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import KFold, cross_val_score

from sklearn.preprocessing import LabelEncoder

import keras

import numpy as np

seed = np.random.seed(7)



# Any results you write to the current directory are saved as output.
def Data():

    '''

    Método que genera datos

    objetos = 11999

    caracteristicas = 500

    clases = 3 desde 0,1,2

    :return:

    '''

    data = np.random.random((11999,500))

    labels = np.random.randint(3, size=(11999,1))

    y_train = keras.utils.to_categorical(labels, num_classes=3)

    

    NeuralNetwork(data, y_train)
def NeuralNetwork(data, target):

    input_dim = data.shape[1]



    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    cvscores = []



    for train, test in skf.split(data, target):

        model = Sequential()



        model.add(Dense(32, activation='relu', input_dim=input_dim))

        model.add(Dense(12, activation='relu'))

        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',

                          metrics=['accuracy'])



        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



        model.fit(data[train], target[train], epochs=5, batch_size=10, verbose=0)

        scores = model.evaluate(data[test], target[test], verbose=0)



        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        cvscores.append(scores[1] * 100)
if __name__ == '__main__':

    Data()