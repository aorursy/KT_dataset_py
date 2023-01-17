import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")

data.head()
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)

data
data.diagnosis = [1 if each=='M' else 0 for each in data.diagnosis]
y = data.diagnosis.values

x = data.drop(["diagnosis"],axis=1)

x,y
from sklearn.model_selection import train_test_split



x_train , x_test , y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=1)

x_train
from tensorflow import keras

from keras.models import Sequential 

from keras.layers import Dense

from sklearn.model_selection import cross_val_score



from keras.wrappers.scikit_learn import KerasClassifier



def build_classifier():

    classifier = Sequential()

    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))

    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 300)

acc = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)

mean = acc.mean()
print("Accuracy mean: "+ str(mean))