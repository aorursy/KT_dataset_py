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
dataset_path = '../input/heart.csv'

dataset = pd.read_csv(dataset_path)

print(dataset.head())
data_len = len(dataset.columns) - 1

X = dataset.iloc[:, 0:data_len]

y = dataset.iloc[:, data_len]
# train 8 : 2 test



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# feature scaling



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
import keras

from keras.models import Sequential

from keras.layers import Dense





classifier = Sequential()



classifier.add(Dense(units = 25, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))

classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



classifier.fit(X_train, y_train, batch_size = 40, epochs = 200)
loss, accuracy = classifier.evaluate(X_test, y_test)

print(accuracy)