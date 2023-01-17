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
import pandas as pd



data = pd.read_csv('../input/Churn_Modelling.csv')

data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis = 1)



Y = data['Exited']

data = data.drop(['Exited'], axis = 1)

data = pd.get_dummies(data)

X = data.drop(['Geography_France', 'Gender_Female'], axis = 1)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)



import keras

from keras.models import Sequential

from keras.layers import Dense



my_ann = Sequential()



my_ann.add(Dense(units=32, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))



my_ann.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))



my_ann.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



#print(my_ann.summary())



my_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



from sklearn.model_selection import train_test_split

[xtrain, xtest, ytrain, ytest] = train_test_split(X, Y, test_size = 0.3, random_state = 42)



my_ann.fit(xtrain, ytrain, batch_size = 10, epochs = 100)



ypred = my_ann.predict(xtest)



ypred = (ypred>0.5)



from sklearn.metrics import accuracy_score

acc = accuracy_score(ytest, ypred)

print(acc)