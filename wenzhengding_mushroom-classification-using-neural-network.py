# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# import keras part

from keras.models import Sequential

from keras.layers import Dense



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read data and see how many data rows and features we have

data = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")

data.shape
# Now Let's see some examples of our data

data.head(6)
# Use one hot encoding to make the categorical data to numerical data

undummy_X = data.iloc[:,1:23]

undummy_y = data.iloc[:, 0]

X = pd.get_dummies(undummy_X)

y = pd.get_dummies(undummy_y)

# Split data set into training and testing set and check the testing set size

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

X_test.shape
# Fit the training set with neural network 

classifier=Sequential()

classifier.add(Dense(64,activation='relu',input_dim=117))

classifier.add(Dense(2,activation='softmax'))

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train,batch_size=10,epochs=100)
# Get the prediction and the accuracy score

from sklearn.metrics import accuracy_score

y_pred=classifier.predict(X_test)



# Process the prediction to make it suitable for accuracy_score function

y_pred=y_pred>0.5

y_pred_int = y_pred.astype(int)



accuracy_score(y_pred_int, y_test)
# Let's check the first 10 values in test set

y_test.head(10)
# Let's check the first 10 predictions

y_pred_int[:10]