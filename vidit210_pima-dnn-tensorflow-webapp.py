# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
df.head()
df.info()
df.describe()
X = df.iloc[:,:-1].values

X.shape
y = df.iloc[:,-1].values

y.shape
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2)
X_train.shape
y_train.shape
X_test.shape
y_test.shape
import tensorflow as tf

from tensorflow.keras.layers import Dense,Dropout

from tensorflow.keras.models import Sequential
model = Sequential()

model.add(Dense(units =256,activation='relu',input_dim=8))

model.add(Dropout(0.2))

model.add(Dense(units=128,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(units=64,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(units=32,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(units=16,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(units=1,activation='sigmoid'))
model.summary()
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=3000)
y_pred = model.predict(X_test)

y_pred=(y_pred>0.5)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred,y_test)
cm
model.save("pima.h5")