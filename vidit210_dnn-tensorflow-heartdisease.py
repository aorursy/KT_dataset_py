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
import pandas as pd

import numpy as np 

import tensorflow as tf

from tensorflow.keras.layers import Dense,Dropout

from tensorflow.keras.models import Sequential
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.head()
X = df.iloc[:,:-1].values

y=df.iloc[:,-1].values
X.shape
y.shape
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
model = Sequential()

X_train.shape

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
y_train.shape

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
model.add(Dense(units=128,activation='relu',input_dim = 13))

model.add(Dropout(0.2))

model.add(Dense(units=64,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(units=1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(X_train,y_train,epochs=1000,batch_size=32)
test_loss,test_accuracy = model.evaluate(X_test,y_test)
y_pred = model.predict(X_test)

y_pred = y_pred >=0.5
y_pred[0:10]
y_test[0:10]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

cm