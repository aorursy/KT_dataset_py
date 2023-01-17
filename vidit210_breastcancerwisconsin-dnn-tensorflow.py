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
df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df.head()
df.drop(columns=['id','Unnamed: 32'],inplace=True)
df.head()
diagnosis_dict = {'M':0,'B':1}
df['diagnosis'] = df['diagnosis'].map(diagnosis_dict)
df.head()
X = df.iloc[:,1:].values

y = df.iloc[:,0].values

X.shape
y.shape
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train.shape
y_train
model = Sequential()

model.add(Dense(units=128,activation='relu',input_dim=30))

model.add(Dropout(0.2))

model.add(Dense(units=64,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(units=1,activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=10)
test_loss,test_accuracy = model.evaluate(X_test,y_test)
from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
y_pred = y_pred >=0.5
cm = confusion_matrix(y_test,y_pred)
y_pred

y_test
cm
model.save('pcw.h5')