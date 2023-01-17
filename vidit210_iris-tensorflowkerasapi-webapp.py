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

import seaborn as sns
df = pd.read_csv("/kaggle/input/iris/Iris.csv")

df.head()
df.tail()

df.drop(columns=['Id'],inplace=True)

df.head()

df.info()

df.describe()

df['Species'].unique()

species_dummies = pd.get_dummies(df.Species)

df=pd.concat([df,species_dummies],axis=1)
df.drop(columns='Species',inplace=True)

df.head()
df.tail()
X =df.iloc[:,:4]

y=df.iloc[:,4:]
X.head()

y.head()
type(X)
type(y)
X=X.values

y=y.values

type(X)
type(y)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
import tensorflow as tf

from tensorflow.keras.layers import Dense,Dropout

from tensorflow.keras.models import Sequential
model = Sequential()

model.add(Dense(units=128,activation='relu',input_dim=4))

model.add(Dropout(0.2))

model.add(Dense(units=100,activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(units=3,activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print(X_train.shape)

print(y_train.shape)
model.fit(X_train,y_train,epochs=200)

from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
y_pred

y_pred.round()

y_test
y_pred = (y_pred > 0.5) 

y_pred
a=np.argmax(y_pred,axis=1)
a
b=np.argmax(y_test,axis=1)

b
cm = confusion_matrix(a,b)

cm
df.tail()
model.predict([[5.9,3.0,5.1,1.8]]).round()

model.save("iris.h5")
