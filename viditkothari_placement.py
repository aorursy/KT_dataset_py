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
import matplotlib as plt
import seaborn as sb
import tensorflow as tf
from sklearn import model_selection
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder 
data = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.head()
data['workex'].replace(to_replace ="Yes", 
                 value =1,inplace=True) 
data['workex'].replace(to_replace ="No", 
                 value =0,inplace=True) 

data['status'].replace(to_replace ="Placed", 
                 value =1,inplace=True) 
data['status'].replace(to_replace ="Not Placed", 
                 value =0,inplace=True) 

data['gender'].replace(to_replace ="M", 
                 value =1,inplace=True) 
data['gender'].replace(to_replace ="F", 
                 value =0,inplace=True) 

#We can also convert categorical data using LabelEncoder
le = LabelEncoder() 
  
data['ssc_b']= le.fit_transform(data['ssc_b']) 
data['hsc_b']= le.fit_transform(data['hsc_b'])
data['hsc_s']= le.fit_transform(data['hsc_s'])
data['degree_t']= le.fit_transform(data['degree_t'])
data['specialisation']= le.fit_transform(data['specialisation'])


data.head()
X = data[data.columns[1:12]]
Y = data[data.columns[13]]
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.95)
model = Sequential()
model.add(Dense(32, input_dim=11, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='RMSProp', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=200, batch_size=5,verbose = 0)
# evaluate the keras model
_, accuracy = model.evaluate(X_test, Y_test)
print('Accuracy: %.2f' % (accuracy*100))