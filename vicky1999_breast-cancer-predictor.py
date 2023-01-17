# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

data.head()
labels=data['diagnosis'].replace({'M':0,'B':1})

labels.head()
data.columns

data=data.drop(['diagnosis','Unnamed: 32'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.2,random_state=40)
from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(x_train,y_train)
pred=model.predict(x_test)
from sklearn.metrics import accuracy_score

print("Accuracy using Logisitic Regression : ",accuracy_score(y_test,pred))
from sklearn.tree import DecisionTreeClassifier

model2=DecisionTreeClassifier()

model2.fit(x_train,y_train)

pred=model2.predict(x_test)

print("Accuracy using Decision Tree: ",accuracy_score(y_test,pred))
from sklearn.ensemble import RandomForestClassifier

model3=RandomForestClassifier()

model3.fit(x_train,y_train)

pred=model3.predict(x_test)

print("Accuracy using Random Forest: ",accuracy_score(y_test,pred))
import tensorflow as tf

network=tf.keras.models.Sequential([

    tf.keras.layers.Dense(512,activation='relu'),

    tf.keras.layers.Dense(128,activation='relu'),

    tf.keras.layers.Dense(16,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')

])
network.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
network.fit(x_train.to_numpy(),y_train.to_numpy(),epochs=10)
pred=model.predict(x_test)

print("Accuracy using Neural Network : ",accuracy_score(y_test,pred))