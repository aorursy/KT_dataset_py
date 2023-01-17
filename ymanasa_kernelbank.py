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


bankdata=pd.read_csv("../input/Churn_Modelling.csv")

bankdata.head(10)
bankdata.describe
bankdata.columns
bank_data=bankdata.drop(["RowNumber","Surname"],axis=1)
bank_data.columns


bank_data.isnull().sum()

# no null values
# dealing with categorial values

from sklearn import preprocessing

le_Geography=preprocessing.LabelEncoder()

le_Geography.fit(bank_data.Geography.unique())

bank_data.Geography=le_Geography.transform(bank_data.Geography)
le_Gender=preprocessing.LabelEncoder()

le_Gender.fit(bank_data.Gender.unique())

bank_data.Gender=le_Gender.transform(bank_data.Gender)
bank_data.head()
# scaling data

from scipy import stats

bank_data['CreditScore_norm']=stats.zscore(bank_data['CreditScore'])

bank_data['Age_norm']=stats.zscore(bank_data['Age'])

bank_data['Balance_norm']=stats.zscore(bank_data['Balance'])

bank_data['EstimatedSalary_norm']=stats.zscore(bank_data['EstimatedSalary'])
bank_data.columns
X=bank_data.drop(["CustomerId","CreditScore","Age","Balance","EstimatedSalary","Exited"],axis=1)

X.head()
X.shape
Y=bank_data.iloc[:,bank_data.columns=="Exited"]

Y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers
model=keras.Sequential([layers.Dense(64,activation=tf.nn.sigmoid,input_shape=[10]),

                       layers.Dense(64,activation=tf.nn.sigmoid),

                       layers.Dense(2,activation=tf.nn.sigmoid)

                       ])
model.compile(loss='sparse_categorical_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=10)