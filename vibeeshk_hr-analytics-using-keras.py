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
data=pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

data.head() 

#Replace categorical variables

data['Attrition'].replace('No',0,inplace=True)

data['Attrition'].replace('Yes',1,inplace=True)

data= pd.get_dummies(data)

data
#Split the training and testing data

from sklearn.model_selection import train_test_split



y=data['Attrition']

x=data.drop('Attrition',axis=1) 



#Splitting training and testing data

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70,test_size=0.30, random_state=0)
#Apply Nueral Networks



import tensorflow as tf



from tensorflow import keras

model=keras.Sequential([

    keras.layers.Dense(15,input_shape=(55,)),

    keras.layers.Dense(7,activation=tf.nn.relu),

    keras.layers.Dense(2,activation="softmax")

])

model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['acc'])

prediction=model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))