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
filepath= '/kaggle/input/mushroom-classification/mushrooms.csv'
data=pd.read_csv(filepath)
data
classs=data['class']

features=data.drop('class',axis=1)
#One hot encoding

features= pd.get_dummies(features)

features
#Change the categorical variables in the Classs column to numbers

classs.replace('p',0,inplace=True)

classs.replace('e',1,inplace=True)
classs
from sklearn.model_selection import train_test_split



y=classs

x=features



#Splitting training and testing data

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.70,test_size=0.30, random_state=0)
import tensorflow as tf



from tensorflow import keras

model=keras.Sequential([

    keras.layers.Dense(32,input_shape=(117,)),

    keras.layers.Dense(20,activation=tf.nn.relu),

    keras.layers.Dense(2,activation="softmax")

    

])
features.shape
model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['acc']) 
prediction=model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))
type(prediction)
prediction_features= model.predict(x_test)
prediction_features
a=prediction_features.tolist()
pred=[]

for i in a:

   # print(i[0])

    if i[0]>i[1]:

        pred.append(0)

    else:

        pred.append(1)

        
#'pred' is the prediction of the test data

pred
#To find the models accuracy:

def accuracy(x, y):

    return (100.0 * len(set(x) & set(y))) / len(set(x) | set(y))



print(accuracy(pred, y_test))