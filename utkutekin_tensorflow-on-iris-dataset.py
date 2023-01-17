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
# import libraries

import tensorflow as tf

import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix
# load data

data = pd.read_csv('../input/iris/Iris.csv', index_col = 0)
data.head()
x = data.iloc[:,0:-1] # independent variable

y = data.iloc[:,-1:] # depedent variable (target variable)
x.head()
y.head()
# standardize indepedent variables

sc = StandardScaler()

x_scaled = sc.fit_transform(x)
# categorical to numeric

le = LabelEncoder()

y_labeled = le.fit_transform(y)
# train and test data

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_labeled, test_size = 0.33)

N, D = x_train.shape
# build tensorflow model

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(5, kernel_initializer = 'uniform', activation = 'relu', input_dim = D))

model.add(tf.keras.layers.Dense(5, kernel_initializer = 'uniform', activation = 'relu'))

model.add(tf.keras.layers.Dense(3, kernel_initializer = 'uniform', activation = 'softmax'))

model.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

r = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 500)
plt.figure(figsize = (16,6))

plt.plot(r.history['accuracy'], label = 'accuracy')

plt.plot(r.history['val_accuracy'], label = 'val_accuracy')

plt.grid()

plt.legend()
y_pred = model.predict(x_test)
y_pred = pd.DataFrame(data = y_pred)
y_pred = y_pred.idxmax(axis = 1)
cm = confusion_matrix(y_test, y_pred)

print(cm)
acc_score = accuracy_score(y_test, y_pred)

print(acc_score)