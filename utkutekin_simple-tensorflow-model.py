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

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, LabelEncoder

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score
# load data

data = pd.read_csv('../input/churn-modellingcsv/Churn_Modelling.csv')
data.info() # general information
# row numbers, customer id, surname are useless, so drop from data

data = data.drop(['CustomerId', 'Surname'], axis = 1)
# seperate data as idepedent variables and dependent variable

x = data.iloc[:,0:-1]

y = data.iloc[:,-1:]
# seperate indepedent variables for preprocessing (categorical and numerical)

x_numeric = x.select_dtypes(exclude = "object")

x_categoric = x.select_dtypes(include = "object")
# scale data

sc = StandardScaler()

x_numeric_scaled = sc.fit_transform(x_numeric)

x_numeric_scaled = pd.DataFrame(data = x_numeric_scaled, columns = x_numeric.columns)
# convert categorical data to numeric data

le = LabelEncoder()

for col in x_categoric.columns:

    x_categoric[col] = le.fit_transform(x_categoric[col])
# concat indepedent variables

final_x = pd.concat([x_numeric_scaled, x_categoric], axis = 1)
# seperate data as train and test

x_train, x_test, y_train, y_test = train_test_split(final_x, y, test_size = 0.33)

N, D = x_train.shape
# tensorflow model

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(5, kernel_initializer = 'uniform', activation = 'relu', input_dim = D))

model.add(tf.keras.layers.Dense(5, kernel_initializer = 'uniform', activation = 'relu'))

model.add(tf.keras.layers.Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

r = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 500)
# visualisation of accuracy

plt.plot(r.history['accuracy'], label = 'accuracy')

plt.plot(r.history['val_accuracy'], label = 'val_accuracy')

plt.grid()

plt.legend()

plt.show()
# visualisation of loss

plt.plot(r.history['loss'], label = 'loss')

plt.plot(r.history['val_loss'], label = 'val_loss')

plt.grid()

plt.legend()

plt.show()
y_pred = model.predict(x_test)

y_pred = y_pred > 0.5

le2 = LabelEncoder()

y_pred = le.fit_transform(y_pred)
# metrics

acc_score = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print(acc_score)
print(cm)