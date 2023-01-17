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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv('../input/15cse308nndleval/train.csv')
df.head()
df.shape
df.isnull().sum()
data = df.dropna()
data.isnull().sum()
data.shape
data.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['department'] = le.fit_transform(data['department'])

data['region'] = le.fit_transform(data['region'])
data['education'] = le.fit_transform(data['education'])
data['gender'] = le.fit_transform(data['gender'])
data['recruitment_channel'] = le.fit_transform(data['recruitment_channel'])


data.head()
data.describe()
from keras import Sequential
from keras.layers import Dense, Activation, Dropout
x_train = data.drop(["is_promoted"], axis=1)
y_train = data["is_promoted"]
y_train.head()
x_train.head()
model = Sequential()
model.add(Dense(8, input_dim=13, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(2, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])


model.fit(x_train, y_train, batch_size=10, epochs=32,  validation_split=0.2)
dt = pd.read_csv('../input/15cse308nndleval/test.csv')
dt.shape
dt.isnull().sum()
data_test = dt.dropna()
data_test.head()
data_test.isnull().sum()
data_test['department'] = le.fit_transform(data_test['department'])
data_test['region'] = le.fit_transform(data_test['region'])
data_test['education'] = le.fit_transform(data_test['education'])
data_test['gender'] = le.fit_transform(data_test['gender'])
data_test['recruitment_channel'] = le.fit_transform(data_test['recruitment_channel'])


# x_test = data_test.drop(["is_promoted"], axis=1)
data_test.shape
y_train=y_train[0:5179]
p = model.fit(data_test,y_train)





