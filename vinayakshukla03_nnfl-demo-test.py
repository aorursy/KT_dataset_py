# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv("../input/nnfl-demo-lab-1/train-2.csv")
df_train['Type'].value_counts()

X = df_train.drop('Type', axis = 1).values
y = df_train['Type'].values.reshape(-1, 1)
X = np.delete(X, [9], axis = 1)
print(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 101)
sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.fit_transform(X_test)
print(X_train_scaled.shape)
print(X_test.shape)
print(y_train.shape)
from keras.utils import to_categorical
def encode(data):
    encoded = to_categorical(data)
    return encoded
y_train_encoded = encode(y_train)
y_train_encoded = np.delete(y_train_encoded, [0,4], axis = 1)
y_train_encoded
y_test_encoded = encode(y_test)
y_test_encoded = np.delete(y_test_encoded, [0,4], axis = 1)
model = Sequential()
model.add(Dense(25, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(10, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train_scaled, y_train_encoded, validation_data = (X_test_scaled, y_test_encoded), batch_size = 1, epochs = 50)

df_test = pd.read_csv('/kaggle/input/nnfl-demo-lab-1/test-2.csv')
X_predict = df_test.drop('Id', axis = 1)
predictions = model.predict(X_predict)
print(predictions)
