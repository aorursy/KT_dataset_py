# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

import tensorflow as tf

import keras

from keras.utils import to_categorical

import seaborn as sns

from sklearn.compose import ColumnTransformer





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv(r'/kaggle/input/crystal-system-properties-for-liion-batteries/lithium-ion batteries.csv')
data.head()
data = data.drop(['Materials Id', 'Volume', 'Nsites', 'Density (gm/cc)'], axis=1)
numerical_transformer = StandardScaler()

label_transformer = OrdinalEncoder()



n_cols = [c for c in data.columns if data[c].dtype in ['int64', 'float64', 'int32', 'float32']]

obj_cols = [c for c in data.columns if data[c].dtype in ['object', 'bool']]

print(n_cols, obj_cols)



ct = ColumnTransformer([

    ('num', numerical_transformer, n_cols),

    ('non_num', label_transformer, obj_cols),

])

processed = ct.fit_transform(data)

new_data = pd.DataFrame(columns=data.columns, data=processed)

new_data.head()
X = new_data.drop('Crystal System', axis=1)

y = new_data['Crystal System']
plt.figure(figsize=(12, 10))

corr_matrix = X.corr()

sns.heatmap(corr_matrix, lw=0.5, cmap='coolwarm', annot=True)
def train_model():

    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2, shuffle=True)

    y_encoded = to_categorical(y_train)

    model = keras.models.Sequential()

    model.add(keras.layers.Dense(1024, activation='relu'))

    model.add(keras.layers.Dense(512, activation='relu'))

    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(256, activation='relu'))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(128, activation='relu'))

    model.add(keras.layers.Dense(128, activation='relu'))

    model.add(keras.layers.Dense(64, activation='softsign'))

    model.add(keras.layers.Dense(3, activation='softmax'))



    model.compile(optimizer='adam',

                 loss='categorical_crossentropy',

                 metrics=['accuracy'])



    model.fit(X_train, y_encoded, epochs=100, verbose=False)



    preds=model.predict_classes(X_test)

    

    return accuracy_score(y_test, preds)



def train_n_strats(n_strats):

    score=[]

    for i in range(n_strats):

        score.append(train_model())

    print(f'Average score of {n_strats} runs: '+ str(sum(score)/len(score)))

    

train_n_strats(10)