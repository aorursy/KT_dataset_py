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
import tensorflow as tf

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import matplotlib.pyplot as plt

#%% load and preprocess data set

train = pd.read_csv('../input/titanic/train.csv')

train_copy = train.copy()

test = pd.read_csv('../input/titanic/test.csv')

test_copy = test.copy()
#%% train data preperation

train = train.drop(['Name', 'PassengerId', 'Cabin', 'Ticket'], axis = 1)

y_train = pd.DataFrame(data = train['Survived'])



col_missing_data = [col for col in train.columns if train[col].isnull().any()]

value = train.isnull().sum()

#print(value[value > 0])



# fill emptry age cells with average

train['Age'].fillna((train['Age'].mean()), inplace = True)

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].value_counts().index[0])

train = train.drop(['Survived'], axis = 1)
#%% test data preperation

test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

test['Age'].fillna((test['Age'].mean()), inplace = True)

test['Fare'].fillna((test['Fare'].mean()), inplace = True)
#%% concat data for same scalling

final_data = pd.concat([train, test], axis = 0)

test_onehot = final_data.iloc[:,-1:]

ohe = OneHotEncoder()

test2 = pd.DataFrame(data = ohe.fit_transform(test_onehot).toarray(), columns = ['C','Q','S'], index = final_data.index)

final_data = final_data.drop(['Embarked'], axis = 1)

final_2 = pd.concat([final_data, test2], axis = 1)

le = LabelEncoder()

final_2['Sex'] = le.fit_transform(final_2['Sex'])



sc = StandardScaler()

final_2.iloc[:,0:6] = sc.fit_transform(final_2.iloc[:,0:6])
#%% seperate again as train and test

train_new = final_2.iloc[0:891,:]

test_new = final_2.iloc[891:,:]

N,D = train_new.shape
#%% tensorflow model

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(5, kernel_initializer = 'uniform', activation = 'relu', input_dim = D))

model.add(tf.keras.layers.Dense(5, kernel_initializer = 'uniform', activation = 'relu'))

model.add(tf.keras.layers.Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#%% fitting model

r = model.fit(train_new, y_train, epochs = 500)
#%% predictions

predictions = model.predict(test_new)

for i in range(len(predictions)):

    if predictions[i] > 0.5:

        predictions[i] = 1

    else:

        predictions[i] = 0



predictions_n = pd.DataFrame(data = predictions, columns = ['Survived'])
output = pd.concat([test_copy['PassengerId'],predictions_n], axis = 1)

output.to_csv("my_submission.csv", index = False)

print("done!")