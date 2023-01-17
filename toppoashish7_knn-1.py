# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# importing datasets 

train_data = pd.read_csv("/kaggle/input/forest-cover-type-prediction/train.csv")

test_data = pd.read_csv("/kaggle/input/forest-cover-type-prediction/test.csv")
print(train_data.shape)

print(test_data.shape)
train_data.info()
test_data.info()
print(train_data.columns)

train_data.head()
print(test_data.columns)

test_data.head()
# checking if any column has got any nan value in train_data

train_data.isnull().sum()
# checking if any column has got any nan value in test_data

test_data.isnull().sum()
#saving this particular column to satisfy submission requirement

y_test_id = test_data['Id']

y_test_id.shape #for confirmation
#dropping id data as it's not a good feature

train_data.drop(['Id'], axis=1, inplace=True)

test_data.drop(['Id'], axis=1, inplace=True)

print(train_data.shape, test_data.shape) #for confirmation
Y = train_data['Cover_Type']

X = train_data.drop(['Cover_Type'],axis=1)
print(X.shape, Y.shape)
#splitting the train_data for training and evaluation data

X_train, X_test, y_train, y_test = train_test_split(X, Y) 
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))

predictions = knn.predict(test_data)
Submission=pd.DataFrame(data=predictions,columns=['Cover_Type'])

Submission.head()
Submission['Id']=y_test_id

Submission.set_index('Id',inplace=True)
Submission.head()
Submission.to_csv('Submission.csv')