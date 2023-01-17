# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/airline-passenger-satisfaction/train.csv')

train_df.head()
train_df.shape
train_df.info()
train_df.isnull().sum()
train_df['Arrival Delay in Minutes'].fillna((train_df['Arrival Delay in Minutes'].mean()), inplace=True)
train_df.isna().sum()
train_df['satisfaction'].value_counts().plot(kind='bar', rot=0, title='People are more neutral/dissatisfied than satisfied with the airline')
customer_df = train_df.groupby(['satisfaction', 'Customer Type'])['satisfaction'].count().unstack('Customer Type')

customer_df.plot(kind='bar',figsize = (10,5), rot=0, title='Loyal customers are more dissatisfied with the airline')
class_df = train_df.groupby(['satisfaction', 'Class'])['satisfaction'].count().unstack('Class')

class_df.plot(kind='bar',figsize = (10,5), rot=0, title='People travelling business class are most satisfied')
train_df['Class'].value_counts().plot(kind='bar', rot=0, title='People travel more via Eco rather than Eco Plus')
train_df['Type of Travel'].value_counts().plot(kind='bar', rot=0)
legroom_df = train_df.groupby(['satisfaction', 'Leg room service'])['satisfaction'].count().unstack('Leg room service')

legroom_df.plot(kind='bar',figsize = (10,5), rot=0, colormap='Blues', title='Leg room service needs to be improved')
clean_df = train_df.groupby(['satisfaction', 'Cleanliness'])['satisfaction'].count().unstack('Cleanliness')

clean_df.plot(kind='bar',figsize = (10,5), rot=0, colormap='Blues', title='People are not finding the airplanes clean')
train_df['Gender'].value_counts().plot(kind='bar', rot=0)
gender_df = train_df.groupby(['satisfaction', 'Gender'])['satisfaction'].count().unstack('Gender')

gender_df.plot(kind='bar', rot=0)
train_df = train_df.drop(['Unnamed: 0', 'id'], axis=1)

train_df.head()
train_df['satisfaction'].replace({'satisfied': 1, 'neutral or dissatisfied': 0}, inplace=True)
train_df['Gender'].replace({'Male': 0, 'Female': 1}, inplace=True)
train_df['Type of Travel'].replace({'Personal Travel': 0, 'Business travel': 1}, inplace=True)
train_df['Customer Type'].replace({'Loyal Customer': 0, 'disloyal Customer': 1}, inplace=True)
train_df['Class'].replace({'Eco Plus': 0, 'Business': 1, 'Eco': 2}, inplace=True)
from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()

scaler.fit(train_df)

train_df = pd.DataFrame(scaler.transform(train_df), index = train_df.index, columns = train_df.columns)
x_train = train_df.drop('satisfaction', axis = 1)

y_train = train_df['satisfaction']
model = LogisticRegression(random_state = 24)

model.fit(x_train, y_train)
test_df = pd.read_csv('/kaggle/input/airline-passenger-satisfaction/test.csv')

test_df.head()
test_df['satisfaction'].replace({'satisfied': 1, 'neutral or dissatisfied': 0}, inplace=True)

test_df['Gender'].replace({'Male': 0, 'Female': 1}, inplace=True)

test_df['Type of Travel'].replace({'Personal Travel': 0, 'Business travel': 1}, inplace=True)

test_df['Customer Type'].replace({'Loyal Customer': 0, 'disloyal Customer': 1}, inplace=True)

test_df['Class'].replace({'Eco Plus': 0, 'Business': 1, 'Eco': 2}, inplace=True)
test_df['Arrival Delay in Minutes'].fillna((test_df['Arrival Delay in Minutes'].mean()), inplace=True)

test_df = test_df.drop(['Unnamed: 0','id'], axis=1)
scaler.fit(test_df)

test_df = pd.DataFrame(scaler.transform(test_df), index = test_df.index, columns = test_df.columns)
x_test = test_df.drop('satisfaction',axis=1)

y_test = test_df['satisfaction']
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,y_pred)

score