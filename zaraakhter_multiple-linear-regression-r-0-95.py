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
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns 

from sklearn.metrics import r2_score
data = pd.read_csv('../input/fish-market/Fish.csv')
data.head()
data.describe()
data.isnull().sum()
data.Species.value_counts()
#plot count of species

sns.catplot(x="Species", kind="count", palette="ch:.25", data=data, height=5, aspect=3).set(title = 'Count of Species')
#average weights of fish against species

avg_weights = data.groupby('Species')['Weight'].mean()

plt.figure(figsize = (15,10))

avg_weights.plot(kind = 'bar')

plt.ylabel('Avg Weight')

plt.title('Average Weight of Species')

plt.show()
#Pearson's correlation 

data[data.columns[1:]].corr()['Weight'][:]

plt.figure(figsize = (15,10))

sns.heatmap(data.corr(), annot=True)


data_weight = data['Weight']

q1 = data_weight.quantile(0.25)

q3 = data_weight.quantile(0.75)

iqr = q3 - q1

lowerend = q1 - (1.5 * iqr)

upperend = q3 + (1.5 * iqr)
outliers = data_weight[(data_weight < lowerend) | (data_weight > upperend)]

outliers
#outliers in dataset



data[142:145]
#removing outliers in dataset

data = data.drop([142,143,144])
cat_species = pd.get_dummies(data['Species'], prefix='species')
data = data.drop(columns = 'Species')
data.head()
#concat categorical columns with data

data = pd.concat([data, cat_species], axis = 1)
data.head()
#separating target column 

X = data.iloc[:,1:]

y = data[['Weight']]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
#size of train and test datasets

print('x_train: ', np.shape(x_train))

print('y_train: ', np.shape(y_train))

print('x_test: ', np.shape(x_test))

print('y_test: ', np.shape(y_test))

from sklearn.linear_model import LinearRegression
lreg = LinearRegression()

lreg.fit(x_train, y_train)
#model parameters

print('Model intercept: ', lreg.intercept_)

print('Model coefficients: ', lreg.coef_)
#prediction on train

y_pred = lreg.predict(x_train)
r2_score(y_train, y_pred)
#prediction on test

y_pred = lreg.predict(x_test)
r2_score(y_test, y_pred)


result = pd.DataFrame(y_test)

result = result.reset_index(drop= True)

result.head()
y_pred_new = pd.DataFrame(y_pred, columns = ['Predicted Weight'])
result = pd.concat([result, y_pred_new], axis = 1)
result