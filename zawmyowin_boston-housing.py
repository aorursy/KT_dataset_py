# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Import packages

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Provide Data

boston_dataset = load_boston()

print(boston_dataset.keys())

boston = pd.DataFrame(boston_dataset.data, columns = 

                      boston_dataset.feature_names)

boston['MEDV']=boston_dataset.target

boston.columns = map(str.lower,boston.columns)
# Data Processing

b_null = boston.isnull().sum()

print(b_null)
# Exploratory Data Analysis

sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.distplot(boston['medv'],bins=30)

plt.savefig('Exploratory.png',dpi=500)

plt.show()
# Correlation

correlation_matrix = boston.corr().round(2)

sns_plot=sns.heatmap(data=correlation_matrix, annot=True)
# Variation

plt.figure(figsize=(20,5))

features =['lstat','rm']

target = boston['medv']

for i, col in enumerate(features):

    plt.subplot(1,len(features),i+1)

    x = boston[col]

    y=target

    plt.scatter(x,y,marker='o')

    plt.title(col)

    plt.xlabel(col)

    plt.ylabel('medv')
#prepare data for training model

X = pd.DataFrame(np.c_[boston['lstat'],boston['rm']],\

columns =['lstat','rm'])

y = boston['medv']
#split train and test data

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3418,\

                                         random_state=0)
# Create Model and Fit

model = LinearRegression().fit(X_train,y_train)
# Predict

train_pred = model.predict(X_train)
# Model evaluation

r_sq = model.score(X_train,y_train)

print('The model performance for training set')

print('--------------------------------------')

print('coefficient of determination:',r_sq)

print('Mean Absolute Error:',mean_absolute_error(y_train,train_pred))

print('Root Mean Squared Error:',np.sqrt

      (mean_absolute_error(y_train,train_pred)))