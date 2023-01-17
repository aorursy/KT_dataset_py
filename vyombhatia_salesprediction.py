import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing tools for preprocessing and feature engineering of the data:

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from category_encoders import CatBoostEncoder

from sklearn.preprocessing import LabelEncoder



# Importing plotting tools to plot the data:

import seaborn as sns

import matplotlib.pyplot as plt



# Importing Algorithms so that I can train the data:

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor



# Importing a loss tool to check how well our algorithms is doing:

from keras.losses import mean_absolute_percentage_error

data = pd.read_csv("../input/predict-demand/train.csv")

data.head()
data.isnull().sum()
dropdata = data.dropna()
y = dropdata['quantity']



dropdata.drop(['quantity', 'id'], inplace=True, axis=1)
sns.set_context("poster", font_scale=.7)

plt.figure(figsize=(7,7))

sns.set_palette('RdYlBu')

sns.countplot(dropdata['city'])
sns.set_palette('PiYG')

plt.figure(figsize=(10,10))

sns.set_context("poster", font_scale=0.7)

sns.countplot(dropdata['shop'])
sns.set_palette('RdPu')

plt.figure(figsize=(10,10))

sns.set_context("poster", font_scale=0.7)

sns.scatterplot(data = dropdata, y='price', x=y, hue='capacity')
sns.set_palette('YlOrRd')

plt.figure(figsize=(10,10))

sns.set_context("poster", font_scale=0.7)

sns.countplot(dropdata['brand'])
c = (data.dtypes == 'object')

categorical_col = list(c[c].index)
enc = CatBoostEncoder()

enc.fit(dropdata[categorical_col], y)

dropdata[categorical_col] = enc.transform(dropdata[categorical_col])
xtrain, xtest, ytrain, ytest = train_test_split(dropdata, y, train_size=0.9, test_size=0.1)
xgmodel = XGBRegressor(n_estimators=1000)



xgmodel.fit(xtrain, ytrain)



xgPreds = xgmodel.predict(xtest)



print('The Mean Accuracy for XGBoost Regressor model is,', 100 - mean_absolute_percentage_error(ytest, xgPreds), '%')