# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

from heapq import nlargest 

import scipy.stats as stats

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns; sns.set()

from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

import lightgbm as lgb

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import warnings

warnings.filterwarnings('ignore')

pd.set_option('max_columns', 105)
train = pd.read_csv("../input/user-analyze-train.csv")
train.columns
train.head()
train['Product'].describe()
train.describe()
plt.subplots(figsize=(10, 8))

plt.xticks(rotation=90)

sns.boxplot(x='Product', y="Female", data=train)
plt.subplots(figsize=(14, 8))

plt.xticks(rotation=90)

sns.boxplot(x='Mobile', y="Product", data=train)
plt.subplots(figsize=(14, 8))

plt.xticks(rotation=90)

sns.boxplot(x='Product', y="Female", data=train)
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat)
numerical_feats = train.dtypes[train.dtypes != "object"].index

print("Number of Numerical features: ", len(numerical_feats))



categorical_feats = train.dtypes[train.dtypes == "object"].index

print("Number of Categorical features: ", len(categorical_feats))
print(train[numerical_feats].columns)

print("*"*100)

print(train[categorical_feats].columns)

train[numerical_feats].head()
train[categorical_feats].head()
train["Age"].value_counts()
train["Female"].value_counts()
train.columns
test = pd.get_dummies(train)
test.shape