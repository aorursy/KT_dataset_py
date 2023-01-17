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
train = "../input/home-data-for-ml-course/train.csv"

train_data = pd.read_csv(train)
train_data.head()
train_data.shape
train_data.isnull().sum()
null_pc = (train_data.isnull().sum()/train_data.shape[0])*100

null_pc[null_pc > 15]
train_data = train_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1)
train_data.shape
train_data.info()
null_train = [cols for cols in train_data.columns if train_data[cols].isnull().any()]
train_data[null_train].dtypes
train_data.info()
numerical = train_data[null_train].select_dtypes('float64')
numerical.isnull().sum()
numerical.head()
train_data = train_data.drop(numerical, axis = 1)
for cols in numerical:

    mean = numerical[cols].mean()

    print("Mean of "+cols+" : "+str(mean))
for cols in numerical:

    mean = numerical[cols].mean()

    numerical[cols] = numerical[cols].fillna(mean)
numerical.isnull().sum()
train_data.shape
train_data = pd.concat([train_data, numerical], axis = 1)
train_data.shape
train_data.info()
categorical = train_data[null_train].select_dtypes('object')

categorical
categorical.isnull().sum()
train_data = train_data.drop(categorical, axis = 1)
train_data.shape
for vals in categorical:

    mode = categorical[vals].mode()

    print("mode "+vals+" : "+ str(mode))
for cols in categorical:

    mode = categorical[cols].mode()

    categorical[cols] = categorical[cols].fillna(mode[0])
categorical.isnull().sum()
train_data = pd.concat([train_data, categorical], axis = 1)
train_data.shape
train_data.info()
train_data.head()
from sklearn.preprocessing import LabelEncoder
cols = train_data.dtypes == object

object_cols = list(cols[cols].index)

object_cols
Le = LabelEncoder()
for cols in object_cols:

    train_data[cols] = Le.fit_transform(train_data[cols])
train_data.head()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
corr = train_data.corr()

corr
corr
corr['SalePrice'].sort_values(ascending = False).head(20)
features = corr['SalePrice'].sort_values(ascending = False).head(20).index
features
final_data = train_data[features]
final_data.head()
X = final_data.drop('SalePrice', axis = 1)
y = final_data.SalePrice
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.2, random_state = 0)
rf = RandomForestRegressor()

rf_model = rf.fit(train_X, train_y)
val_preds = rf_model.predict(val_X)

mae = mean_absolute_error(val_preds, val_y)

mae
test = "../input/home-data-for-ml-course/test.csv"

test_data = pd.read_csv(test)
features_test = features.drop('SalePrice')

test_X = test_data[features_test]
test_X.head()
test_X.isnull().sum()
test_X.dtypes
test_X = test_X.fillna(test_X.mean())

test_X.isnull().sum()
test_X["Foundation"] = test_X['Foundation'].astype("object")
test_X["Foundation"].value_counts()
test_X['Foundation'] = test_X["Foundation"].map({"PConc" : 1, "CBlock" : 2, "BrkTil" : 3, "Slab" : 4, "Stone" : 5, "Wood" : 6})
test_X.head()
test_preds = rf_model.predict(test_X)

test_preds
output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)