# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
import scipy.stats as stats
%matplotlib inline

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor


from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('../input/train.csv')
Y = data['SalePrice']
features_raw = data.drop('SalePrice', axis = 1)
# Drop features with many NA
features_raw.drop(['Alley','MiscFeature','PoolQC','Fence', 'FireplaceQu'], axis = 1, inplace = True)
# Split features into categorical and numerical data
num_var = list(features_raw.select_dtypes(include=['int64', 'float64']).columns)
cat_var = list(features_raw.select_dtypes(include=['object']).columns)

data_cat = features_raw[cat_var]
data_num = features_raw[num_var]
# fill nan with most frequent and mean
data_cat = data_cat.apply(lambda x:x.fillna(x.value_counts().index[0]))
data_num = data_num.fillna(data_num.mean())
seed = 12345
num_folds = 5

# Cross Validation Linear Regression R^2
kfold = KFold(n_splits=10, random_state=7)
model = LinearRegression()
scoring = 'r2'
results = cross_val_score(model, data_num, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)" % (results.mean(), results.std()))
# Cross Validation GBM Regression R^2
kfold = KFold(n_splits=10, random_state=7)
model = GradientBoostingRegressor()
scoring = 'r2'
results = cross_val_score(model, data_num, Y, cv=kfold, scoring=scoring)
print("R^2: %.3f (%.3f)" % (results.mean(), results.std()))
test_data = pd.read_csv('../input/test.csv')
test_data.drop(['Alley','MiscFeature','PoolQC','Fence', 'FireplaceQu'], axis = 1, inplace = True)

num_var = list(features_raw.select_dtypes(include=['int64', 'float64']).columns)
cat_var = list(features_raw.select_dtypes(include=['object']).columns)

test_cat = test_data[cat_var]
test_num = test_data[num_var]
test_cat = test_cat.apply(lambda x:x.fillna(x.value_counts().index[0]))
test_cat_ohe = pd.get_dummies(test_cat)

test_num = test_num.fillna(test_num.mean())

X_test = test_num.join(test_cat_ohe)
# Prepare the model with the best found data transforms, algorithm and hyperparameters on the entire training dataset
# Regression problem
scaler = StandardScaler().fit(data_num)
rescaledX = scaler.transform(data_num)
model = GradientBoostingRegressor()
model.fit(rescaledX, Y)
# transform the validation dataset and validate with model

rescaledValidationX = scaler.transform(test_num)
predictions = model.predict(rescaledValidationX)

