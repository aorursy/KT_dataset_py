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
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
train.head()
# check data types
print(train.dtypes.value_counts())
print(train.select_dtypes('object').columns)
train.drop(['ID'],axis=1,inplace=True)
test.drop(['ID'],axis=1,inplace=True)
train.head()
test.head()
print(train.shape)
print(test.shape)
# check missing value
train.isnull().any().sum()
train=pd.read_feather('san_train.feather')
target=train.target
target.describe()
train.to_feather('san_train.feather')
# check correlation of each column with the target
cor=train.corr()['target'].sort_values()
print(cor.head())
print(cor.tail())
cor.tail(270)
# drop unrelated columns
cor_reset=cor.reset_index(level=0)
cor_reset.dropna(inplace=True)
cor_reset.head()
# drop week related columns
# * .00-.19 “very weak”
# *  .20-.39 “weak”
# *  .40-.59 “moderate”
# *  .60-.79 “strong”
# * .80-1.0 “very strong”
cor_reset=cor_reset.loc[(cor_reset.target >= 0.2) & (cor_reset.target != 1.0)]
# cor_reset.shape
cor_reset.head()
cor_reset.shape
cols_remain=cor_reset['index']
len(cols_remain)
train_data=train[cols_remain]
test_data=test[cols_remain]
train_data.head()
# save to feather
train_data.to_feather('san_train_data.feather')
test_data.to_feather('san_test_data.feather')
train_df=pd.read_feather('san_train_data.feather')
test_df=pd.read_feather('san_test_data.feather')
# use pca
from sklearn.decomposition import PCA
pca=PCA(0.95)
pca.fit(train_df)
train_pca=pca.transform(train_df)
test_pca=pca.transform(test_df)
print(train_pca.shape)
print(test_pca.shape)
target.hist()
target=np.log1p(target)
target.hist()
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.grid_search import GridSearchCV
# param_grid = {'polynomialfeatures__degree': np.arange(4),
#               'linearregression__fit_intercept': [True, False],
#               'linearregression__normalize': [True, False]}
# def PolynomialRegression(degree=2, **kwargs):
#     return make_pipeline(PolynomialFeatures(degree),
#                          LinearRegression(**kwargs))
# grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)
# grid.fit(train_pca,target)
# grid.best_params_
from sklearn.cross_validation import cross_val_score
from xgboost import XGBRegressor
model=XGBRegressor()
scores=cross_val_score(model, train_df, target, cv=5)
scores
train_t=train.drop(['target'],axis=1)
train_t.head()
# we have got low score, let's see what will be the score without pca
my_model=XGBRegressor()
scores=cross_val_score(model, train_t, target, cv=5)
scores
