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
dataset = pd.read_csv('../input/kc_house_data.csv')
dataset.head()
dataset.info()
dataset["price"].value_counts()
dataset.describe()
%matplotlib inline
from matplotlib import *
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['figure.figsize'] = 20, 15

dataset.hist(bins=50)
plt.show()
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(dataset, test_size=0.25, random_state=42)
print(len(train_set), "train +", len(test_set), "test")
corr = dataset.corr()
corr["price"].sort_values(ascending=False)
# correlation scatter plot between price and sqft_living
dataset.plot(kind="scatter", x="sqft_living", y="price", alpha=0.1)
# We need to convert the categorical label date into a numeric one
# so we use a label encoder on the 'date' feature
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
dataset_cat = dataset[["date"]]
dataset_cat_encoded = encoder.fit_transform(dataset_cat)
dataset_cat_encoded

print(encoder.classes_)
# OrdinalEncoder converts encoder to 1d array, which is necessary for OneHotEncoder
# to be used. OneHotEncoder used as ML algs assume numbers close together are 
# inherently similar. However, for categorical attr's this isnt true
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
dataset_cat_encoded = ordinal_encoder.fit_transform(dataset_cat)
dataset_cat_encoded[:10]
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder(sparse=False)
dataset_cat_1hot = cat_encoder.fit_transform(dataset_cat)
dataset_cat_1hot
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


dataset = train_set.drop("price", axis=1)
dataset_labels = train_set["price"].copy()
dataset_num = dataset.drop("date", axis=1)    

num_attribs = list(dataset_num)
cat_attribs = ["date"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', OneHotEncoder(sparse=False))
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)
])

dataset_prep = full_pipeline.fit_transform(dataset)
dataset_prep
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(dataset_prep, dataset_labels)
from sklearn.metrics import mean_squared_error

dataset_predictions = lin_reg.predict(dataset_prep)
lin_mse = mean_squared_error(dataset_labels, dataset_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()

forest_reg.fit(dataset_prep, dataset_labels)

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, 
                           scoring='neg_mean_squared_error')

grid_search.fit(dataset_prep, dataset_labels)
dataset_predictions = forest_reg.predict(dataset_prep)
forest_mse = mean_squared_error(dataset_labels, dataset_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
from sklearn.svm import SVR

param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=4)

grid_search.fit(dataset_prep, dataset_labels)
