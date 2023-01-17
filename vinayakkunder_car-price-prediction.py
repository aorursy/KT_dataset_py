import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import PIL

import tensorflow.keras.backend as K

from tqdm import tqdm

import os

import seaborn as sns
df = pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')
df.head()
df.drop('Car_Name', inplace=True,axis=1)
df.head()
print(df['Seller_Type'].unique())

print(df['Transmission'].unique())

print(df['Owner'].unique())
df.isnull().sum()
df.describe()
df['Total_years'] = 2020-df['Year']

df.head()
dataset = pd.get_dummies(df, drop_first=True)

dataset.head()
dataset.corr()
features_mat = dataset.corr()

plt.figure(figsize=(10,10))

g = sns.heatmap(features_mat,annot=True)
dataset.drop('Year',inplace=True,axis=1)
X = dataset.iloc[:,1:]

Y = dataset.iloc[:,0]
X.head()
Y.head()
Y.shape
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

model = ExtraTreesRegressor()

model.fit(X,Y)
print(model.feature_importances_)

x_vals, y_vals = list(X.columns.values), model.feature_importances_

plt.bar(x_vals, y_vals)

y_pos = range(len(x_vals))

plt.xticks(y_pos, x_vals, rotation=90)

plt.show()
#plot max import features sorted

feature_imp = pd.Series(model.feature_importances_,index=X.columns)

feature_imp.nlargest(5).plot(kind='barh')

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)
reggressor = RandomForestRegressor()
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

# max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10, 15, 100]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 5, 10]
# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}



print(random_grid)
# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations

rf_random = RandomizedSearchCV(estimator = reggressor, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,Y_train)
rf_random.best_params_
rf_random.best_score_
pred = rf_random.predict(X_test)
sns.distplot(pred-Y_test)
plt.scatter(pred,Y_test)
from sklearn import metrics

print("MSE:",metrics.mean_squared_error(Y_test, pred))

# print("accuracy_score:",metrics.(Y_test, pred))
import pickle

op = open('random_forest_reg_model.pkl','wb')

pickle.dump(rf_random, op)