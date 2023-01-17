# import libraries for data analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# list data files that are connected to the kernel
import os
os.listdir('../input/')
# read the train.csv file into a datframe
df_train = pd.read_csv('../input/train.csv')
print('Shape: ', df_train.shape)
df_train.head()
# read the test.csv file into a datframe
df_test = pd.read_csv('../input/test.csv')
print('Shape: ', df_test.shape)
df_test.head()
# number of each type of column
df_train.dtypes.value_counts()
print('There are', df_train.isnull().any().sum(), 'missing values in df_train')
print('There are', df_test.isnull().any().sum(), 'missing values in df_test')
zeros_train = df_train[df_train != 0].count(axis=0)/len(df_train.index)
zeros_train = zeros_train.to_frame()

print('There are', len(zeros_train[zeros_train[0] < 0.01]), 'columns in df_train where the amount of non-zero values is less than/equal to 1%')
zeros_train.hist(bins=50, figsize=(10,5));
zeros_test = df_train[df_test != 0].count(axis=0)/len(df_test.index)
zeros_test = zeros_test.to_frame()

print('There are', len(zeros_test[zeros_test[0] <= 0.01]), 'columns in df_test where the amount of non-zero values is less than/equal to 1%')
zeros_test.hist(bins=50, figsize=(10,5));
# create a copy of the dataframes
train = df_train.copy()
test = df_test.copy()

# extract the labels
train_labels = df_train['target']

# log transform the labels:
train_labels_log = np.log1p(train_labels)

# remove target and ID columns 
train = train.drop(['target', 'ID'], axis=1)
test = test.drop(['ID'], axis=1)

# align train and test data, keep only columns present in both dataframes
train, test = train.align(test, join = 'inner', axis = 1)

print('Train shape: ', train.shape)
print('Test shape: ', test.shape)
# import libraries
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

# define the number number of splits, alphas
kfolds = KFold(n_splits=5, random_state=42)

# define a function to calculate the RMSLE (root mean squared logarithmic error) for a given model
def rmsle_cv(model):
    rmsle = np.sqrt((np.log(-cross_val_score(model, train, train_labels_log, scoring="neg_mean_squared_error", cv = kfolds))))
    return(rmsle)
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMRegressor

lgbm_model = Pipeline([
    ('scl', RobustScaler()),
    ('lightgbm', LGBMRegressor(objective='regression',
                               n_estimators=450,
                               learning_rate=0.01))
    ]).fit(train, train_labels_log)

rmsle_cv(lgbm_model).mean()
# predict on the test data
preds = lgbm_model.predict(test)

preds = np.expm1(preds)
# make a submission dataframe
submit = df_test.loc[:, ['ID']]
submit.loc[:, 'target'] = preds

# Save the submission dataframe
submit.to_csv('submission.csv', index = False)