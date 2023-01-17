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
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
train_data = pd.read_csv('../input/train.csv')
print(train_data.shape)
train_data.head()
test_data = pd.read_csv('../input/test.csv')
print(test_data.shape)
test_data.head()
trainId = train_data['Id']
testId = test_data['Id']
train_data_1 = train_data.drop(['Id', 'idhogar'], axis=1)
test_data_1 = test_data.drop(['Id', 'idhogar'], axis=1)
train_data_1 = pd.get_dummies(train_data_1)
test_data_1 = pd.get_dummies(test_data_1)
train_data_1['Id'] = trainId
test_data_1['Id'] = testId
train_label = train_data_1['Target']
train_data_1, test_data_1 = train_data_1.align(test_data_1, join='inner', axis=1)
train_data_1['Target'] = train_label
print(train_data_1.shape)
print(test_data_1.shape)
poly_features = train_data_1[['escolari', 'cielorazo', 'meaneduc', 'hogar_nin', 'r4t1', 'SQBhogar_nin', 'Target']]
poly_features_test = test_data_1[['escolari', 'cielorazo', 'meaneduc', 'hogar_nin', 'r4t1', 'SQBhogar_nin']]
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median')
poly_target = poly_features['Target']
poly_features = poly_features.drop(columns=['Target'])
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)
from sklearn.preprocessing import PolynomialFeatures
poly_transformer = PolynomialFeatures(degree=3)
poly_transformer.fit(poly_features)
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print(poly_features.shape)
poly_transformer.get_feature_names(['escolari', 'cielorazo', 'meaneduc', 'hogar_nin', 'r4t1', 'SQBhogar_nin'])[:15]
poly_features = pd.DataFrame(poly_features, columns=poly_transformer.get_feature_names(['escolari', 'cielorazo', 'meaneduc', 'hogar_nin', 'r4t1', 'SQBhogar_nin']))
poly_features['Target'] = poly_target
poly_corrs = poly_features.corr()['Target'].sort_values()
poly_features_test = pd.DataFrame(poly_features_test, columns=poly_transformer.get_feature_names(['escolari', 'cielorazo', 'meaneduc', 'hogar_nin', 'r4t1', 'SQBhogar_nin']))
poly_features['Id'] = train_data_1['Id']
train_data_poly = train_data_1.merge(poly_features, on='Id', how='left')
poly_features_test['Id'] = test_data_1['Id']
test_data_poly = test_data_1.merge(poly_features_test, on='Id', how='left')
train_data_poly, test_data_poly = train_data_poly.align(test_data_poly, join='inner', axis=1)
print(train_data_poly.shape)
print(test_data_poly.shape)
train_data_1.drop(['Id'], axis=1, inplace=True)
test_data_1.drop(['Id'], axis=1, inplace=True)
train_data_poly.drop(['Id'], axis=1, inplace=True)
test_data_poly.drop(['Id'], axis=1, inplace=True)
imputer = Imputer(strategy='median')
poly_features = imputer.fit_transform(train_data_poly)
poly_features_test = imputer.transform(test_data_poly)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
poly_features = scaler.fit_transform(train_pca)
poly_features_test = scaler.transform(test_pcs)
# from sklearn.ensemble import RandomForestClassifier
# rf_poly = RandomForestClassifier(n_estimators=100, random_state=50, verbose=1, n_jobs=-1)
# rf_poly.fit(poly_features, train_label)
import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=100, objective='multiclass', 
                           class_weight = 'balanced', learning_rate = 0.05, 
                           reg_alpha = 0.1, reg_lambda = 0.1, 
                           subsample = 0.8, n_jobs = -1, random_state = 50)
model.fit(poly_features, train_label)
predictions = model.predict(poly_features_test)
submit = pd.read_csv('../input/sample_submission.csv')
submit['Target'] = predictions
submit.to_csv('rf_poly_pca.csv', index=False)
