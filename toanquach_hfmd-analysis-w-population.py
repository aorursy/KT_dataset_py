import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
#TODO: Statistical checks
# Conditional Heteroskedasticity occurs when the error terms (the difference between a predicted value by a regression and the real value) are dependent on the data — for example, the error terms grow when the data point (along the x-axis) grow.
# Multicollinearity is when error terms (also called residuals) depend on each other.
# Serial correlation is when one data (feature) is a formula (or completely depends) of another feature.

# Feature importance with XGBoost
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
hfmd_data = pd.read_csv('/kaggle/input/hfmd-vietnam/hfmd_data_analysis/hfmd_data_analysis/hfmd_data_province.csv', header=0, names=['PROVINCE', 'MONTH', 'YEAR', 'TOTAL_CASES'])

climate_data = pd.read_csv('/kaggle/input/hfmd-vietnam/hfmd_data_analysis/hfmd_data_analysis/climate_per_year_per_month.csv')
climate_data.drop(climate_data.columns[0], axis=1, inplace=True)

social_data = pd.read_csv('/kaggle/input/hfmd-vietnam/hfmd_data_analysis/hfmd_data_analysis/social_data.csv', delimiter=',')
social_data
data = pd.merge(hfmd_data, climate_data, how='inner', on=['PROVINCE', 'MONTH', 'YEAR'])
data = pd.merge(social_data[['PROVINCE', 'YEAR', 'POPULATION_DENSITY']], data, how='inner', on=['PROVINCE', 'YEAR'])
feature_columns = ['PRECTOT',	'PS',	'QV2M',	'RH2M', 'T2M', 'T2MWET', 'T2M_MAX', 'T2M_MIN',	'T2M_RANGE', 'TS', 'WS10M',	'WS10M_MAX',	'WS10M_MIN',	'WS10M_RANGE',	'WS50M',	'WS50M_MAX',	'WS50M_MIN', 'WS50M_RANGE', 'POPULATION_DENSITY']
target = data['TOTAL_CASES']
feature_data = data[feature_columns]
corr = data.corr()

plt.figure(figsize=(15, 8))

ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
data['PROVINCE'].unique()
total_monthly_record = []
for i in range(1, 13):
    total_monthly_record.append(data[(data['YEAR'] == 2017) & (data['MONTH'] == i)]['TOTAL_CASES'].sum())
    
fig, ax = plt.subplots(figsize=(12, 9))
sns.barplot(list(range(1, 13)), total_monthly_record)
total_monthly_record = []
for i in range(1, 13):
    total_monthly_record.append(data[(data['YEAR'] == 2018) & (data['MONTH'] == i)]['TOTAL_CASES'].sum())
    
fig, ax = plt.subplots(figsize=(12, 9))
sns.barplot(list(range(1, 13)), total_monthly_record)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
X_train, X_test, y_train, y_test = train_test_split(feature_data, target, train_size=0.8, stratify=data['PROVINCE'], random_state=42)
feature_scaler = StandardScaler()
feature_scaler.fit(X_train)

target_scaler = StandardScaler()
target_scaler.fit(np.array(y_train).reshape((-1, 1)))

X_train_preprocess = feature_scaler.transform(X_train)
X_test_preprocess = feature_scaler.transform(X_test)

y_train_preprocess = target_scaler.transform(np.array(y_train).reshape((-1, 1)))
y_test_preprocess = target_scaler.transform(np.array(y_test).reshape((-1, 1)))
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
def mean_absolute_percentage_error(y_true, y_pred):
  
  '''
    Calculate mean absolute percentage error loss
  '''

  # y_true, y_pred = check_arrays(y_true, y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
linear = LinearRegression()
parameters = {}
linear_grid_search_cv = GridSearchCV(linear, parameters, scoring=('neg_mean_absolute_error'), cv=5, verbose=5, n_jobs=-1)
linear_grid_search_cv.fit(X_train_preprocess, y_train_preprocess)
print(f'Best score (MAE): {linear_grid_search_cv.best_score_}')
print(f'Best parameters: {linear_grid_search_cv.best_params_}')
linear_best = linear_grid_search_cv.best_estimator_
coef = linear_best.coef_

fig, ax = plt.subplots(figsize=(12, 9))
sns.barplot(coef[0], X_train.columns)
y_pred = linear_best.predict(X_test_preprocess)

mae = mean_absolute_error(target_scaler.inverse_transform(y_test_preprocess), target_scaler.inverse_transform(y_pred))

mape = mean_absolute_percentage_error(target_scaler.inverse_transform(y_test_preprocess), target_scaler.inverse_transform(y_pred))

print(f'MAE loss:{mae}')
print(f'MAPE loss:{mape}')
tree = DecisionTreeRegressor()
parameters = {'criterion': ['mse', 'mae'], 
              'max_depth': [2, 5, 8, 16, 20],
              'min_samples_split': [2, 4]}
tree_grid_search_cv = GridSearchCV(tree, parameters, scoring=('neg_mean_absolute_error'), cv=5, verbose=5, n_jobs=-1)
tree_grid_search_cv.fit(X_train_preprocess, y_train_preprocess)
print(f'Best score (MAE): {tree_grid_search_cv.best_score_}')
print(f'Best parameters: {tree_grid_search_cv.best_params_}')
tree_best = tree_grid_search_cv.best_estimator_
feature_importances = tree_best.feature_importances_

fig, ax = plt.subplots(figsize=(12, 9))
sns.barplot(feature_importances, X_train.columns)
y_pred = tree_best.predict(X_test_preprocess)

mae = mean_absolute_error(target_scaler.inverse_transform(y_test_preprocess), target_scaler.inverse_transform(y_pred))

mape = mean_absolute_percentage_error(target_scaler.inverse_transform(y_test_preprocess), target_scaler.inverse_transform(y_pred))

print(f'MAE loss:{mae}')
print(f'MAPE loss:{mape}')
forest = RandomForestRegressor(n_jobs=-1)
parameters = {'n_estimators': [10, 30, 50, 100, 500], 
              'max_depth': [2, 5, 8, 10, 15],
              'min_samples_split': [2, 4]}
forest_grid_search_cv = GridSearchCV(forest, parameters, scoring=('neg_mean_absolute_error'), cv=5, verbose=5, n_jobs=-1)
forest_grid_search_cv.fit(X_train_preprocess, y_train_preprocess)
print(f'Best score (MAE): {forest_grid_search_cv.best_score_}')
print(f'Best parameters: {forest_grid_search_cv.best_params_}')
forest_best = forest_grid_search_cv.best_estimator_
feature_importances = forest_best.feature_importances_

fig, ax = plt.subplots(figsize=(12, 9))
sns.barplot(feature_importances, X_train.columns)
y_pred = forest_best.predict(X_test_preprocess)

mae = mean_absolute_error(target_scaler.inverse_transform(y_test_preprocess), target_scaler.inverse_transform(y_pred))

mape = mean_absolute_percentage_error(target_scaler.inverse_transform(y_test_preprocess), target_scaler.inverse_transform(y_pred))

print(f'MAE loss:{mae}')
print(f'MAPE loss:{mape}')
feature_columns = ['PRECTOT',	'PS',	'QV2M',	'RH2M', 'T2M', 'T2MWET', 'T2M_MAX', 'T2M_MIN',	'T2M_RANGE', 'TS', 'WS10M',	'WS10M_MAX',	'WS10M_MIN',	'WS10M_RANGE',	'WS50M',	'WS50M_MAX',	'WS50M_MIN', 'WS50M_RANGE']
linear_feature_columns_2 = ['PS', 'T2M', 'T2MWET', 'T2M_MAX', 'T2M_MIN',	'T2M_RANGE', 'TS']
linear_feature_columns = ['T2M_MAX', 'TS', 'WS10M_MAX']
target = data['TOTAL_CASES']

# target = data['CASE_OVER_POPULATION']
feature_data = data[linear_feature_columns_2]
X_train, X_test, y_train, y_test = train_test_split(feature_data, target, train_size=0.8, stratify=data['PROVINCE'], random_state=42)
feature_scaler = StandardScaler()
feature_scaler.fit(X_train)

target_scaler = StandardScaler()
target_scaler.fit(np.array(y_train).reshape((-1, 1)))

X_train_preprocess = feature_scaler.transform(X_train)
X_test_preprocess = feature_scaler.transform(X_test)

y_train_preprocess = target_scaler.transform(np.array(y_train).reshape((-1, 1)))
y_test_preprocess = target_scaler.transform(np.array(y_test).reshape((-1, 1)))
linear = LinearRegression()
parameters = {}
linear_grid_search_cv = GridSearchCV(linear, parameters, scoring=('neg_mean_absolute_error'), cv=5, verbose=5, n_jobs=-1)
linear_grid_search_cv.fit(X_train_preprocess, y_train_preprocess)
print(f'Best score (MAE): {linear_grid_search_cv.best_score_}')
print(f'Best parameters: {linear_grid_search_cv.best_params_}')
linear_best = linear_grid_search_cv.best_estimator_
coef = linear_best.coef_

fig, ax = plt.subplots(figsize=(12, 9))
sns.barplot(coef[0], X_train.columns)
y_pred = linear_best.predict(X_test_preprocess)

mae = mean_absolute_error(target_scaler.inverse_transform(y_test_preprocess), target_scaler.inverse_transform(y_pred))

mape = mean_absolute_percentage_error(target_scaler.inverse_transform(y_test_preprocess), target_scaler.inverse_transform(y_pred))

print(f'MAE loss:{mae}')
print(f'MAPE loss:{mape}')
tree = DecisionTreeRegressor()
parameters = {'criterion': ['mse', 'mae'], 
              'max_depth': [2, 5, 8, 10],
              'min_samples_split': [2, 4]}
tree_grid_search_cv = GridSearchCV(tree, parameters, scoring=('neg_mean_absolute_error'), cv=5, verbose=5, n_jobs=-1)
tree_grid_search_cv.fit(X_train_preprocess, y_train_preprocess)
print(f'Best score (MAE): {tree_grid_search_cv.best_score_}')
print(f'Best parameters: {tree_grid_search_cv.best_params_}')
tree_best = tree_grid_search_cv.best_estimator_
feature_importances = tree_best.feature_importances_

fig, ax = plt.subplots(figsize=(12, 9))
sns.barplot(feature_importances, X_train.columns)
y_pred = tree_best.predict(X_test_preprocess)

mae = mean_absolute_error(target_scaler.inverse_transform(y_test_preprocess), target_scaler.inverse_transform(y_pred))

mape = mean_absolute_percentage_error(target_scaler.inverse_transform(y_test_preprocess), target_scaler.inverse_transform(y_pred))

print(f'MAE loss:{mae}')
print(f'MAPE loss:{mape}')
forest = RandomForestRegressor(n_jobs=-1)
parameters = {'n_estimators': [10, 30, 50, 100, 500, 1000],
              'max_depth': [2, 5, 8, 10, 15],
              'min_samples_split': [2, 4]}
forest_grid_search_cv = GridSearchCV(forest, parameters, scoring=('neg_mean_absolute_error'), cv=5, verbose=5, n_jobs=-1)
forest_grid_search_cv.fit(X_train_preprocess, y_train_preprocess)
print(f'Best score (MAE): {forest_grid_search_cv.best_score_}')
print(f'Best parameters: {forest_grid_search_cv.best_params_}')
forest_best = forest_grid_search_cv.best_estimator_
feature_importances = forest_best.feature_importances_

fig, ax = plt.subplots(figsize=(12, 9))
sns.barplot(feature_importances, X_train.columns)
y_pred = forest_best.predict(X_test_preprocess)

mae = mean_absolute_error(target_scaler.inverse_transform(y_test_preprocess), target_scaler.inverse_transform(y_pred))

mape = mean_absolute_percentage_error(target_scaler.inverse_transform(y_test_preprocess), target_scaler.inverse_transform(y_pred))

print(f'MAE loss:{mae}')
print(f'MAPE loss:{mape}')
feature_columns = ['PRECTOT',	'PS',	'QV2M', 'T2MWET', 'T2M_MAX', 'T2M_MIN',	'T2M_RANGE', 'TS', 'WS10M_MAX',	'WS10M_MIN',	'WS10M_RANGE', 'WS50M_RANGE']
linear_feature_columns_2 = ['PS', 'T2M', 'T2MWET', 'T2M_MAX', 'T2M_MIN',	'T2M_RANGE', 'TS']
linear_feature_columns = ['T2M_MAX', 'TS', 'WS10M_MAX']
target = data['TOTAL_CASES']

feature_data = data[feature_columns]
X_train, X_test, y_train, y_test = train_test_split(feature_data, target, train_size=0.8, stratify=data['PROVINCE'], random_state=42)
feature_scaler = StandardScaler()
feature_scaler.fit(X_train)

target_scaler = StandardScaler()
target_scaler.fit(np.array(y_train).reshape((-1, 1)))

X_train_preprocess = feature_scaler.transform(X_train)
X_test_preprocess = feature_scaler.transform(X_test)

y_train_preprocess = target_scaler.transform(np.array(y_train).reshape((-1, 1)))
y_test_preprocess = target_scaler.transform(np.array(y_test).reshape((-1, 1)))
linear = LinearRegression()
parameters = {}
linear_grid_search_cv = GridSearchCV(linear, parameters, scoring=('neg_mean_absolute_error'), cv=5, verbose=5, n_jobs=-1)
linear_grid_search_cv.fit(X_train_preprocess, y_train_preprocess)
print(f'Best score (MAE): {linear_grid_search_cv.best_score_}')
print(f'Best parameters: {linear_grid_search_cv.best_params_}')
linear_best = linear_grid_search_cv.best_estimator_
coef = linear_best.coef_

fig, ax = plt.subplots(figsize=(12, 9))
sns.barplot(coef[0], X_train.columns)
y_pred = linear_best.predict(X_test_preprocess)

mae = mean_absolute_error(target_scaler.inverse_transform(y_test_preprocess), target_scaler.inverse_transform(y_pred))

mape = mean_absolute_percentage_error(target_scaler.inverse_transform(y_test_preprocess), target_scaler.inverse_transform(y_pred))

print(f'MAE loss:{mae}')
print(f'MAPE loss:{mape}')
tree = DecisionTreeRegressor()
parameters = {'criterion': ['mse', 'mae'], 
              'max_depth': [2, 5, 8, 10],
              'min_samples_split': [2, 4]}
tree_grid_search_cv = GridSearchCV(tree, parameters, scoring=('neg_mean_absolute_error'), cv=5, verbose=5, n_jobs=-1)
tree_grid_search_cv.fit(X_train_preprocess, y_train_preprocess)
print(f'Best score (MAE): {tree_grid_search_cv.best_score_}')
print(f'Best parameters: {tree_grid_search_cv.best_params_}')
tree_best = tree_grid_search_cv.best_estimator_
feature_importances = tree_best.feature_importances_

fig, ax = plt.subplots(figsize=(12, 9))
sns.barplot(feature_importances, X_train.columns)
y_pred = tree_best.predict(X_test_preprocess)

mae = mean_absolute_error(target_scaler.inverse_transform(y_test_preprocess), target_scaler.inverse_transform(y_pred))

mape = mean_absolute_percentage_error(target_scaler.inverse_transform(y_test_preprocess), target_scaler.inverse_transform(y_pred))

print(f'MAE loss:{mae}')
print(f'MAPE loss:{mape}')
forest = RandomForestRegressor(n_jobs=-1)
parameters = {'n_estimators': [10, 30, 50, 100, 500, 1000],
              'max_depth': [2, 5, 8, 10, 15],
              'min_samples_split': [2, 4]}
forest_grid_search_cv = GridSearchCV(forest, parameters, scoring=('neg_mean_absolute_error'), cv=5, verbose=5, n_jobs=-1)
forest_grid_search_cv.fit(X_train_preprocess, y_train_preprocess)
print(f'Best score (MAE): {forest_grid_search_cv.best_score_}')
print(f'Best parameters: {forest_grid_search_cv.best_params_}')
forest_best = forest_grid_search_cv.best_estimator_
feature_importances = forest_best.feature_importances_

fig, ax = plt.subplots(figsize=(12, 9))
sns.barplot(feature_importances, X_train.columns)
y_pred = forest_best.predict(X_test_preprocess)

mae = mean_absolute_error(target_scaler.inverse_transform(y_test_preprocess), target_scaler.inverse_transform(y_pred))

mape = mean_absolute_percentage_error(target_scaler.inverse_transform(y_test_preprocess), target_scaler.inverse_transform(y_pred))

print(f'MAE loss:{mae}')
print(f'MAPE loss:{mape}')

data['PROVINCE'].unique()
PROVINCE = 'BENTRE'

# feature_columns = ['PRECTOT', 'PS', 'QV2M', 'RH2M', 'T2M', 'T2MWET', 'WS10M', 'WS50M']
feature_columns = ['PRECTOT', 'PS', 'QV2M', 'RH2M', 'T2M', 'T2MWET', 'WS10M']
target = data['TOTAL_CASES']

# target = data['CASE_OVER_POPULATION']
feature_data = data[feature_columns]
province_feature = feature_data[data['PROVINCE'] == PROVINCE]
province_target = target[data['PROVINCE'] == PROVINCE]
data[data['PROVINCE'] == PROVINCE]
feature_scaler = StandardScaler()
province_feature_preprocess = feature_scaler.fit_transform(province_feature)
linear = LinearRegression()
parameters = {}
linear_grid_search_cv = GridSearchCV(linear, parameters, scoring=('neg_mean_absolute_error'), cv=5, verbose=5, n_jobs=-1)
linear_grid_search_cv.fit(province_feature_preprocess, province_target)
print(f'Best score (MAE): {linear_grid_search_cv.best_score_}')
print(f'Best parameters: {linear_grid_search_cv.best_params_}')
linear_best = linear_grid_search_cv.best_estimator_
coef = linear_best.coef_

fig, ax = plt.subplots(figsize=(12, 9))
sns.barplot(coef, province_feature.columns)
tree = DecisionTreeRegressor()
parameters = {'criterion': ['mse', 'mae'], 
              'max_depth': [2, 5, 8, 10],
              'min_samples_split': [2, 4]}
tree_grid_search_cv = GridSearchCV(tree, parameters, scoring=('neg_mean_absolute_error'), cv=5, verbose=5, n_jobs=-1)
tree_grid_search_cv.fit(province_feature_preprocess, province_target)
print(f'Best score (MAE): {tree_grid_search_cv.best_score_}')
print(f'Best parameters: {tree_grid_search_cv.best_params_}')
tree_best = tree_grid_search_cv.best_estimator_
feature_importances = tree_best.feature_importances_

fig, ax = plt.subplots(figsize=(12, 9))
sns.barplot(feature_importances, province_feature.columns)
forest = RandomForestRegressor(n_jobs=-1)
parameters = {'n_estimators': [10, 30, 50, 100, 500, 1000],
              'max_depth': [2, 5, 8, 10, 15],
              'min_samples_split': [2, 4]}
forest_grid_search_cv = GridSearchCV(forest, parameters, scoring=('neg_mean_absolute_error'), cv=5, verbose=5, n_jobs=-1)
forest_grid_search_cv.fit(province_feature_preprocess, province_target)
print(f'Best score (MAE): {forest_grid_search_cv.best_score_}')
print(f'Best parameters: {forest_grid_search_cv.best_params_}')
forest_best = forest_grid_search_cv.best_estimator_
feature_importances = forest_best.feature_importances_

fig, ax = plt.subplots(figsize=(12, 9))
sns.barplot(feature_importances, province_feature.columns)

